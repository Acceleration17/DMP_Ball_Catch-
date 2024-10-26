import numpy

numpy.bool8 = numpy.bool_
import sys
import pickle
import gym
import gym_thing
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import time
from dmp import DMP
from policy import BallCatchDMPPolicy

PATH_TO_HOME = str(pathlib.Path(__file__).parent.resolve())
PATH_TO_NLOPTJSON = str(
    (
        pathlib.Path(__file__).parent
        / "ballcatch_env"
        / "gym_thing"
        / "nlopt_optimization"
        / "config"
        / "nlopt_config_stationarybase.json"
    ).resolve()
)

TRAINED_DMP_PATH = "results/trained_dmp.pkl"


def make_env(
    online=True,
    dataset=None,
    n_substeps=1,
    gravity_factor_std=0.0,
    n_substeps_std=0.0,
    mass_std=0.0,
    act_gain_std=0.0,
    joint_noise_std=0.0,
    vicon_noise_std=0.0,
    control_noise_std=0.0,
    random_training_index=False,
):
    # Create thing env
    initial_arm_qpos = np.array([1, -0.3, -1, 0, 1.6, 0])
    initial_ball_qvel = np.array([-1, 1.5, 4.5])
    initial_ball_qpos = np.array([1.22, -1.6, 1.2])
    zero_base_qpos = np.array([0.0, 0.0, 0.0])

    training_data = None
    if not online:
        training_data = np.load(dataset)
        print("OFFLINE")

    env = gym.make(
        "ballcatch-v0",
        model_path="robot_with_cup.xml",
        nlopt_config_path=PATH_TO_NLOPTJSON,
        initial_arm_qpos=initial_arm_qpos,
        initial_ball_qvel=initial_ball_qvel,
        initial_ball_qpos=initial_ball_qpos,
        initial_base_qpos=zero_base_qpos,
        pos_rew_weight=0.1,
        rot_rew_weight=0.01,
        n_substeps=n_substeps,
        online=online,
        training_data=training_data,
        gravity_factor_std=gravity_factor_std,
        n_substeps_std=n_substeps_std,
        mass_std=mass_std,
        act_gain_std=act_gain_std,
        joint_noise_std=joint_noise_std,
        vicon_noise_std=vicon_noise_std,
        control_noise_std=control_noise_std,
        random_training_index=random_training_index,
    )
    env.reset()
    return env


def test_policy(eval_env, policy, eval_episodes=5, render_freq=1, seed=1):
    # Set seeds
    eval_env.seed(seed)
    np.random.seed(seed)

    avg_reward = 0.0
    successes = 0

    for eps in range(eval_episodes):
        print(f"\rEvaluating Episode {eps+1}/{eval_episodes}", end="")
        state, done, truncated = eval_env.reset(), False, False
        policy.set_goal(state=state, goal=eval_env.env.goal)
        while not (done or truncated):
            action = policy.select_action(np.array(state))
            state, reward, done, truncated, info = eval_env.step(action)
            if render_freq != 0 and eps % render_freq == 0:
                eval_env.render()
                # time.sleep(0.01)
            avg_reward += reward
        if truncated and not done:
            actual_done = False
        else:
            actual_done = done

        if actual_done:
            successes += 1

    avg_reward /= eval_episodes
    success_pct = float(successes) / eval_episodes

    print("")
    print("---------------------------------------")
    print("Evaluation over {} episodes: {:.3f}, success%: {:.3f}".format(eval_episodes, avg_reward, success_pct))
    print("---------------------------------------")
    print("")
    return avg_reward, success_pct


def load_dataset(dataset_path="data/demos.pkl"):
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    return [traj[:, :6] for traj in dataset["trajectories"]]  # 1st 6 elements are joint angle


def q2_recons():
    K_vec = 20 * np.ones(6)
    K_vec[4:] = 50  # Increase K_vec for joints 5 and 6
    dmp = DMP(nbasis=80, K_vec=K_vec)

    trajectories = load_dataset()
    dt = 0.04
    # Train a DMP on trajectories[0]
    dmp = DMP(nbasis=50)

    demo = trajectories[0]

    # Interpolate the trajectory for training
    X, T = DMP.interpolate([demo], dt)  # X: (1, length, dofs), T: (1, length)

    # Use the first timestep as the initial position and the last timestep as the goal
    x0 = X[0, 0]  # First position
    g = X[0, -1]  # Goal position

    # Now, pass X and T directly to learn()
    dmp.learn(X, T)  # X shape (1, length, 6), T shape (1, length)
    dmp.save(TRAINED_DMP_PATH)  # Save the trained DMP

    tau = (T[0, -1] - T[0, 0]).item()  # Convert to scalar using .item()

    rollout = dmp.rollout(dt=dt, tau=tau, x0=x0, g=g)  # Reconstruct the trajectory

    rollout_time = np.arange(len(rollout)) * dt

    min_length = min(len(T[0]), len(rollout_time), len(rollout))

    # Plot and compare the ground truth (demo) and reconstructed trajectory (rollout) for each joint
    for k in range(6):  # Iterate over the 6 degrees of freedom (DOF)
        plt.figure()
        plt.plot(T[0][:min_length], X[0][:min_length, k], label='Demo')  # Interpolated demo trajectory
        plt.plot(rollout_time[:min_length], rollout[:min_length, k], label='DMP')  # Reconstructed trajectory from DMP
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel(f"Joint {k+1} angle (rad)")
        plt.title(f"Reconstruction for Joint {k+1}")
        plt.savefig(f'results/recons_{k}.png')  # Save each plot to the results folder
        plt.close()



def q2_tuning():
    dt = 0.04
    trajectories = load_dataset()

    # Interpolate the trajectories to a common length
    X, T = DMP.interpolate(trajectories, dt)

    # Define values to try for nbasis and K_vec
    nbasis_values = [20, 30, 40, 50, 60]
    K_values = [1, 5, 20, 40, 60]  # scalar values

    num_demos = X.shape[0]
    num_timesteps = X.shape[1]
    num_dofs = X.shape[2]

    best_rmse = float('inf')
    best_nbasis = None
    best_K_vec = None

    # For recording the results
    results = []

    for nbasis in nbasis_values:
        for K in K_values:
            K_vec = K * np.ones(num_dofs)
            # Create a DMP with these settings
            dmp = DMP(nbasis=nbasis, K_vec=K_vec)
            # Train the DMP on the demos
            dmp.learn(X, T)
            # Compute the RMSE over all demos
            total_rmse = 0.0
            for i in range(num_demos):
                x0 = X[i, 0]  # Initial position
                g = X[i, -1]  # Goal position
                tau = (T[i, -1] - T[i, 0]).item()
                time_steps = T[i] - T[i, 0]
                # Reconstruct the trajectory using rollout2
                rollout = dmp.rollout2(time_steps=time_steps, tau=tau, x0=x0, g=g)
                # Compute RMSE
                rmse = np.sqrt(np.mean((X[i] - rollout) ** 2))
                total_rmse += rmse
            avg_rmse = total_rmse / num_demos
            results.append((nbasis, K, avg_rmse))
            print(f"nbasis={nbasis}, K={K}, avg_rmse={avg_rmse}")
            # Update best settings if RMSE is improved
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_nbasis = nbasis
                best_K_vec = K_vec

    # Print best settings
    print(f"Best settings: nbasis={best_nbasis}, K_vec={best_K_vec}, RMSE={best_rmse}")

    # Train the final DMP with the best settings
    dmp = DMP(nbasis=best_nbasis, K_vec=best_K_vec)
    dmp.learn(X, T)
    dmp.save(TRAINED_DMP_PATH)




def main():
    env = make_env(n_substeps=1)
    dmp = DMP.load(TRAINED_DMP_PATH)

    tau = 0.56 # Replace with the actual tau used during DMP training
    policy = BallCatchDMPPolicy(dmp, dt=env.dt, tau=tau)
    test_policy(env, policy, eval_episodes=20, render_freq=1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == "recons":
        q2_recons()
    elif sys.argv[1] == "tuning":
        q2_tuning()