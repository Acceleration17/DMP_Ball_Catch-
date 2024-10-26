# CSC2626 Assignment 2 - Dynamic Movement Primitives

##  Basic Setup
This project requires python >= 3.7. First, run `python3 -m venv myenv` to set up the pip environment.

*Note: For ARM-based Macs, use the Rosetta terminal and run `arch -arm64 python3 -m venv myenv` to create the environment. To run the Rosetta terminal, right-click on the Terminal app under `/Applications/Utilities/` and click on `Get Info` and select the `Open using Rosetta` option.*

To install the python requirements:
```bash
source myenv/bin/activate
pip install -r requirements.txt
pip install -e ballcatch_env/
```

### Install Mujoco:
The last thing you need is the mujoco simulator. On Linux or macOS-x86_64, you can install it as follows:

1. Download mujoco 2.1.0 from https://github.com/deepmind/mujoco/releases/tag/2.1.0
2. Install by placing the unzipped folder "mujoco210" under `~/.mujoco/`:

   a. For linux: `mkdir ~/.mujoco && tar -xvf mujoco210-linux-x86_64.tar.gz && mv mujoco210 ~/.mujoco/`

   b. For macOS-x86_64: `mkdir ~/.mujoco && tar -xvf mujoco210-macos-x86_64.tar.gz && mv mujoco210 ~/.mujoco/`

*Note: Macs with Apple sillicon (M1-M3) do not support mujoco 2.1.0. Here is a workaround from   https://github.com/openai/mujoco-py/issues/682:*
1. Install `glfw` via `brew install glfw`
2. Download MuJoCo2.1.1 image that ends with a `*.dmg` from https://github.com/google-deepmind/mujoco/releases/tag/2.1.1. `mujoco2.1.1` is released as a Framework. You can copy the `MuJoCo.app` into `/Applications/` folder.
3. Run `chmod +x install-mujoco.sh && ./install-mujoco.sh` (Note: Make sure to run the MuJoCo app once before running this command.)
## Notes

If you try to run the GUI, you may get errors telling you to set LD_LIBRARY_PATH or LD_PRELOAD. If so, this should fix it in Linux (or similar commands on Mac):

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

```
In case it does not recognize `libGLEW.so`, install it using the following commands:
```bash
sudo apt update
sudo apt-get install patchelf
sudo apt-get install python3-dev build-essential libssl-dev libffi-dev libxml2-dev
sudo apt-get install libxslt1-dev zlib1g-dev libglew-dev python3-pip
```

For other issues, please post them on Piazza.
