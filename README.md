## Install

```shell
# Install SAPIEN dev version, example for 3.8, you can choose a different whl file for 3.7, 3.9, 3.10
pip3 install sapien>=2.1.0
```

Download data file for hand detector and scene
from [Google Drive Link](https://drive.google.com/file/d/1Xe3jgcIUZm_8yaFUsHnO7WJWr8cV41fE/view?usp=sharing).
Place the `day.ktx` at `assets/misc/ktx/day.ktx`.

```shell
gdown https://drive.google.com/uc?id=1Xe3jgcIUZm_8yaFUsHnO7WJWr8cV41fE
```

## File Structure

- `hand_teleop`: main entry for the environment, utils, and other staff needs for teleoperation and RL training.
- `assets`: robot and object models, and other static files
- `main`: entry files
