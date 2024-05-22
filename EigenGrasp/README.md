# Eigengrasp Utilities

## Files

- `eigengrasp.py`: Implement for Eigengrasp.
- `main.py`: Entrance for the tool.

## How to use this tool

- 1. Data Generation
    
    Generate enough grasp data in `DexGraspNet` and pack them in a single `*.npy` file, such as the `grasp_mat.npy` in this folder.
    `grasp_mat.npy` contains 60800 grasp qpos generated on 474 objects, which is large enough in most cases.

- 2. Train Eigengrasp Model

    Run the following instruction to train a model.
```bash
python main.py --mode train --data_num -1 --dim 7 --output grasp_model.pkl --loss_result loss_result.csv
```

    Or simply run the following instruction to train a model with default parameters.

```bash
python main.py --mode train --dim 7
```

    You can check results and loss in terminal or files for detailed ones.


