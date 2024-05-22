import argparse
import numpy as np
import pandas as pd
from eigengrasp import EigenGrasp
from tqdm import tqdm
np.set_printoptions(precision=3)


def random_rows(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Randomly select `n` rows from the given ndarray `arr`.

    :param arr: Input ndarray of shape (N, M)
    :param n: Number of rows to select
    :return: A new ndarray of shape (n, M) containing randomly selected rows
    """
    # 确保 n 不大于 arr 的行数
    if n > arr.shape[0]:
        raise ValueError("n should be less than or equal to the number of rows in arr.")

    # 随机选择行索引
    indices = np.random.choice(arr.shape[0], size=n, replace=False)

    return arr[indices]


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", help="train or test")
parser.add_argument("--data", type=str, default="grasp_mat.npy", help="data file to load (train mode)")
parser.add_argument("--data_num", type=int, default=-1, help="random sample data number for training, -1 means all data")
parser.add_argument("--dim", type=int, default=2, help="reduced dimension")
parser.add_argument("--output", type=str, default="grasp_model.pkl", help="model file to save (train mode)")
parser.add_argument("--model", type=str, default="grasp_model.pkl", help="model file to load (test mode)")
parser.add_argument("--num", type=int, default=50, help="number of test cases (test mode)")
parser.add_argument("--loss_result", type=str, default=None, help="result file to save (test mode)")
args = parser.parse_args()
data = np.load(args.data)
print(data.shape)
D, M = data.shape[1], args.dim
model = EigenGrasp(D, M)
if args.mode == "train":
    if args.data_num != -1:
        data = random_rows(data, args.data_num)
    N = data.shape[0]
    model.fit_joint_values(data)
    model.dump_to_file(args.output)
    results = pd.DataFrame(index=range(1, N+1, 1), columns=["mse", "rmse", "mae", "mpe", "mape"])
    with tqdm(total=N, desc="Analysing") as pbar:
        for idx in range(N):
            joint_vals = data[idx]
            reduced_vals = model._transformed_joint_angles[idx]
            restored_vals = model.compute_grasp(reduced_vals)
            mse = np.mean((restored_vals - joint_vals) ** 2)
            rmse = np.sqrt(np.mean((restored_vals - joint_vals) ** 2))
            mae = np.mean(np.abs(restored_vals - joint_vals))
            mpe = np.mean(np.abs(restored_vals - joint_vals) / np.abs(joint_vals))
            mape = np.mean(np.abs(restored_vals - joint_vals) / np.abs(joint_vals))
            results.loc[idx] = [mse, rmse, mae, mpe, mape]
            pbar.update(1)
    results.loc["mean"] = results.mean(axis=0)
    results.loc["std"] = results.std(axis=0)
    results.loc["max"] = results.max(axis=0)
    results.loc["min"] = results.min(axis=0)
    if args.loss_result:
        results.to_csv(args.loss_result, index=True)
    print(results.loc[["mean", "std", "max", "min"]])
    eigen_vals, eigen_ratios, accumulate_ratios = model.get_eigen_values_and_ratio()
    print("Eigen Values:", eigen_vals)
    print("Ratios:", eigen_ratios)
    print("Accumulated Ratios:", [f"{i:.3f}" for i in accumulate_ratios])
else:
    model.load_from_file(args.model)
    eigen_vals, eigen_ratios, accumulate_ratios = model.get_eigen_values_and_ratio()
    print("Eigen Values:", eigen_vals)
    print("Ratios:", eigen_ratios)
    print("Accumulated Ratios:", accumulate_ratios)
    
    