import ordinal_patterns
import os
import re
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import gc
import time

gc.enable()

MODEL = "fn_sub_hopf_new"
ORDER = 2
STEP = 1

DATA_DIR = "/Users/giuliotirabassi/Documents/vegetation_network/data/simulations"
MODEL_DATA_DIR = os.path.join(DATA_DIR, MODEL)
FILES = [f for f in os.listdir(MODEL_DATA_DIR) if f.endswith(".csv")]
RESULTS_DIR = (
    "/Users/giuliotirabassi/Documents/vegetation_network/results/spatial_entropy"
)
RESULTS_FILE = os.path.join(
    RESULTS_DIR, f"order_{ORDER}", f"step_{STEP}", f"{MODEL}.csv"
)


def read_data_file(filename, model_dir, is1d=False):
    data = pd.read_csv(os.path.join(model_dir, filename), header=None, sep=r"\s+")
    t, n = data.shape
    ll = int(n ** 0.5)
    assert ll ** 2 == n
    return data.values.reshape((t, ll, ll))


def strip_info_from_filename(filename):
    return re.findall(r"(\w+)=(\-?\d+\.\d+)", filename.split("_")[-1])[0]


def compute_entropy_and_complexity(filename):
    data = read_data_file(filename, MODEL_DATA_DIR)
    entropy = []
    complexity = []
    for i in range(data.shape[0]):
        op = ordinal_patterns.SpatialOrdinalPattern(
            data[i, :, :], order=ORDER, step=STEP
        )
        entropy.append(op._probs.compute_entropy(bayes=True))
        complexity.append(
            op._probs.compute_js_div() * op._probs.compute_entropy(bayes=True)
        )
        # complexity.append(op.entropy)
        # complexity.append(op.complexity)

    entropy = np.mean(entropy), np.std(entropy)
    complexity = np.mean(complexity), np.std(complexity)
    return {"H": entropy, "C": complexity}


if __name__ == "__main__":
    futures = {}
    with ProcessPoolExecutor(os.cpu_count()) as executor:
        for file_name in FILES:
            info = strip_info_from_filename(file_name)
            futures[info] = executor.submit(compute_entropy_and_complexity, file_name)
        results = {}
        while len(results) != len(futures):
            for info in futures:
                if not futures[info].done():
                    time.sleep(1)
                    continue
                if not results.get(info):
                    print("Getting results for ", info)
                    results[info] = futures[info].result()

    df = pd.DataFrame.from_dict(results, orient="index").reset_index()
    df["mean_H"] = df["H"].apply(lambda x: x[0])
    df["std_H"] = df["H"].apply(lambda x: x[1])
    df["mean_C"] = df["C"].apply(lambda x: x[0])
    df["std_C"] = df["C"].apply(lambda x: x[1])
    param_name = df["level_0"].values[0]
    df[param_name] = df["level_1"].astype(float)
    df = df[[param_name, "mean_H", "std_H", "mean_C", "std_C"]]

    df.to_csv(RESULTS_FILE, index=False)
