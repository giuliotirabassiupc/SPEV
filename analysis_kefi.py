import ordinal_patterns
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import gc
import time
import pickle
from analysis_eby import running_mean, compute_morani_nan_robust, binary_pathches
import matplotlib.pyplot as plt


gc.enable()

ORDER = 2
STEP = 2
PATCH_SIZE = 1

MODEL = "cellular_automata"
DATA_DIR = "/Users/giuliotirabassi/Documents/vegetation_network/data/simulations/"
DATA_FILE = "snapshots_kefi_vegetated_2.pkl"  # "snapshots_kefi.pkl"
RES_DIR = "/Users/giuliotirabassi/Documents/vegetation_network/results/spatial_entropy/cellular_automata/"
RES_FILE = os.path.join(
    RES_DIR, f"order={ORDER}_step={STEP}_patch={PATCH_SIZE}_vegetated.pkl"
)
MODEL_DATA_DIR = os.path.join(DATA_DIR, MODEL)


def read_data_file(filename, model_dir):
    with open(os.path.join(model_dir, filename), "rb") as f:
        data = pickle.load(f)
    return data


def compute_entropy_and_complexity(frames):
    h = []
    c = []
    s = []
    for frame in frames:
        dd = np.array(frame) == 1
        dd = running_mean(dd, PATCH_SIZE)
        if PATCH_SIZE == 1:
            distr = binary_pathches(dd, patch_size=ORDER)
        else:
            op = ordinal_patterns.SpatialOrdinalPattern(
                dd, order=ORDER, step=STEP, complexity="JS"
            )
            distr = op._probs
        h.append(distr.compute_entropy(bayes=True, normalize=True))
        c.append(
            distr.compute_js_div() * distr.compute_entropy(bayes=True, normalize=True)
        )
        s.append(compute_morani_nan_robust(dd))
    return h, c, s


if __name__ == "__main__":
    snapshots = read_data_file(DATA_FILE, MODEL_DATA_DIR)
    futures = {}
    with ProcessPoolExecutor(os.cpu_count()) as executor:
        for m in snapshots:
            futures[m] = executor.submit(compute_entropy_and_complexity, snapshots[m])
        results = {}
        while len(results) != len(futures):
            for info in futures:
                if not futures[info].done():
                    time.sleep(1)
                    continue
                if not results.get(info):
                    print("Getting results for ", info)
                    results[info] = futures[info].result()
    re_h = {m: h for m, (h, _, _) in results.items()}
    re_c = {m: c for m, (_, c, _) in results.items()}
    re_s = {m: s for m, (_, _, s) in results.items()}
    tree_cover = {m: np.array(v) == 1 for m, v in snapshots.items()}

    with open(RES_FILE, "wb") as f:
        pickle.dump(
            {"average_tree_cover": tree_cover, "H": re_h, "C": re_c, "SC": re_s}, f
        )

    fig, axes = plt.subplots(nrows=4, figsize=(5, 7), sharex=True)
    for i, (name, var) in enumerate(
        [
            ("Average Tree Cover", tree_cover),
            ("Entropy", re_h),
            ("Complexity", re_c),
            ("Spatial\nCorrelation", re_s),
        ]
    ):
        axes[i].errorbar(
            var.keys(),
            [np.mean(v) for v in var.values()],
            yerr=None
            if name == "Average Tree Cover"
            else [np.std(v) for v in var.values()],
            lw=0,
            elinewidth=2,
            marker="o",
        )
        axes[i].set_ylabel(name)
        axes[i].grid(ls=":", color="grey")
    plt.show()
