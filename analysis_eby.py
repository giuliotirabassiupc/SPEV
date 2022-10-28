import ordinal_patterns
import os
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import tqdm
from scipy.signal import convolve2d
from scipy.interpolate import interp1d
from itertools import combinations_with_replacement, permutations
from discrete_distributions import DiscreteDistribution
import pickle
from collections import Counter

gc.enable()

PATCH_SIZE = 5
BINARY_PATCH_SIZE = 1  # put as -1 to never use binary OP
STEP_REDUCTION = 12  # transect_width / STEP_REDUCTION = step_size
ORDER = 2
STEP = 1  # of the spatial ordinal pattern for longer correlations

DATA_FILE = "transect_5_30m.csv"
DATA_DIR = "/Users/giuliotirabassi/Documents/vegetation_network/data/eby_2014"
RESULTS_DIR = (
    "/Users/giuliotirabassi/Documents/vegetation_network/results/spatial_entropy/eby/"
)


def compute_morani_nan_robust(x):
    xmean = np.nanmean(x)
    xvar = np.nanvar(x)
    if not xvar:
        return 1
    dxx = x - xmean
    corr = dxx * (np.roll(x, 1, axis=0) - xmean)
    corr += dxx * (np.roll(x, -1, axis=0) - xmean)
    corr += dxx * (np.roll(x, 1, axis=1) - xmean)
    corr += dxx * (np.roll(x, -1, axis=1) - xmean)
    corr /= 4
    return np.nanmean(corr) / xvar


def read_data_file(filename):
    df = pd.read_csv(os.path.join(DATA_DIR, filename), header=None)
    df[df < 0] = float("NaN")
    return df.values


def running_mean(a, step):
    kernel = np.ones((step, step)) / (step * step)
    b = convolve2d(a, kernel, "valid")
    return b[::step, :][:, ::step]


def binary_pathches(data, patch_size=2, step=1):
    interval = patch_size * step
    alphabeth = set()
    for x in combinations_with_replacement("01", ORDER ** 2):
        alphabeth.update(permutations(x))
    alphabeth = ["".join(x) for x in alphabeth]
    repres = []
    for i in range(data.shape[0] - patch_size):
        if i + interval > data.shape[0]:
            continue
        parent_slice = data[i : i + interval : step, :]
        for j in range(data.shape[1] - patch_size):
            if j + interval > data.shape[1]:
                continue
            slice = parent_slice[:, j : j + interval : step]
            if np.isnan(slice).sum() != 0:
                continue
            slice = slice.astype(int).astype(str).ravel()
            repres.append("".join(slice))
    return DiscreteDistribution(repres, alphabeth)


def compute_vegetated_patches(field):
    VEG = 0
    BARR = 1
    clusters = np.arange(1, field.size + 1).reshape(field.shape)
    clusters[field == BARR] = 0
    changed = True
    while changed:
        changed = False
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                if j + 1 < field.shape[1] and field[i, j] == VEG == field[i, j + 1]:
                    if clusters[i, j] < clusters[i, j + 1]:
                        clusters[i, j + 1] = clusters[i, j]
                        changed = True
                    elif clusters[i, j] > clusters[i, j + 1]:
                        clusters[i, j] = clusters[i, j + 1]
                        changed = True
                if i + 1 < field.shape[0] and field[i + 1, j] == VEG == field[i, j]:
                    if clusters[i, j] < clusters[i + 1, j]:
                        clusters[i + 1, j] = clusters[i, j]
                        changed = True
                    elif clusters[i, j] > clusters[i + 1, j]:
                        clusters[i, j] = clusters[i + 1, j]
                        changed = True
    ps = Counter(clusters.ravel())
    ps.pop(0, None)
    return Counter(ps.values())


if __name__ == "__main__":
    data = read_data_file(DATA_FILE)
    d = running_mean(data, PATCH_SIZE)
    sc = []
    c = []
    h = []
    avb = []
    h_step = d.shape[1] // STEP_REDUCTION
    h_width = d.shape[1]
    for i in tqdm.tqdm(range(0, d.shape[0], h_step)):
        if i + h_width > d.shape[0]:
            continue
        dd = d[i : i + h_width, :][:, : d.shape[1]]
        if i == 0:
            print(dd.shape)
        if PATCH_SIZE == BINARY_PATCH_SIZE:
            binary = True
            distr = binary_pathches(dd, patch_size=ORDER, step=STEP)
        else:
            binary = False
            op = ordinal_patterns.SpatialOrdinalPattern(
                dd, order=ORDER, step=STEP, complexity="JS"
            )
            distr = op._probs

        entr = distr.compute_entropy(bayes=True, normalize=True)
        h.append(entr)
        c.append(distr.compute_js_div() * entr)
        avb.append(np.nanmean(dd))
        sc.append(
            np.mean(
                [
                    compute_morani_nan_robust(dd[:, j : j + dd.shape[0]])
                    for j in range(0, dd.shape[1], dd.shape[0])
                    if j + dd.shape[0] <= dd.shape[1]
                ]
            )
        )
    print(dd.shape)
    h = np.array(h)
    c = np.array(c)
    sc = np.array(sc)
    avb = np.array(avb)

    # rainfall
    trrain = pd.read_csv(
        os.path.join(DATA_DIR, "transect_5_2500m_rainfall.csv"), sep="\t", header=None
    )
    trans_rainfall_len, transect_rainfall_width = trrain.shape
    trans_rainfall_cell_size = 2.5  # km
    tr_rain = [trrain.values[i : i + 3, :].mean() for i in range(trans_rainfall_len)]
    distance_rain = np.arange(1, trans_rainfall_len + 1) * trans_rainfall_cell_size
    interp_rain = interp1d(distance_rain, tr_rain)

    distances = []
    for i in range(0, d.shape[0], h_step):
        if i + h_width > d.shape[0]:
            continue
        distance = (i + 0.5 * h_width) * 0.03 * PATCH_SIZE  # km
        distances.append(distance)
    distances = np.array(distances)

    with open(
        os.path.join(
            RESULTS_DIR,
            f"patch={PATCH_SIZE}_reduc={STEP_REDUCTION}_order={ORDER}_step={STEP}_binary={binary}.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(
            {
                "distances_atc": distances,
                "average_tree_cover": 1 - avb,
                "rain": interp_rain(distances),
                "H": h,
                "C": c,
                "SC": sc,
            },
            f,
        )

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    axes[0].plot(distances, 1 - avb, "o")
    axes[0].set_ylabel("Average\nTree Cover")
    axes[1].plot(distance_rain, tr_rain, "o")
    axes[1].set_ylabel("Rainfall [mm]")
    for ax in axes:
        ax.grid(ls=":", c="grey")

    axes[-1].pcolormesh(
        np.arange(data.shape[0] + 1) * 0.03,
        np.arange(data.shape[1] + 1) * 0.03,
        1 - data.T,
        vmin=-1,
        vmax=1.5,
        cmap="Greens",
    )
    axes[-1].set_xlabel("Distance along transect [km]")
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
    axes[0].plot(interp_rain(distances), 1 - avb, "o")
    axes[0].set_ylabel("Average\nTree Cover")
    axes[1].plot(interp_rain(distances), h, "o")
    axes[1].set_ylabel("Entropy")
    axes[2].plot(interp_rain(distances), c, "o")
    axes[2].set_ylabel("Complexity")
    axes[3].plot(interp_rain(distances), sc, "o")
    axes[3].set_ylabel("Spatial correlation")
    axes[-1].set_xlabel("Rainfall [mm]")
    for ax in axes:
        ax.grid(ls=":", c="grey")
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
    # upper branch
    idx = (1 - avb) > 0.5
    axes[0].plot(interp_rain(distances[idx]), 1 - avb[idx], "o")
    axes[0].set_ylabel("Average\nTree Cover")
    axes[1].plot(interp_rain(distances[idx]), h[idx], "o")
    axes[1].set_ylabel("Entropy")
    axes[2].plot(interp_rain(distances[idx]), c[idx], "o")
    axes[2].set_ylabel("Complexity")
    axes[3].plot(interp_rain(distances[idx]), sc[idx], "o")
    axes[3].set_ylabel("Spatial correlation")
    axes[-1].set_xlabel("Rainfall [mm]")
    for ax in axes:
        ax.grid(ls=":", c="grey")
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
    # lower branch
    idx = ((1 - avb) < 0.3) & ((1 - avb) > 0)
    axes[0].plot(interp_rain(distances[idx]), 1 - avb[idx], "o")
    axes[0].set_ylabel("Average\nTree Cover")
    axes[1].plot(interp_rain(distances[idx]), h[idx], "o")
    axes[1].set_ylabel("Entropy")
    axes[2].plot(interp_rain(distances[idx]), c[idx], "o")
    axes[2].set_ylabel("Complexity")
    axes[3].plot(interp_rain(distances[idx]), sc[idx], "o")
    axes[3].set_ylabel("Spatial correlation")
    axes[-1].set_xlabel("Rainfall [mm]")
    for ax in axes:
        ax.grid(ls=":", c="grey")
    fig.tight_layout()
    plt.show()
