import pickle
import netCDF4
import os
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pyhdf.SD
from ordinal_patterns import SpatialOrdinalPattern
from analysis_eby import compute_morani_nan_robust
import tqdm
import re

RES_DIR = "../results/spatial_entropy/modis/"
PLOTS = False
SAVE_RES = True


def find_modis_box_indices(modis_filename):
    return [int(v) for v in re.findall(r"\.h(\d\d)v(\d\d)\.", modis_filename)[0]]


J = 2000
MIN_IDX = 0
MAX_IDX = -1
VEG_FILENAME = "MOD44B.A2019065.h20v08.006.2020099011202.hdf"
RAIN_FILENAME = (
    "g4.timeAvgMap.TRMM_3B43_7_precipitation.19980101-20191231.21W_4S_62E_24N.nc"
)
RAIN_NAME = "TRMM_3B43_7_precipitation"


J = 100
MIN_IDX = 500
MAX_IDX = -1700
VEG_FILENAME = "MOD44B.A2020065.h12v10.006.2021155164413.hdf"
RAIN_FILENAME = (
    "g4.timeAvgMap.TRMM_3B43_7_precipitation.19981201-20191231.83W_33S_29W_4N.nc"
)
RAIN_NAME = "TRMM_3B43_7_precipitation"


J = 2850
MIN_IDX = 1600
MAX_IDX = -1700
VEG_FILENAME = "MOD44B.A2020065.h19v08.006.2021155172226.hdf"
RAIN_FILENAME = (
    "g4.timeAvgMap.TRMM_3B43_7_precipitation.19980101-20191231.21W_4S_62E_24N.nc"
)
RAIN_NAME = "TRMM_3B43_7_precipitation"


J = 3450
MIN_IDX = 2200
MAX_IDX = -600
VEG_FILENAME = "MOD44B.A2020065.h12v10.006.2021155164413.hdf"
RAIN_FILENAME = (
    "g4.timeAvgMap.TRMM_3B43_7_precipitation.19981201-20191231.83W_33S_29W_4N.nc"
)
RAIN_NAME = "TRMM_3B43_7_precipitation"


J = 100
MIN_IDX = 500
MAX_IDX = -1700
VEG_FILENAME = "MOD44B.A2020065.h11v08.006.2021155155552.hdf"
RAIN_FILENAME = (
    "g4.timeAvgMap.TRMM_3B43_7_precipitation.19980101-20191231.83W_7S_42W_14N.nc"
)
RAIN_NAME = "TRMM_3B43_7_precipitation"


J = 0
MIN_IDX = 0
MAX_IDX = -1
VEG_FILENAME = "MOD44B.A2019065.h20v08.006.2020099011202.hdf"
RAIN_FILENAME = (
    "g4.timeAvgMap.TRMM_3B43_7_precipitation.19980101-20191231.21W_4S_62E_24N.nc"
)
RAIN_NAME = "TRMM_3B43_7_precipitation"


h_cell, v_cell = find_modis_box_indices(VEG_FILENAME)
i_cell = 17 - v_cell
j_cell = h_cell

DATA_DIR = (
    "/Users/giuliotirabassi/Documents/vegetation_network/data/satellite_vegetation"
)
VEG_NAME = "Percent_Tree_Cover"

SECTION_STEP = 50
width = 200
ORDER = 2
STEP = 1

RES_FILE = f"order={ORDER}_step={STEP}_v={v_cell}_h={h_cell}_J={J}.pkl"

# read data rain and vegetation
data_rain = netCDF4.Dataset(os.path.join(DATA_DIR, RAIN_FILENAME))
data_veg = pyhdf.SD.SD(os.path.join(DATA_DIR, VEG_FILENAME), pyhdf.SD.SDC.READ)
data_veg = data_veg.select(VEG_NAME).get().astype(float)
data_veg[data_veg > 100] = np.nan
data_veg = data_veg[::-1, :]
n_points = data_veg.shape[0]

# interpolate rain over vegetation space
N_H_CELLS = 36
N_V_CELLS = 18
proj_lat = np.linspace(-90, 90, N_V_CELLS + 1)
proj_lon = np.linspace(-180, 180, N_H_CELLS + 1)

finer_proj_lon = np.linspace(proj_lon[j_cell], proj_lon[j_cell + 1], n_points + 1)
finer_proj_lon = (finer_proj_lon[1:] + finer_proj_lon[:-1]) / 2
finer_proj_lat = np.linspace(proj_lat[i_cell], proj_lat[i_cell + 1], n_points + 1)
finer_proj_lat = (finer_proj_lat[1:] + finer_proj_lat[:-1]) / 2

xx_proj, yy_proj = np.meshgrid(finer_proj_lon, finer_proj_lat)

xx = xx_proj / np.cos(yy_proj * np.pi / 180)
yy = yy_proj

interpolator = RegularGridInterpolator(
    (data_rain["lat"], data_rain["lon"]),
    data_rain[RAIN_NAME][:],
    method="linear",
    bounds_error=False,
    fill_value=np.nan,
)

points = np.stack([yy.ravel(), xx.ravel()]).T.data
interpolated_rain = interpolator(points).reshape(xx.shape)

if PLOTS:
    # plot vegetation field and overimposed vegetation
    plt.pcolormesh(xx, yy, data_veg, cmap="Greens")
    plt.contour(xx, yy, interpolated_rain, cmap="Blues", levels=20)
    plt.show()

    # plot rainfall/vegetation profiles
    for j in range(0, data_veg.shape[0], 50):
        mv = []
        mr = []
        for i in range(0, data_veg.shape[0], 30):
            mv.append(np.nanmean(data_veg[i : i + width, :][:, j : j + width]))
            mr.append(np.nanmean(interpolated_rain[i : i + width, :][:, j : j + width]))
        plt.plot(mr, mv, "o")
        plt.title(j)
        plt.show()

# select specific profile
veg = data_veg[MIN_IDX:MAX_IDX, :][:, J : J + width]
rai = interpolated_rain[MIN_IDX:MAX_IDX, :][:, J : J + width]
xxx = xx[MIN_IDX:MAX_IDX, :][:, J : J + width]
yyy = yy[MIN_IDX:MAX_IDX, :][:, J : J + width]

left_side = np.column_stack((xxx.min(axis=1), yyy.min(axis=1)))
right_side = np.column_stack((xxx.max(axis=1), yyy.max(axis=1)))[::-1, :]
border = np.vstack((left_side, right_side, left_side[0, :]))

# analyse complexity and entropy
avg_v = []
c = []
h = []
s = []
r = []
for i in tqdm.tqdm(range(0, veg.shape[0], SECTION_STEP)):
    avg_v.append(np.nanmean(veg[i : i + width, :]))
    r.append(np.nanmean(rai[i : i + width, :]))
    op = SpatialOrdinalPattern(
        veg[i : i + width, :], order=ORDER, step=STEP, complexity="JS"
    )
    en = op._probs.compute_entropy(bayes=True, normalize=True)
    h.append(en)
    c.append(op._probs.compute_js_div() * en)
    s.append(compute_morani_nan_robust(veg[i : i + width, :]))

if PLOTS:
    fig, axes = plt.subplots(nrows=4, figsize=(5, 7), sharex=True)
    for i, (name, var) in enumerate(
        [
            ("Average Tree Cover", avg_v),
            ("Entropy", h),
            ("Complexity", c),
            ("Spatial\nCorrelation", s),
        ]
    ):
        axes[i].plot(r, var, "o")
        axes[i].set_ylabel(name)
        axes[i].grid(ls=":", color="grey")
    plt.show()

if SAVE_RES:
    with open(os.path.join(RES_DIR, RES_FILE), "wb",) as f:
        pickle.dump(
            {"average_tree_cover": avg_v, "H": h, "C": c, "SC": s, "rain": r}, f
        )
