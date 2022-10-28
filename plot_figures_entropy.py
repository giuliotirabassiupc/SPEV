from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np

MODEL = "biomass"
ORDERS = [2, (3, 2), 3]
STEPS = [1, 2, 4]
POSSIBLE_PARAMS = {"R", "I"}

RESULTS_DIR = (
    "/Users/giuliotirabassi/Documents/vegetation_network/results/spatial_entropy"
)
RESULTS_FILE = os.path.join(RESULTS_DIR, "order_{order}", "step_{step}", f"{MODEL}.csv")


if __name__ == "__main__":

    # HC plots
    for order in ORDERS:
        n_states = (
            np.math.factorial(order ** 2)
            if isinstance(order, int)
            else np.math.factorial(order[0] * order[1])
        )
        max_entopy = np.log(n_states)
        for step in STEPS:
            res_file = RESULTS_FILE.format(step=step, order=order)
            if not os.path.isfile(res_file):
                continue
            df = pd.read_csv(res_file)
            for param_name in POSSIBLE_PARAMS:
                if param_name in df.columns:
                    break
            plt.plot(
                df[param_name],
                df["mean_H"] / max_entopy,
                marker="o",
                lw=0,
                label=f"order={order}\nstep={step}",
            )
        plt.ylabel("Entropy")
        plt.grid(color="grey", ls=":")
    plt.xlabel(param_name)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(FIG_DIR, f"Entropy_{MODEL}.png"), dpi=300)

