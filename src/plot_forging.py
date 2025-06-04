import argparse as ap
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
import numpy as np
import utils

linestyles   = ["solid", "dashed", "dashdot", "dotted"]
TOTAL_LS     = len(linestyles)
markerstyles = ["+", "x", "o"]

font = {'weight' : 'bold',
        'size'   : 11}

matplotlib.rc('font', **font)


def load_params(rd):
    with (rd / "parameters.json").open('r') as f:
        params = json.load(f)
    return params



def main(args):
    for rd in args.run_dirs:
        plt.cla()
        with (rd / "parameters.json").open('r') as f:
            params = json.load(f)


        tp = params["trace_parameters"]
        model = tp["model"]; dataset= tp["dataset"]
        b = tp["batch_size"]; ff = params["f"]


        results = np.load(rd / "results.npz")
        errors = results["approx_errors"]
        losses = results["losses"]
        gnorms = results ["gnorms"]
        print(errors.shape)

        # y_mean = np.mean(errors, 0)
        # y_min = np.min(errors, 0)[:,1]
        # y_max = np.max(errors, 0)[:,1]
        # errs = [y_mean[:,1] - y_min, y_max - y_mean[:,1]]

        
        # print(y_min, y_min.shape)

        # plt.errorbar(y_mean[:,0], y_mean[:,1], yerr=errs, fmt='-o')

        for run in errors:
            plt.plot(run[:,0], run[:,1])

        if "threshold" in params:
            print("plotting t")
            plt.axhline(y=params["threshold"], color='r', linestyle="-.")
        plt.xlabel("step")
        plt.ylabel(r'$\epsilon_{approx}$')
        plt.title(f"{model} {dataset} (b = {b}) forging {ff} example(s)")

        plt.yscale('log', base=10)

        plt.savefig(rd/"approx_errors.pdf")

        plt.cla()

        for run in losses:
            plt.plot(run[:,0], run[:,1])

        plt.xlabel("step")
        plt.ylabel(r'$\ell$')
        plt.title(f"{model} {dataset} (b = {b}) forging {ff} example(s)")
        plt.yscale('log', base=10)

        plt.savefig(rd/"losses.pdf")

        plt.cla()

        for run in losses:
            plt.plot(run[:,0], run[:,1])

        plt.xlabel("step")
        plt.ylabel(r'$\|\nabla_{\theta}\ell(x,y)\|_2$')
        plt.title(f"{model} {dataset} (b = {b}) forging {ff} example(s)")
        plt.yscale('log', base=10)

        plt.savefig(rd/"gnorms.pdf")


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("run_dirs", nargs='+', type=Path)
    parser.add_argument("--name", type=str, default="")
    args = parser.parse_args()

    main(args)
