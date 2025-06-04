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
        plt.clf()
        print(f"on {str(rd)}")
        with (rd / "parameters.json").open('r') as f:
            params = json.load(f)


        total_runs = params["total_runs"]
        colours = cm.rainbow(np.linspace(0, 1, total_runs))

        errors_l2 = np.load(rd / "recomp_errors_l2.npz")["recomp_errors"]
        errors_l_inf = np.load(rd / "recomp_errors_l_inf.npz")["recomp_errors"]

        max_errors_l2 = []
        zeros_l2 = []
        # plot l2
        for j in range(total_runs):
            errorsj = errors_l2[j]
            errorsjX = errorsj[:,0] ; errorsjY = errorsj[:,1]
            max_errors_l2.append(errorsjY.max())
            zeros_l2.append(len(np.where(errorsjY == 0.0)[0]) / len(errorsjX))
            plt.plot(errorsjX,
                     errorsjY,
                     marker=markerstyles[j % 3],
                     ls=linestyles[j % TOTAL_LS],
                     color=colours[j],
                     alpha=0.7)

        max_errors_l2 = np.stack(max_errors_l2)
        zeros_l2 = np.stack(zeros_l2)

        model = params["generator_params"]["model"]
        dataset = params["generator_params"]["dataset"]
        epochs = params["generator_params"]["epochs"]
        batch_size = params["generator_params"]["batch_size"]
        k = params["generator_params"]["every"]
        title = rf"{model} {dataset} $b = {batch_size}, E = {epochs}, k = {k}$,  {total_runs} runs"
        plt.xlabel(r'step')
        plt.ylabel(rf'$\ell_2$')
        plt.yscale('log')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(rd / "recomp_errors_l2.pdf")

        plt.clf()

        print(title)
        print(f"     l2: {max_errors_l2.mean():.2e} +- {max_errors_l2.std():.2e}, max = {max_errors_l2.max()}")
        print(f"zero l2: {zeros_l2.mean():.2f} +- {zeros_l2.std():.2f}")

        # plot l_inf
        for j in range(total_runs):
            errorsj = errors_l_inf[j]
            errorsjX = errorsj[:,0] ; errorsjY = errorsj[:,1]

            plt.plot(errorsjX,
                     errorsjY,
                     ls=linestyles[j % TOTAL_LS],
                     color=colours[j],
                     alpha=0.6)


        model = params["generator_params"]["model"]
        dataset = params["generator_params"]["dataset"]
        epochs = params["generator_params"]["epochs"]
        batch_size = params["generator_params"]["batch_size"]
        k = params["generator_params"]["every"]
        plt.xlabel(r'step')
        plt.ylabel(rf'$\ell_{{\infty}}$')
        # plt.yscale('log')
        plt.title(rf"{model} {dataset} $b = {batch_size}, E = {epochs}, k = {k}$,  {total_runs} runs")
        plt.tight_layout()
        plt.savefig(rd / "recomp_errors_l_inf.pdf")


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("run_dirs", type=Path, nargs='+')

    args = parser.parse_args()

    main(args)
