from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from structure.rectangular_pythtb import *
from definitions.hwc import *
from draw_structure import *

def plot_band_rectangular(variables, w_soc, nks, jacutingaite, jacutingaite_name, occ):
    (fig, ax) = plt.subplots(1, 1, figsize=(4, 3.5))

    path = [
            [ 0, 0], 
            [ 0,-0.5], 
            [-0.5,-0.5], 
            [-0.5, 0], 
            [ 0, 0]
        ]
    
    label = (r'$\Gamma$', 'X', 'S', 'Y', r'$\Gamma$')

    my_model = set_model(*variables)
    (k_vec, k_dist, k_node) = my_model.k_path(path, nks)

    evals = my_model.solve_all(k_vec)

    ax.set_xlim([ 0, k_node[-1]])
    ax.set_xticks(k_node)
    ax.set_xticklabels(label)
    ax.set_title(jacutingaite_name, fontsize=18)
    ax.set_ylabel("Energy (eV)", labelpad=8, fontsize=16)
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([-0.6,-0.4,-0.2, 0, 0.2, 0.4, 0.6])

    minor_locator = AutoMinorLocator(5)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(length=7, right=True)
    ax.hlines(0, 0, k_node[-1], colors='gray', linestyle=':')

    for n in range(len(k_node)):
        ax.axvline(x=k_node[n], linewidth=0.5, color='k')

    for n in range(10):
        ax.plot(k_dist, evals[n], color="m", linewidth=2, linestyle='--')

    fig.tight_layout()

    bands = './results/{}_{}_bands_rectangular.pdf'.format(jacutingaite, w_soc)

    fig.savefig(bands)

    hwc_plot(my_model, jacutingaite, jacutingaite_name, w_soc, occ)

    draw_structure(my_model, jacutingaite)
