from pythtb import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def hwc_plot(model, jacutingaite, jacutingaite_name, w_soc, occ):
    my_array = wf_array(model, [100, 100])
    my_array.solve_on_grid([-0.5, -0.5])
    wan_cent = my_array.berry_phase(
        occ=occ, 
        dir=1, 
        contin=False, 
        berry_evals=True
        )
    wan_cent /= (2.0*np.pi)

    nky = wan_cent.shape[0]
    ky = np.linspace(0., 1., nky)

    (fig, ax) = plt.subplots(1, 1, figsize=(4, 4))

    for shift in range(-2, 3):
        for band in occ:
            ax.plot(ky, wan_cent[:, band]+float(shift), "k.")
            
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel('Hybrid Wannier Center \n direction $x$',
                  labelpad=8, fontsize=14)
    ax.set_xlabel(r'$k_y$')
    ax.set_xticks([ 0.0, 0.5, 1.0])
    ax.set_xlim(0.0, 1.0)
    ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    ax.axvline(x=.5, linewidth=0.5, color='k')
    ax.set_title(jacutingaite_name)

    fig.tight_layout()

    hwc_plot = './results/{}_{}_hwc.pdf'.format(jacutingaite, w_soc)

    fig.savefig(hwc_plot)
