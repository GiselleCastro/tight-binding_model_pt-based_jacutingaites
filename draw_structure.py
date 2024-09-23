import numpy as np
import matplotlib.pyplot as plt

def draw_structure(self, jacutingaite):

    fig = plt.figure(figsize=[plt.rcParams["figure.figsize"][0],
                              plt.rcParams["figure.figsize"][0]])
    ax = fig.add_subplot(111, aspect='equal')

    def proj(v):
        return [v[0], v[1]]

    def to_cart(red):
        return np.dot(red, self._lat)

    for i in range(self._norb):
        pos = to_cart(self._orb[i])
        pos = proj(pos)
        if (self._orb[i][2] == 0.5):
            ax.plot([pos[0]], [pos[1]], "o", c='#E26A00',
                    mec="w", mew=0.0, zorder=10, ms=9.0)
        else:
            ax.plot([pos[0]], [pos[1]], "o", c='#007A08',
                    mec="w", mew=0.0, zorder=10, ms=9.0)

    xl = ax.set_xlim()
    yl = ax.set_ylim()
    centx = (xl[1]+xl[0])*0.5
    centy = (yl[1]+yl[0])*0.5
    mx = max([xl[1]-xl[0], yl[1]-yl[0]])
    extr = 0.05
    ax.set_xlim(centx-mx*(0.5+extr), centx+mx*(0.5+extr))
    ax.set_ylim(centy-mx*(0.5+extr), centy+mx*(0.5+extr))

    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    fig.savefig("{}_structure.pdf".format(jacutingaite))
