import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

def polarmosaic(y,rg,r1,r2):
    precision = 8
    themin = np.min(y)
    themax = np.max(y)
    y = (y - themin) / (themax - themin)
    x = (np.arange(y.shape[0] + 1) - 0.5) / y.shape[0] * 2 * np.pi
    xs = np.zeros((4 * precision, y.shape[0]))
    ys = np.zeros((4 * precision, y.shape[0]))
    for i in range(y.shape[0]):
        xc, yc = wedgecoords(x[i], x[i+1], r1, r2, precision)
        xs[:,i] = xc
        ys[:,i] = yc

    baseline = (themin + themax) / 2  
    colors = ['blue', 'white', 'red']
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

    fig, ax = plt.subplots()
    colors_mapped = cmap(y)
    for i in range(xs.shape[1]):
        ax.fill(xs[:,i], ys[:,i], color=colors_mapped[i,:], clip_on=False)

    norm = TwoSlopeNorm(vmin=themin, vcenter=baseline, vmax=themax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Response Value')
    plt.show()
    # colors = plt.cm.jet(y)
    # for i in range(xs.shape[1]):
    #     ax.fill(xs[:,i], ys[:,i], color=colors[i,:], clip_on=False)

    # norm = Normalize(vmin=themin, vmax=themax)
    # sm = ScalarMappable(norm=norm, cmap='jet')
    # cbar = plt.colorbar(sm, ax=ax)
    # # cbar.set_label('Normalized Value')
    return

def wedgecoords(t1,t2,r1,r2,precision):
    ts = np.array([t1,t2,t2,t1,t1])
    rs = np.array([r1, r1, r2, r2, r1])
    xs = []
    ys = []
    for ii in range(4):
        tb = ts[ii + 1]
        ta = ts[ii]
        rb = rs[ii + 1]
        ra = rs[ii]
        dr = rb - ra
        dt = tb - ta
        for jj in np.linspace(0, 1, precision, endpoint=False):
            rn = ra + jj * dr
            tn = ta + jj * dt
            xs.append(rn * np.cos(tn))
            ys.append(rn * np.sin(tn))
    return np.array(xs), np.array(ys)


