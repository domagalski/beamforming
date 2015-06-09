import numpy as np
import datetime

import misc_data_io as misc
from ch_util import andata, ephemeris as eph, tools

def solve_gain(data, feeds=None):
    """
    Steps through each time/freq pixel, generates a Hermitian matrix and
    calculates gains from its largest eigenvector.

    Parameters
    ----------
    data : np.ndarray[nfreq, nprod, ntime]
        Visibility array to be decomposed
    feed_loc : list
        Which feeds to include. If :obj:`None` include all feeds.

    Returns
    -------
    dr : np.ndarray[nfreq, ntime]
        Dynamic range of solution.
    gain : np.ndarray[nfreq, nfeed, ntime]
        Gain solution for each feed, time, and frequency
    """

    # Expand the products to correlation matrices
    corr_data = tools.unpack_product_array(data, axis=1, feeds=feeds)

    nfeed = corr_data.shape[1]

    gain = np.zeros((data.shape[0], nfeed, data.shape[-1]), np.complex128)
    dr = np.zeros((data.shape[0], data.shape[-1]), np.float64)

    for fi in range(data.shape[0]):
        for ti in range(data.shape[-1]):

            cd = corr_data[fi, :, :, ti]

            if not np.isfinite(cd).all():
                continue

            # Normalise and solve for eigenvectors
            xc, ach = tools.normalise_correlations(cd)
            evals, evecs = tools.eigh_no_diagonal(xc, niter=5)

            # Construct dynamic range and gain
            dr[fi, ti] = evals[-1] / np.abs(evals[:-1]).max()
            gain[fi, :, ti] = ach * evecs[:, -1] * evals[-1]**0.5

    return dr, gain

nfeed = 256

arrx = np.zeros([nfeed / 2, nfeed / 2], np.complex128)
arry = arrx.copy()

xfeeds = range(nfeed / 4) + range(2 * nfeed / 4, 3 * nfeed / 4)
yfeeds = range(nfeed / 4, 2 * nfeed / 4) + range(3 * nfeed / 4, 4 * nfeed / 4)

xcorrs = []
ycorrs = []

for ii in range(nfeed/2):
     for jj in range(ii, nfeed/2):
          xcorrs.append(misc.feed_map(xfeeds[ii], xfeeds[jj], nfeed))
          ycorrs.append(misc.feed_map(yfeeds[ii], yfeeds[jj], nfeed))

print "Lengths", len(xcorrs), len(ycorrs)

corrinputs = tools.get_correlator_inputs(\
                datetime.datetime(2015, 6, 1, 0, 0, 0), correlator='K7BP16-0004')


inpx = []
inpy = []

for i in range(nfeed / 2):
    inpx.append(corrinputs[xfeeds[i]])
    inpy.append(corrinputs[yfeeds[i]])


R = andata.Reader('/scratch/k/krs/jrs65/chime_archive/20150517T220649Z_pathfinder_corr/00044096_0000.h5')
R.freq_sel = 304
R.time_sel = [820, 920]

X = R.read()

data = X.vis[:, :, 50]

data_fs_x = tools.fringestop_pathfinder(\
     X.vis[:, xcorrs], eph.transit_RA(X.timestamp), X.freq, inpx, eph.CasA)

data_fs_y = tools.fringestop_pathfinder(\
     X.vis[:, ycorrs], eph.transit_RA(X.timestamp), X.freq, inpy, eph.CasA)

dx, ax = solve_gain(data_fs_x, feeds=xfeeds)
dy, ay = solve_gain(data_fs_y, feeds=yfeeds)




