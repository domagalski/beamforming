import numpy as np
import datetime
import h5py

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

def read_data(reader_obj, src, prod_sel, freq_sel=None, del_t=50):
    R = reader_obj

    # Figure out when calibration source transits
    src_trans = eph.transit_times(src, R.time[0]) 

    # Select +-100 seconds of transit
    time_range = np.where((R.time < src_trans + del_t) & (R.time > src_trans - del_t))[0]
    print time_range

    R.time_sel = [time_range[0], time_range[-1]]
    R.prod_sel = prod_sel
    R.freq_sel = freq_sel

    and_obj = R.read()

    return and_obj

def run_gain_solver(freq_range):
    #Xx = read_data(R, src, xcorrs, freq_sel=freq_range, del_t=50)
    Xy = read_data(R, src, ycorrs, freq_sel=freq_range)   
    print Xy.vis.shape
    #data_fs_x = tools.fringestop_pathfinder(\
    #     Xx.vis, eph.transit_RA(Xx.timestamp), Xx.freq, inpx, src)

    #del Xx
    data_fs_y = tools.fringestop_pathfinder(\
         Xy.vis, eph.transit_RA(Xy.timestamp), Xy.freq, inpy, src)

    print Xy.freq
    #dx, ax = solve_gain(data_fs_x)
    dy, ay = solve_gain(data_fs_y)
    
    return ay#, ay

def solve_untrans(filename, freq, corrs, inp):

    
    f = h5py.File(filename, 'r')
    v = f['vis'][945:965, freq[0]:freq[-1]+1, corrs]
    times = f['index_map']['time'].value['ctime'][945:965]

    vis = v['r'] + 1j * v['i']
    vis = np.transpose(vis, (1, 2, 0))
    freq_MHZ = 800.0 - np.array(freq) / 1024.0 * 400.

    data_fs = tools.fringestop_pathfinder(vis, eph.transit_RA(times), freq_MHZ, inp, src) 

    dy, ay = solve_gain(data_fs)

    return ay

src = eph.CygA

nfeed = 256

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


fn = '/scratch/k/krs/jrs65/chime_archive/20150517T220649Z_pathfinder_corr/00044096_0000.h5'
fn = '/mnt/gong/archive/20150517T220649Z_pathfinder_corr/00044096_0000.h5'
fn = '/mnt/gong/archive/20150531T044659Z_pathfinder_corr/00022098_0000.h5'
fn = '/mnt/gamelan/untransposed/20150611T200054Z_pathfinder_corr/00044436_0000.h5'
fn = '/mnt/gong/archive/20150611T200054Z_pathfinder_corr/00044436_0000.h5'

R = andata.Reader(fn)

fch = 32

for nu in range(1024 // fch):
    freq_range = range(nu * fch, (nu+1) * fch)
    print freq_range
    ax = run_gain_solver(freq_range)
#    ax = solve_untrans(fn, freq_range, xcorrs, inpx)

    outfile = 'outtest_cygx_' + np.str(nu) + '.hdf5'
    f = h5py.File(outfile, 'w')
    f.create_dataset('ax', data=ax)
#f.create_dataset('ay', data=ay)
    f.close()
    del ax





