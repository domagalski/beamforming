import numpy as np
import datetime
import h5py

import misc_data_io as misc
from ch_util import andata, ephemeris as eph, tools

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')    

def plt_gains(vis, nu, img_name='out.png'):
    fig = plt.figure(figsize=(14,14))
    
    for i in range(128):
        fig.add_subplot(32, 4, i+1)
        plt.plot(np.angle(vis[nu, misc.feed_map(1, i+1, 128)]))
        plt.axis('off')
        plt.axhline(0.0)
        plt.ylim(-np.pi, np.pi)

    fig.savefig(img_name)

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

    dr[dr != dr] = 0.0

    print "Dynamic range max:", dr.max()
    print dr[-1]
    return dr, gain

def read_data(reader_obj, src, prod_sel, freq_sel=None, del_t=50):
    R = reader_obj

    # Figure out when calibration source transits
    src_trans = eph.transit_times(src, R.time[0]) 

    # Select +-100 seconds of transit
    time_range = np.where((R.time < src_trans + del_t) & (R.time > src_trans - del_t))[0]

    R.time_sel = [time_range[0], time_range[-1]]
    R.prod_sel = prod_sel
    R.freq_sel = freq_sel

    and_obj = R.read()

    return and_obj

def run_gain_solver(freq_range, xcorrs, ycorrs, inpx, inpy):

    Xy = read_data(R, src, ycorrs, freq_sel=freq_range, del_t=1000)   
    Xx = read_data(R, src, xcorrs, freq_sel=freq_range, del_t=1000)

    datay = Xy.vis[..., ::2]
    datay -= 0.5 * (datay[..., 0] + datay[..., -1])[..., np.newaxis] 

    datax = Xx.vis[..., ::2]
    datax -= 0.5 * (datax[..., 0] + datax[..., -1])[..., np.newaxis] 

    data_fs_x = tools.fringestop_pathfinder(\
         datax, eph.transit_RA(Xx.timestamp[::2]), Xx.freq, inpx, src)

    del Xx
    
    data_fs_y = tools.fringestop_pathfinder(\
         datay, eph.transit_RA(Xy.timestamp[::2]), Xy.freq, inpy, src)

    del Xy

    dx, ax = solve_gain(data_fs_x)
    dy, ay = solve_gain(data_fs_y)

    del data_fs_x, data_fs_y

    return ax, dx, ay, dy

def solve_untrans(filename, corrs, inp, src, nfreq=1024):
    del_t = 400

    f = h5py.File(filename, 'r')

    times = f['index_map']['time'].value['ctime']

    src_trans = eph.transit_times(src, times[0])

    # Select +-100 seconds of transit                                                                                                 
    t_range = np.where((times < src_trans + del_t) & (times > src_trans - del_t))[0]

    times = times[t_range[0]:t_range[-1]:2]

    print eph.transit_RA(times)

    Gains = np.zeros([nfreq, 128, len(times)], np.complex128)
    
    print "Reading full array"
    
    v = f['vis'][t_range[0]:t_range[-1]:2, :, corrs]

    print "Done reading array"

    for i in range(16):

        frq = range(i * nfreq // 16, (i+1) * nfreq // 16)

        print "      ", frq[0], ":", frq[-1]

        vis = v['r'][:, frq] + 1j * v['i'][:, frq]
        vis = np.transpose(vis, (1, 2, 0))
        vis -= 0.5 * (vis[..., 0] + vis[..., -1])[..., np.newaxis]
        
        freq_MHZ = 800.0 - np.array(frq) / 1024.0 * 400.
        
        data_fs = tools.fringestop_pathfinder(vis, eph.transit_RA(times), freq_MHZ, inp, src)
        dr, a = solve_gain(data_fs)

        Gains[frq] = a 

        plt_gains(data_fs, 0, img_name='./phs_plots/dfs' + np.str(frq[0]) + '.png')
        dfs_corr = correct_dfs(data_fs, Gains[frq], nfeed=128)
        plt_gains(dfs_corr, 0, img_name='./phs_plots/dfs_corr' + np.str(frq[0]) + '.png')
        

    del v, vis, data_fs

    return Gains

def correct_dfs(dfs, Gains, nfeed=256):
    """ Corrects fringestopped visibilities
    """

    dfs_corrm = dfs.copy()
    
    for i in range(nfeed):
        for j in range(i, nfeed):
            dfs_corrm[:, misc.feed_map(i, j, 128)] *= np.conj(Gains[:, i] * np.conj(Gains[:, j]))
            
    return dfs_corrm

def rearrange_inp(and_obj, corrinputs, nfeeds=256):
    """ Blegh
    """
    feeds = range(nfeeds)

    inp_real = []
    flist = []

    for feed in feeds[:]:
        for inp in range(nfeeds):

            if np.str(corrinputs[inp].input_sn)==np.str(and_obj.index_map['input'][feed][-1]):
                inp_real.append(corrinputs[inp])

                continue

    return inp_real, flist

def rearrange_list(corrinputs, nfeeds=256):
    inp_real = []

    for i in range(256):
        inp_real.append(corrinputs[flist[i]])

    return inp_real

def gen_inp(nfeed=256):
    """ Generate input information for feeds

    Parameters
    ----------
    feeds : list
         Feeds whose input info is needed
    nfeeds : int
         Number of feeds in total

    Returns
    -------
    corrinput_real : 
         All 256 inputs
    inpx : 
         Only x feeds
    inpy : 
         Only y feeds
    """

    # Assumes a standard layout for 128 feeds on each cyl
    xfeeds = range(nfeed/4) + range(2 * nfeed/4, 3 * nfeed/4)
    yfeeds = range(nfeed/4, 2 * nfeed/4) + range(3 * nfeed/4, 4 * nfeed/4)

    xcorrs = []
    ycorrs = []

    for ii in range(nfeed/2):
        for jj in range(ii, nfeed/2):
            xcorrs.append(misc.feed_map(xfeeds[ii], xfeeds[jj], nfeed))
            ycorrs.append(misc.feed_map(yfeeds[ii], yfeeds[jj], nfeed))

    corrinputs = tools.get_correlator_inputs(\
        datetime.datetime(2015, 6, 1, 0, 0, 0), correlator='K7BP16-0004')

    # Need to rearrange to match order in the correlated data
    corrinput_real = rearrange_list(corrinputs, nfeeds=256)

    inpx = []
    inpy = []
    
    for i in range(nfeed/2):
        inpx.append(corrinput_real[xfeeds[i]])
        inpy.append(corrinput_real[yfeeds[i]])

    return corrinput_real, inpx, inpy, xcorrs, ycorrs

def select_corrs(data, feeds, nfeed=256):
    corrs = []
    
    for ii in range(len(feeds)):
        for jj in range(ii, len(feeds)):
            corrs.append(misc.feed_map(feeds[ii], feeds[jj], nfeed))
    
    return data[:, corrs]

def fringestop_do_it_all(fn, feeds, freq, src, return_unfs=False):                                                 
    r = andata.Reader(fn)
    r.freq_sel = freq
    X = r.read()
    times = r.time[::2]

    print "Read it, bruh"

    del_t = 750.0
    src_trans = eph.transit_times(src, times[0])

    # Select +-100 seconds of transit                                                                                                                                       
    t_range = np.where((times < src_trans + del_t) & (times > src_trans - del_t))[0]

    corrinput_real = gen_inp()[0]
    inp = np.array(corrinput_real)[feeds]

    data = X.vis[..., ::2]
    data -= (data[..., t_range[0]] + data[..., t_range[-1]])[..., np.newaxis] / 2.0
    print t_range[0], t_range[-1]

    data = select_corrs(data, feeds)

    ra = eph.transit_RA(times)

    dfs = tools.fringestop_pathfinder(data, ra, X.freq, inp, src)

    if return_unfs is True:
        return dfs, ra, data
    else:
        return dfs, ra

flist = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
         60, 61, 62, 63, 16, 17, 18, 19, 20, 21, 22, 23, 
         24, 25, 26, 27, 28, 29, 30, 31, 240, 241, 242, 
         243, 244, 245, 246, 247, 248, 249, 250, 251, 252,
         253, 254, 255, 208, 209, 210, 211, 212, 213, 214,
         215, 216, 217, 218, 219, 220, 221, 222, 223, 32, 
         33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 
         46, 47, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
         13, 14, 15, 224, 225, 226, 227, 228, 229, 230, 231, 
         232, 233, 234, 235, 236, 237, 238, 239, 192, 193,
         194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
         204, 205, 206, 207, 112, 113, 114, 115, 116, 117,
         118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
         80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 
         93, 94, 95, 176, 177, 178, 179, 180, 181, 182, 183, 
         184, 185, 186, 187, 188, 189, 190, 191, 144, 145, 
         146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 
         156, 157, 158, 159, 96, 97, 98, 99, 100, 101, 102, 
         103, 104, 105, 106, 107, 108, 109, 110, 111, 64, 65,
         66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 
         79, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
         170, 171, 172, 173, 174, 175, 128, 129, 130, 131, 132, 
         133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143]



