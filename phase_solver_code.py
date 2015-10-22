import sys

import numpy as np
import datetime
import h5py
import glob

import misc_data_io as misc
from ch_util import andata, ephemeris as eph, tools

import matplotlib
matplotlib.use('Agg')    

import matplotlib.pyplot as plt

def plt_gains(vis, nu, img_name='out.png', bad_chans=[]):
    fig = plt.figure(figsize=(14, 14))
    
    # Plot up 128 feeds correlated with antenna "ant"
    ant = 1
    angle_err = 0
    for i in range(128):
        fig.add_subplot(32, 4, i+1)

        if i==ant:
            plt.plot(vis[nu, misc.feed_map(ant, i+1, 128)])
            plt.xlim(0, len(vis[nu, 0]))
        else:
            angle_err += np.mean(abs(np.angle(vis[nu, misc.feed_map(ant, i+1, 128)]))) / 127.0
            plt.plot((np.angle(vis[nu, misc.feed_map(ant, i+1, 128)])))
            #plt.plot(vis[nu, misc.feed_map(ant, i+1, 128)])
            plt.axis('off')
            plt.axhline(0.0, color='black')

            oo = np.round(np.std(np.angle(vis[nu, misc.feed_map(ant, i+1, 128)]) * 180 / np.pi))
            plt.title(np.str(oo) + ',' + np.str(i))

            if i in bad_chans:
                plt.plot(np.angle(vis[nu, misc.feed_map(ant, i+1, 128)]), color='red')
            plt.ylim(-np.pi, np.pi)

    plt.title(np.str(180 / np.pi * angle_err))
            
    del vis

    print "\n Wrote to %s \n" % img_name

    fig.savefig(img_name)

    del fig

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

    print "Dynamic range max: %f" % dr.max()
    print dr[0]

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

def solve_untrans(filename, corrs, feeds, inp, src, nfreq=1024, transposed=False, nfeed=128):
    del_t = 600

    f = h5py.File(filename, 'r')

    times = f['index_map']['time'].value['ctime']
    print len(times), eph.transit_RA(times)
    src_trans = eph.transit_times(src, times[0])
    
    # try to account for differential arrival time from 
    # cylinder rotation. 
    del_phi = (src._dec - np.radians(eph.CHIMELATITUDE)) * np.sin(np.radians(1.988))
    del_phi *= (24 * 3600.0) / (2 * np.pi)

    # Adjust the transit time accordingly
    src_trans += del_phi

    # Select +- del_t of transit, accounting for the mispointing 
    t_range = np.where((times < src_trans + del_t) & (times > src_trans - del_t))[0]
 
    print "\n...... This data is from %s starting at RA: %f ...... \n" \
        % (eph.unix_to_datetime(times[0]), eph.transit_RA(times[0]))

    assert (len(t_range) > 0), "Source is not in this acq"

    Gains = np.zeros([nfreq, nfeed], np.complex128)
    
    print "Starting the solver"
    
    nsplit = 32
    times = times[t_range[0]:t_range[-1]]
    
    k=0
    for i in range(nsplit):
        k+=1
        ## Divides the arrays up into nfreq / nsplit freq chunks and solves those
        frq = range(i * nfreq // nsplit, (i+1) * nfreq // nsplit)
        
        print "      %d:%d \n" % (frq[0], frq[-1])

        # Read in time and freq slice if data has already been transposed
        if transposed is True:
            v = f['vis'][frq[0]:frq[-1]+1, corrs, :]
            v = v[..., t_range[0]:t_range[-1]]
            vis = v['r'] + 1j * v['i']

            autos = auto_corrs(nfeed)
            offp = (abs(vis[:, autos, 0::2]).mean() > (abs(vis[:, autos, 1::2]).mean())).astype(int)

            vis = vis[..., offp::2]

            gg = f['gain_coeff'][frq[0]:frq[-1]+1, feeds, 0]
            gain_coeff = (gg['r'] + 1j * gg['i'])

            del gg
            
        # Read in time and freq slice if data has not yet been transposed
        if transposed is False:
            v = f['vis'][t_range[0]:t_range[-1]:2, frq[0]:frq[-1]+1, corrs]
            vis = v['r'][:] + 1j * v['i'][:]

            gg = f['gain_coeff'][0, frq[0]:frq[-1]+1, feeds]
            gain_coeff = gg['r'][:] + 1j * gg['i'][:]

            del v

            vis = np.transpose(vis, (1, 2, 0))
        
        if k==1: times = times[offp::2]
        # Remove fpga gains from data
        vis = remove_fpga_gains(vis, gain_coeff, nfeed=nfeed)

        # Remove offset from galaxy
        vis -= 0.5 * (vis[..., 0] + vis[..., -1])[..., np.newaxis]
   
        freq_MHZ = 800.0 - np.array(frq) / 1024.0 * 400.
    
        baddies = np.where(np.isnan(tools.get_feed_positions(inp)[:, 0]))[0]

        # Fringestop to location of "src"
        data_fs = tools.fringestop_pathfinder(vis, eph.transit_RA(times), freq_MHZ, inp, src)

        del vis

        dr, a = solve_gain(data_fs)

        trans_pix = np.argmax(np.bincount(np.argmax(dr, axis=-1)))

        Gains[frq] = a[..., trans_pix-2:trans_pix+2].mean(-1)


        # Plot up post-fs phases to see if everything has been fixed
        if frq[0] == 9 * nsplit:
            print "======================"
            print "   Plotting up freq: %d" % frq[0]
            print "======================"

            plt_gains(data_fs, 0, img_name='./phs_plots/dfs' + np.str(frq[17]) + '.png', bad_chans=baddies)
            dfs_corr = correct_dfs(data_fs, np.angle(Gains[frq])[..., np.newaxis], nfeed=128)
            plt_gains(dfs_corr, 0, img_name='./phs_plots/dfs_corrmeanflah' + np.str(frq[17]) + '.png', bad_chans=baddies)

            del dfs_corr

        del data_fs, a

    return Gains

def fs_from_file(filename, frq, src, nfreq=1024, del_t=900, transposed=True):

    f = h5py.File(filename, 'r')

    times = f['index_map']['time'].value['ctime'] + 10.6

    src_trans = eph.transit_times(src, times[0])

    # try to account for differential arrival time from cylinder rotation. 

    del_phi = (src._dec - np.radians(eph.CHIMELATITUDE)) * np.sin(np.radians(1.988))
    del_phi *= (24 * 3600.0) / (2 * np.pi)

    # Adjust the transit time accordingly                                                                                             
    src_trans += del_phi

    # Select +- del_t of transit, accounting for the mispointing                                                                      
    t_range = np.where((times < src_trans + del_t) & (times > src_trans - del_t))[0]

    times = times[t_range[0]:t_range[-1]]#[offp::2] test

    print "Time range:", times[0], times[-1]

    print "\n...... This data is from %s starting at RA: %f ...... \n" \
        % (eph.unix_to_datetime(times[0]), eph.transit_RA(times[0]))

#    times = times[t_range[0]:t_range[-1]]

    if transposed is True:
        v = f['vis'][frq[0]:frq[-1]+1, :]
        v = v[..., t_range[0]:t_range[-1]]
        vis = v['r'] + 1j * v['i']


     # Read in time and freq slice if data has not yet been transposed
    if transposed is False:
         v = f['vis'][t_range[0]:t_range[-1]:2, frq[0]:frq[-1]+1, :]
         vis = v['r'][:] + 1j * v['i'][:]
         del v

         vis = np.transpose(vis, (1, 2, 0))

    inp = gen_inp()[0]
    # Remove offset from galaxy                                                                                                                                                                                 
    #vis -= 0.5 * (vis[..., 0] + vis[..., -1])[..., np.newaxis]

    freq_MHZ = 800.0 - np.array(frq) / 1024.0 * 400.
    print len(inp)

    baddies = np.where(np.isnan(tools.get_feed_positions(inp)[:, 0]))[0]

    # Fringestop to location of "src"                                                                                                                                                                        
    data_fs = tools.fringestop_pathfinder(vis, eph.transit_RA(times), freq_MHZ, inp, src)


    return data_fs

def remove_fpga_gains(vis, gains, nfeed=128):
    """ Remove fpga phases
    """

    print "............ Removing FPGA gains ............ \n"

    # Get gain matrix for visibilites g_i \times g_j^*
    #gains_corr = gains[:, :, np.newaxis] * np.conj(gains[:, np.newaxis])
    
    # Take only upper triangle of gain matrix
    #ind = np.triu_indices(nfeed)
    
    #gains_mat = np.zeros([vis.shape[0], len(ind[0])], dtype=gains.dtype)

    for nu in range(vis.shape[0]):
        for i in range(nfeed):
            for j in range(i, nfeed):
                phi = np.angle(gains[nu, i] * np.conj(gains[nu, j]))
                vis[nu, misc.feed_map(i, j, nfeed)] *= np.exp(-1j * phi)
    
    return vis

#    for nu in range(vis.shape[0]):
#        gains_mat[nu] = gains_corr[nu][ind]
        
#    phase = np.angle(gains_mat)[..., np.newaxis]

#    return vis * np.exp(-1j * phase)
    
def correct_dfs(dfs, Gains, nfeed=256):
    """ Corrects fringestopped visibilities
    """

    dfs_corrm = dfs.copy()

    for i in range(nfeed):
        for j in range(i, nfeed):
            #dfs_corrm[:, misc.feed_map(i, j, 128)] *= np.conj(Gains[:, i] * np.conj(Gains[:, j]))
            dfs_corrm[:, misc.feed_map(i, j, 128)] *= np.exp(-1j * (Gains[:, i] - Gains[:, j])) 
            
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

    return corrinput_real, inpx, inpy, xcorrs, ycorrs, xfeeds, yfeeds

def auto_corrs(nfeed):
    autos = []

    for ii in range(nfeed):
        autos.append(misc.feed_map(ii, ii, nfeed))

    return autos

def select_corrs(feeds, nfeed=256):
    autos = []
    corrs = []
    
    for ii in range(len(feeds)):
        for jj in range(ii, len(feeds)):
            if ii==jj:
                autos.append(misc.feed_map(feeds[ii], feeds[jj], nfeed))
            else:
                corrs.append(misc.feed_map(feeds[ii], feeds[jj], nfeed))
    
    return autos, corrs

def sum_corrs(data, feeds):
    autos, corrs = select_corrs(feeds)
    
    return data[:, autos].sum(1) + 2 * data[:, corrs].sum(1)

    
def fill_nolan(times, ra, dec, feed_positions):

    nt = len(times)
    na = 256

    PH = np.zeros([1, 32896, nt])

    for tt in range(len(times)):
        pht = nolan_phases(times[tt], ra, dec, feed_positions)

        PH[0, :, tt] = (pht.repeat(na).reshape(-1, na)\
             - (pht.repeat(na).reshape(-1, na)).transpose())[np.triu_indices(256)]

    return PH

def nolan_ra(unix_time):
    D2R = np.pi / 180.0
    TAU = 2 * np.pi

    one_over_c = 3.3356
    phi_0 = 280.46 #- 0.211
    lst_rate = 360./86164.09054

    inst_long = -119.6175
    inst_lat = 49.3203

    j2000_unix = 946728000
    
    # This function disagrees with Kiyo's ephemeris.transit_RA
    # since Nolan does not account for precession/nutation. This offset
    # should take care of that.
    precession_offset = (unix_time - j2000_unix) * 0.012791 / (365 * 24 * 3600) 

    lst = phi_0 + inst_long + lst_rate*(unix_time - j2000_unix) - precession_offset
    lst = np.fmod(lst, 360)

    return lst

def nolan_phases(unix_time, ra, dec, feed_positions):
    D2R = np.pi / 180.0
    TAU = 2 * np.pi

    one_over_c = 3.3356
    phi_0 = 280.46
    lst_rate = 360./86164.09054

    inst_long = -119.62
    inst_lat = 49.32

    j2000_unix = 946728000
    lst = phi_0 + inst_long + lst_rate*(unix_time - j2000_unix)

    lst = np.fmod(lst, 360.)
    hour_angle = lst - ra

    alt = np.sin(dec*D2R)*np.sin(inst_lat*D2R)+np.cos(dec*D2R)*np.cos(inst_lat*D2R)*np.cos(hour_angle*D2R)
    alt = np.arcsin(alt)

    az = (np.sin(dec*D2R) - np.sin(alt)*np.sin(inst_lat*D2R))/(np.cos(alt)*np.cos(inst_lat*D2R))
    az = np.arccos(az)

    if np.sin(hour_angle*D2R) >= 0:
        az = TAU - az

    phases = np.zeros([256])

    for i in range(256):
        projection_angle = 90*D2R - np.arctan2(feed_positions[2*i+1],feed_positions[2*i])
        offset_distance  = np.cos(alt)*np.sqrt(feed_positions[2*i]*feed_positions[2*i] + feed_positions[2*i+1]*feed_positions[2*i+1])
        effective_angle  = projection_angle - az

        phases[i] = 2 * np.pi * np.cos(effective_angle) * offset_distance * one_over_c

    return phases

def fringestop_pathfinder(timestream, ra, freq, feeds, src, frick=None):

    import scipy.constants
    ephemeris = eph

    # Calculate the hour angle                                                                                                                       \
                                                                                                                                                      
    ha = (np.radians(ra) - src._ra)[np.newaxis, np.newaxis, :]

    _PF_LAT = np.radians(ephemeris.CHIMELATITUDE)   # Latitude of pathfinder                                                                         \
                                                                                                                                                      

    # Get feed positions                                                                                                                             \
                                                                                                                                                      
    feedpos = tools.get_feed_positions(feeds)    

    xp, yp = feedpos[:, 0], feedpos[:, 1]

    # Calculate baseline separations and pack into product array                                                                                     \
                                                                                                                                                      
    xd = xp[:, np.newaxis] - xp[np.newaxis, :]
    yd = yp[:, np.newaxis] - yp[np.newaxis, :]
    xd = tools.pack_product_array(xd, axis=0)
    yd = tools.pack_product_array(yd, axis=0)

    # Calculate wavelengths and UV place separations                                                                                                 \

    if frick != None:
        frick = 800.0 - 400 / 1024.0 * frick
#        freq = freq * 0.0 + frick

    print "Using %d" % freq

    wv = scipy.constants.c * 1e-6 / freq[:, np.newaxis, np.newaxis]
    u = xd[np.newaxis, :, np.newaxis] / wv
    v = yd[np.newaxis, :, np.newaxis] / wv


    # Construct fringestop phase and set any non CHIME feeds to have zero phase                                                                      \
                                                                                                                                                      
    fs_phase = tools.fringestop_phase(ha, _PF_LAT, src._dec, u, v) 
    fs_phase = np.where(np.isnan(fs_phase), np.zeros_like(fs_phase), fs_phase)


    return timestream * fs_phase


def fringestop_and_sum(fn, feeds, freq, src, transposed=True, 
            return_unfs=True, meridian=False, del_t=1500, frick=None):             
    """ Take an input file fn and a set of feeds and return 
    a formed beam on src. 
    """

    if transposed is True:
        r = andata.Reader(fn)
        r.freq_sel = freq
        X = r.read()
        times = r.time
    else:
        f = h5py.File(fn, 'r')  
        times = f['index_map']['time'].value['ctime']

    print "Read in data"

    # Get transit time for source 
    src_trans = eph.transit_times(src, times[0]) 

    del_phi = (src._dec - np.radians(eph.CHIMELATITUDE)) * np.sin(np.radians(1.988))
    del_phi *= (24 * 3600.0) / (2 * np.pi)

    # Adjust the transit time accordingly                                                                                                            
    src_trans += del_phi

    # Select +- del_t of transit, accounting for the mispointing                                                                                     
    t_range = np.where((times < src_trans + del_t) & (times > src_trans - del_t))[0]

    times = times[t_range[0]:t_range[-1]]#[offp::2] test

    print "Time range:", times[0], times[-1]
    
    # Generate correctly ordered corrinputs
    corrinput_real = gen_inp()[0]
    inp = np.array(corrinput_real)

    # Ensure vis array is in correct order (freq, prod, time)
    if transposed is True:
        data = X.vis[:, :, t_range[0]:t_range[-1]]
        freq = X.freq
    else:
        v = f['vis'][t_range[0]:t_range[-1], freq, :] 
        vis = v['r'] + 1j * v['i']
        data = vis.transpose()[np.newaxis]

        del vis

        freq = 800 - 400.0 / 1024 * freq
        freq = np.array([freq])

#    autos = auto_corrs(256)
#    offp = (abs(data[:, autos, 0::2]).mean() > (abs(data[:, autos, 1::2]).mean())).astype(int)
#    data = data[..., offp::2] test

    data_unfs = sum_corrs(data.copy(), feeds)
    ra_ = eph.transit_RA(times)
    ra_2 = nolan_ra(times)

    #ra_ = ra_2.copy()

    if meridian is True:
        ra = np.ones_like(ra_) * np.degrees(src._ra)
    else:
        ra = ra_

    print len(inp)
    dfs = tools.fringestop_pathfinder(data.copy(), ra, freq, inp, src)    
    #dfs = fringestop_pathfinder(data.copy(), ra_2, freq, inp, src, frick=frick)
#    dfs = fringestop_pathfinder(data.copy(), ra_1, freq, inp, src, frick=frick)

#    fp = np.loadtxt('/home/connor/feed_layout_decrease.txt')
#    PH = fill_nolan(times, src._ra  * 180.0 / np.pi, src._dec * 180.0 / np.pi, fp)

    dfs_sum = sum_corrs(dfs, feeds)

    if return_unfs is True:
        return dfs_sum, ra_, dfs, data
    else:
        return dfs_sum, ra_

def find_transit_file(dir_nm, unix_time=None, src=None, trans=True, verbose=True):
    flist = glob.glob(dir_nm)
    tdel = []
    
    print "Looking at %d files" % len(flist)

    for ii in range(len(flist)):

        if trans is True:
            try:
                r = andata.Reader(flist[ii])

                if ii==0 and src != None:
                    time_trans = eph.transit_times(src, r.time[0])
                elif unix_time != None:
                    time_trans = unix_time
                else:
                    continue

                tdel.append(np.min(abs(time_trans - r.time)))

                if verbose is True:
                    print flist[ii]
                    print "%f hours away \n" % np.min(abs(time_trans - r.time)) / 3600.0

            except (KeyError, IOError):
                print "That one didn't work"

        elif trans is False:

            try:
                f = h5py.File(flist[ii], 'r')
                times = f['index_map']['time'].value['ctime'][0]
                f.close()

                time_trans = unix_time

                tdel.append(np.min(abs(time_trans - times)))

            except (ValueError, IOError):
                pass

    tdel = np.array(tdel)

    return flist[np.argmin(tdel)], tdel.min()
    


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



