# This code provides the tools for generating
# calibration solution out of transiting 
# point sources.

import sys

import numpy as np
import datetime
import h5py
import glob
import time

import misc_data_io as misc
from ch_util import andata, ephemeris as eph, tools
import correct_pkl as cp

import matplotlib
matplotlib.use('Agg')    

import matplotlib.pyplot as plt

def plt_gains(vis, nu, img_name='out.png', bad_chans=[]):
    """ Plot grid of transit phases to check if both 
    fringestop and the calibration worked. If most
    antennas end up with zero phase, then the calibration worked.

    Parameters
    ----------
    vis : np.ndarray[nfreq, ncorr, ntimes]
        Visibility array 
    nu  : int
        Frequency index to plot up
    """
    fig = plt.figure(figsize=(14, 14))
    
    # Plot up 128 feeds correlated with antenna "ant"
    ant = 1

    # Try and estimate the residual phase error 
    # after calibration. Zero would be a perfect calibration.
    angle_err = 0

    # Go through 128 feeds plotting up their phase during transit
    for i in range(32):
        fig.add_subplot(16, 2, i+1)
        antj = 4 * i + 1
        if i==ant:
            # For autocorrelation plot the visibility rather 
            # than the phase. This gives a sense for when 
            # the source actually transits. 
            plt.plot(vis[nu, misc.feed_map(ant, antj+1, 128)])
            plt.xlim(0, len(vis[nu, 0]))
        else:
            angle_err += np.mean(abs(np.angle(vis[nu, misc.feed_map(ant, antj+1, 128)]))) / 127.0
            plt.plot((np.angle(vis[nu, misc.feed_map(ant, antj+1, 128)])))
            plt.axis('off')
            plt.axhline(0.0, color='black')

            oo = np.round(np.std(np.angle(vis[nu, misc.feed_map(ant, antj+1, 128)]) * 180 / np.pi))
            plt.title(np.str(oo) + ',' + np.str(antj))

            if i in bad_chans:
                plt.plot(np.angle(vis[nu, misc.feed_map(ant, antj+1, 128)]), color='red')
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
    print dr[:]

    return dr, gain

def read_data(reader_obj, src, prod_sel, freq_sel=None, del_t=50):
    """ Use andata tools to select freq, products, 
    and times. 
    """
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

def solve_ps_transit(filename, corrs, feeds, inp, 
          src, nfreq=1024, transposed=False, nfeed=128):
    """ Function that fringestops time slice 
    where point source is in the beam, takes 
    all correlations for a given polarization, and then 
    eigendecomposes the correlation matrix freq by freq
    after removing the fpga phases. It will also 
    plot intermediate steps to verify the phase solution.

    Parameters
    ----------
    filename : np.str
         Full-path filename 
    corrs : list
         List of correlations to use in solver
    feeds : list
         List of feeds to use
    inp   : 
         Correlator inputs (output of ch_util.tools.get_correlator_inputs)
    src   : ephem.FixedBody
         Source to calibrate off of. e.g. ch_util.ephemeris.TauA
    
    Returns
    -------
    Gains : np.array
         Complex gain array (nfreq, nfeed) 
    """

    nsplit = 32 # Number of freq chunks to divide nfreq into
    del_t = 800

    f = h5py.File(filename, 'r')

    # Add half an integration time to each. Hack. 
    times = f['index_map']['time'].value['ctime'] + 10.50
    src_trans = eph.transit_times(src, times[0])
    
    # try to account for differential arrival time from 
    # cylinder rotation. 
    del_phi = (src._dec - np.radians(eph.CHIMELATITUDE)) \
                 * np.sin(np.radians(1.988))
    del_phi *= (24 * 3600.0) / (2 * np.pi)

    # Adjust the transit time accordingly
    src_trans += del_phi

    # Select +- del_t of transit, accounting for the mispointing 
    t_range = np.where((times < src_trans + 
                  del_t) & (times > src_trans - del_t))[0]
 
    print "\n...... This data is from %s starting at RA: %f ...... \n" \
        % (eph.unix_to_datetime(times[0]), eph.transit_RA(times[0]))

    assert (len(t_range) > 0), "Source is not in this acq"

    # Create gains array to fill in solution
    Gains = np.zeros([nfreq, nfeed], np.complex128)
    
    print "Starting the solver"
    
    times = times[t_range[0]:t_range[-1]]
    
    k=0
    
    # Start at a strong freq channel that can be plotted
    # and from which we can find the noise source on-sample
    for i in range(12, nsplit) + range(0, 12):

        k+=1

        # Divides the arrays up into nfreq / nsplit freq chunks and solves those
        frq = range(i * nfreq // nsplit, (i+1) * nfreq // nsplit)
        
        print "      %d:%d \n" % (frq[0], frq[-1])

        # Read in time and freq slice if data has already been transposed
        if transposed is True:
            v = f['vis'][frq[0]:frq[-1]+1, corrs, :]
            v = v[..., t_range[0]:t_range[-1]]
            vis = v['r'] + 1j * v['i']

            if k==1:
                autos = auto_corrs(nfeed)
                offp = (abs(vis[:, autos, 0::2]).mean() > \
                        (abs(vis[:, autos, 1::2]).mean())).astype(int)

                times = times[offp::2]
            
            vis = vis[..., offp::2]

            gg = f['gain_coeff'][frq[0]:frq[-1]+1, 
                    feeds, t_range[0]:t_range[-1]][..., offp::2]

            gain_coeff = gg['r'] + 1j * gg['i']
            
            del gg
            


        # Read in time and freq slice if data has not yet been transposed
        if transposed is False:
            print "TRANSPOSED V OF CODE DOESN'T WORK YET!"
            v = f['vis'][t_range[0]:t_range[-1]:2, frq[0]:frq[-1]+1, corrs]
            vis = v['r'][:] + 1j * v['i'][:]
            del v

            gg = f['gain_coeff'][0, frq[0]:frq[-1]+1, feeds]
            gain_coeff = gg['r'][:] + 1j * gg['i'][:]

            vis = vis[..., offp::2]

            vis = np.transpose(vis, (1, 2, 0))


        # Remove fpga gains from data
        vis = remove_fpga_gains(vis, gain_coeff, nfeed=nfeed, triu=False)

        # Remove offset from galaxy
        vis -= 0.5 * (vis[..., 0] + vis[..., -1])[..., np.newaxis]
   
        # Get physical freq for fringestopper
        freq_MHZ = 800.0 - np.array(frq) / 1024.0 * 400.
    
        baddies = np.where(np.isnan(tools.get_feed_positions(inp)[:, 0]))[0]
        a, b, c = select_corrs(baddies, nfeed=128)

        vis[:, a + b] = 0.0

        # Fringestop to location of "src"
        data_fs = tools.fringestop_pathfinder(vis, eph.transit_RA(times), freq_MHZ, inp, src)

        del vis

        dr, sol_arr = solve_gain(data_fs)

        # Find index of point source transit
        drlist = np.argmax(dr, axis=-1)
        
        # If multiple freq channels are zerod, the trans_pix
        # will end up being 0. This is bad, so ensure that 
        # you are only looking for non-zero transit pixels.
        drlist = [x for x in drlist if x != 0]
        trans_pix = np.argmax(np.bincount(drlist))

        assert trans_pix != 0.0

        Gains[frq] = sol_arr[..., trans_pix-3:trans_pix+4].mean(-1)

        zz = h5py.File('data' + str(i) + '.hdf5','w')
        zz.create_dataset('data', data=dr)
        zz.close()

        print "%f, %d Nans out of %d" % (np.isnan(sol_arr).sum(), np.isnan(Gains[frq]).sum(), np.isnan(Gains[frq]).sum())
        print trans_pix, sol_arr[..., trans_pix-3:trans_pix+4].mean(-1).sum(), sol_arr.mean(-1).sum()

        # Plot up post-fs phases to see if everything has been fixed
        if frq[0] == 12 * nsplit:
            print "======================"
            print "   Plotting up freq: %d" % frq[0]
            print "======================"
            img_nm = './phs_plots/dfs' + np.str(frq[17]) + np.str(np.int(time.time())) +'.png'
            img_nmcorr = './phs_plots/dfs' + np.str(frq[17]) + np.str(np.int(time.time())) +'corr.png'

            plt_gains(data_fs, 0, img_name=img_nm, bad_chans=baddies)
            dfs_corr = correct_dfs(data_fs, np.angle(Gains[frq])[..., np.newaxis], nfeed=128)

            plt_gains(dfs_corr, 0, img_name=img_nmcorr, bad_chans=baddies)

            del dfs_corr

        del data_fs, a

    return Gains

def beamform_correlated_data(filename, frq, src, feeds=None, del_t=900, 
                             transposed=True, absval=False):

    """ Beamform to a given src

    Parameters
    ---------- 
    
    filename : np.str
        name of .h5 file
    frq      : list
        list of frequency indexes ([305] not 305) 
    src      : ephem object
        source to fringestop to, e.g. ch_util.ephemeris.CasA 
    feeds    : list of correlation inputs to 
               use (e.g. range(64) uses only west cyl x-polarization)
    absval   : boolean
        In case data are not calibrated


    Returns 
    -------

    data_beamformed : fringestopped and summed data
    """

    if feeds == None:
        feeds = range(256)

    data_fringestop = fs_from_file(filename, frq, src, 
                              del_t=del_t, transposed=transposed)
    
    if absval is True:
        data_fringestop = np.abs(data_fringestop)

    data_beamformed = sum_corrs(data_fringestop, feeds)


    return data_beamformed
    

def fs_from_file(filename, frq, src,  
                 del_t=900, transposed=True, subtract_avg=False):

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


    if transposed is True:
        v = f['vis'][frq[0]:frq[-1]+1, :]
        v = v[..., t_range[0]:t_range[-1]]
        vis = v['r'] + 1j * v['i']

        del v

    # Read in time and freq slice if data has not yet been transposed
    if transposed is False:
         v = f['vis'][t_range[0]:t_range[-1], frq[0]:frq[-1]+1, :]
         vis = v['r'][:] + 1j * v['i'][:]
         del v
         vis = np.transpose(vis, (1, 2, 0))

    inp = gen_inp()[0]

    # Remove offset from galaxy                                                                                
    if subtract_avg is True:
        vis -= 0.5 * (vis[..., 0] + vis[..., -1])[..., np.newaxis]

    freq_MHZ = 800.0 - np.array(frq) / 1024.0 * 400.
    print len(inp)

    baddies = np.where(np.isnan(tools.get_feed_positions(inp)[:, 0]))[0]

    # Fringestop to location of "src"

    data_fs = tools.fringestop_pathfinder(vis, eph.transit_RA(times), freq_MHZ, inp, src)
#    data_fs = fringestop_pathfinder(vis, eph.transit_RA(times), freq_MHZ, inp, src)


    return data_fs


def remove_fpga_gains(vis, gains, nfeed=128, triu=False):
    """ Remove fpga phases
    """

    print "............ Removing FPGA gains, triu is %r ............ \n" % triu

    # Get gain matrix for visibilites g_i \times g_j^*
    #gains_corr = gains[:, :, np.newaxis] * np.conj(gains[:, np.newaxis])
    
    # Take only upper triangle of gain matrix
    #ind = np.triu_indices(nfeed)
    
    #gains_mat = np.zeros([vis.shape[0], len(ind[0])], dtype=gains.dtype)

    for nu in range(vis.shape[0]):
        for i in range(nfeed):
            for j in range(i, nfeed):
                phi = np.angle(gains[nu, i] * np.conj(gains[nu, j]))

                # CHIME seems to have a lowertriangle X-engine, should
                # in general use Vij = np.conj(xi) * xj
                if triu==True:
                    vis[nu, misc.feed_map(i, j, nfeed)] *= np.exp(-1j * phi)

                elif triu==False:
                    vis[nu, misc.feed_map(i, j, nfeed)] *= np.exp(1j * phi)
    
    return vis

    
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

def select_corrs(feeds, nfeed=256, feeds_cross=None):
    autos = []
    corrs = []
    xycorrs = []

    for ii, feedi in enumerate(feeds):

        for jj, feedj in enumerate(feeds):

            if ii==jj:

                autos.append(misc.feed_map(feedi, feedj, nfeed))

            if jj>ii:
                corrs.append(misc.feed_map(feedi, feedj, nfeed))

            if feeds_cross != None:
                assert len(feeds) <= len(feeds_cross)
                xycorrs.append(misc.feed_map(feedi, feeds_cross[jj], nfeed))

    return autos, corrs, xycorrs

def sum_corrs(data, feeds):
    autos, xcorrs, xycorrs = select_corrs(feeds)
 
#    print "Adding phase errors"
#    phase_rand = np.random.normal(0, 0.5, len(xcorrs))
#    data[:, xcorrs] *= np.exp(1j * 0.5)

    return data[:, autos].sum(1) + 2 * data[:, xcorrs].sum(1)


def make_outfile_name(fn):

    try:
        fname = fn.split('/')
        tstring = fname[-2] + fname[-1][:-3]
    except AttributeError:
        print "fn is None"
        return 

    return tstring
    
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

    ha = (np.radians(ra) - src._ra)[np.newaxis, np.newaxis, :]

    _PF_LAT = np.radians(ephemeris.CHIMELATITUDE)   # Latitude of pathfinder                                                                                                                                                                            
    xp, yp = feedpos[:, 0], feedpos[:, 1]

    xd = xp[:, np.newaxis] - xp[np.newaxis, :]
    yd = yp[:, np.newaxis] - yp[np.newaxis, :]
    xd = tools.pack_product_array(xd, axis=0)
    yd = tools.pack_product_array(yd, axis=0)

    if frick != None:
        frick = 800.0 - 400 / 1024.0 * frick

    print "Using %d" % freq

    wv = scipy.constants.c * 1e-6 / freq[:, np.newaxis, np.newaxis]
    u = xd[np.newaxis, :, np.newaxis] / wv
    v = yd[np.newaxis, :, np.newaxis] / wv

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

    del_phi = 1.30 * (src._dec - np.radians(eph.CHIMELATITUDE)) * np.sin(np.radians(1.988))
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
    
def noise_src_phase(fn_ns_sol, fn_sky_sol, src=eph.CasA, trb=10):
    """ Get noise source phase at the time sky solution was calculated
    (e.g. time of CasA transit) and use that as baseline.

    Parameters
    ----------
    fn_ns_sol : np.str
       file name with noise source solution
    fn_sky_sol : np.str
       file name with sky solution
    src : ephemeris.Object
       object that gave sky solution in fn_sky_sol
    tbr : np.int
       time rebin factor

    Returns
    -------
    phase solution : array_like
       (nfeed, nt) real array of phases
   
    """

    f = h5py.File(fn_ns_sol, 'r')
    fsky = h5py.File(fn_sky_sol, 'r')

    gx = fsky['gainsx'][:]
    gy = fsky['gainsy'][:]

    gains_ns = f['gains'][:]

    feeds = f['channels'][:]
    freq = f['freq_bins'][:]
    toff = f['timestamp_off']

    nt = len(toff)
    nfeed = len(feeds)
    nfreq = len(freq)
    
    toff = toff[::10]

    gains_sky = cp.construct_gain_mat(gx, gy, 64)
    gains_sky = gains_sky[freq, feeds]

    gains_ns = gains_ns[..., :nt // trb * trb].reshape(nfreq, nfeed, -1, trb).mean(-1)
    src_trans = eph.transit_times(src, toff[0])

    trans_pix = np.argmin(abs(src_trans - toff))
    phase = np.angle(gains_ns) - np.angle(gains_ns[..., trans_pix, np.newaxis])

    return np.angle(gains_sky)[:, np.newaxis] + phase 


def find_transit(time0='Now', src=eph.CasA):
    import time
    
    if time0 == 'Now':
        time0 = time.time()

    # Get reference datetime from unixtime
    dt_now = eph.unix_to_datetime(time0)
    dt_now = dt_now.isoformat()

    # Only use relevant characters in datetime string
    dt_str = dt_now.replace("-", "")[:7]
    dirnm = '/mnt/gong/archive/' + dt_str

    filelist = glob.glob(dirnm + '*/*h5')

    # Step through each file and find that day's transit
    for ff in filelist:
        
        try:
            andataReader = andata.Reader(ff)
            acqtimes = andataReader.time
            trans_time = eph.transit_times(src, time0)
            
            #print ff
#            print eph.transit_RA(trans_time), eph.transit_RA(acqtimes[0])
            del andataReader
            
            if np.abs(acqtimes - trans_time[0]).min() < 1000.0:
#                print "On ", eph.unix_to_datetime(trans_time[0])
#                print "foundit in %s \n" % ff
                return ff

                break

        except (KeyError, ValueError, IOError):
            pass

    return None


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




