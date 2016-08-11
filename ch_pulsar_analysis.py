# Liam Connor 4 November 2015 
# Code mainly to dedisperse and fold pulsar data

import numpy as np

#import misc_data_io as misc

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

chime_lat = 49.320

class PulsarPipeline:

    def __init__(self, data_arr, time_stamps):
        self.data = data_arr.copy()
        self.ntimes = self.data.shape[-1]
        self.time_stamps = time_stamps

        if self.ntimes != len(self.time_stamps):
            print self.ntimes, "not equal to", len(self.time_stamps)
            raise Exception('Number of samples disagree')         
        
        self.RA = None
        self.dec = None
        self.RA_src = None

        self.data = np.ma.array(self.data, mask=np.zeros(self.data.shape, dtype=bool))
        self.nfreq = self.data.shape[0]
        self.highfreq = 800.0
        self.lowfreq = 400.0
        self.freq = np.linspace(self.highfreq, self.lowfreq, self.nfreq)

        self.ntimes = self.data.shape[-1]
        self.ncorr = self.data.shape[1]
        print self.ncorr
        self.corrs = range(self.ncorr)
        self.ln = '29'
        print "Data array has shape:", self.data.shape
        
    
    def get_uv(self):
        """ Takes a layout .txt file and generates each baseline's
        freq dependent u,v arrays.

        Returns
        -------
        u : array_like
            (nfreq, ncorr, 1)
        v : array_like
            (nfreq, ncorr, 1)
        """
        fname = '/home/k/krs/connor/code/ch_misc_routines/pulsar/feed_loc_layout' + self.ln + '.txt'
        feed_loc = np.loadtxt(fname)
        d_EW, d_NS = misc.calc_baseline(feed_loc)[:2]

        u = d_EW[np.newaxis, self.corrs, np.newaxis] \
            * self.freq[:, np.newaxis, np.newaxis] * 1e6 / (3e8)

        v = d_NS[np.newaxis, self.corrs, np.newaxis] \
            * self.freq[:, np.newaxis, np.newaxis] * 1e6 / (3e8)
        
        return u, v

    def dm_delays(self, dm, f_ref):
        """
        Provides dispersion delays as a function of frequency. Note
        frequencies are in MHz.
        
        Parameters
        ----------
        dm : float
                Dispersion measure in pc/cm**3
        f_ref : float
                reference frequency for time delays
                
        Returns
        -------
        Vector of delays in seconds as a function of frequency
        """
        return 4.148808e3 * dm * (self.freq**(-2) - f_ref**(-2))


    def fold(self, dm, p0, ntrebin=100, ngate=32):
        """Folds pulsar into nbins after dedispersing it. 
        
        Parameters
        ----------
        p0 : float
                Pulsar period in seconds. 
        dm : float
                Dispersion measure in pc/cm**3
        ntrebin : np.int
                Number of time stamps that go into folded period
        ngate : np.int
                Number of phase bins to fold on
                
        Returns
        -------
        fold_arr : array_like 
                Folded complex array shaped (nfreq, ncorr, ntimes/ntrebin, ngate)
        icount : array_like
                Number of elements in given phase bin (nfreq, ntimes/ntrebin, ngate)

        """        

        ntimes = self.data.shape[-1] // ntrebin
        ncorr = self.ncorr

        fold_arr = np.zeros([self.nfreq, ncorr, ntimes, ngate], np.complex128)
        icount = np.zeros([self.nfreq, ntimes, ngate], np.int32)

        dshape = self.data.shape[:-1] + (ntimes, ntrebin)

        data_rb = self.data[..., :(ntimes*ntrebin)].reshape(dshape)
        
        bins_all = self.phase_bins(p0, dm, ngate)

        for fi in range(self.nfreq):
            if (fi % 128) == 0:
                print "Folded freq", fi
            tau = self.dm_delays(dm, 400)

            bins = bins_all[fi, :(ntrebin*ntimes)].reshape(ntimes, ntrebin)

            for ti in range(ntimes):

                icount[fi, ti] = np.bincount(bins[ti], data_rb[fi, 0, ti] != 0., ngate)

                for corr in range(self.ncorr):

                    data_fold_r = np.bincount(bins[ti], 
                           weights=data_rb[fi, corr, ti].real, minlength=ngate)
                    data_fold_i = np.bincount(bins[ti], 
                           weights=data_rb[fi, corr, ti].imag, minlength=ngate)

                    fold_arr[fi, corr, ti, :] = data_fold_r + 1j * data_fold_i

        return fold_arr, icount[:, np.newaxis], bins_all

    def phase_bins(self, p0, dm, ngate):

        bins = np.zeros([self.nfreq, self.ntimes], np.int)
        
        times = self.time_stamps

        for fi in range(self.nfreq):

            tau = self.dm_delays(dm, 400)
            times_del = times - tau[fi]

            bins[fi] = (((times_del / p0) % 1) * ngate).astype(np.int)

        return bins


    def fold_real(self, dm, p0, times, ntrebin=100, ngate=32):
        """ Folds pulsar into nbins after dedispersing it. 
        
        Parameters
        ----------
        p0 : float
                Pulsar period in seconds. 
        dm : float
                Dispersion measure in pc/cm**3
        ntrebin : np.int
                Number of time stamps that go into folded period
        ngate : np.int
                Number of phase bins to fold on
                
        Returns
        -------
        fold_arr : array_like 
                Folded complex array shaped (nfreq, ncorr, ntimes/ntrebin, ngate)
        icount : array_like
                Number of elements in given phase bin (nfreq, ntimes/ntrebin, ngate)

        """        

        ntimes = self.data.shape[-1] // ntrebin
        ncorr = self.data.shape[1]

        print self.nfreq, ncorr, ntimes, ngate

        fold_arr = np.zeros([self.nfreq, ncorr, ntimes, ngate], self.data.dtype)
        icount = np.zeros([self.nfreq, ncorr, ntimes, ngate], np.int32)

        dshape = self.data.shape[:-1] + (ntimes, ntrebin)

        data_rb = self.data[..., :(ntimes*ntrebin)].reshape(dshape)
        
        bins_all = self.phase_bins_fullarr(p0, dm, ngate, times)

        for fi in range(self.nfreq):

            if (fi % 128) == 0:
                print "Folded freq", fi

            tau = self.dm_delays(dm, 400)

            bins = bins_all[fi, :, :(ntrebin*ntimes)].reshape(ncorr, ntimes, ntrebin)

            for ti in range(ntimes):

                for corr in range(ncorr):
                    icount[fi, corr, ti] = np.bincount(
                        bins[corr, ti], data_rb[fi, corr, ti] != 0., ngate)

                    data_fold_r = np.bincount(bins[corr, ti], 
                             weights=data_rb[fi, corr, ti], minlength=ngate)

                    fold_arr[fi, corr, ti, :] = data_fold_r
             

        return fold_arr, icount


    def phase_bins_fullarr(self, p0, dm, ngate, times):
        """ Get phase bins for a time array with different
        orderings for each frequency and correlation product.
        """

        bins = np.zeros([self.nfreq, self.ncorr, self.ntimes], np.int)
    
        tau = self.dm_delays(dm, 400)       

        for fi in range(self.nfreq):
        
            for corr in range(self.ncorr):

#                print (times[fi, 0] != times[fi, 1]).sum() / np.float(len(times[fi, 0]))

                times_del = times[fi, corr] - tau[fi]
        
                times_del = np.array(times_del).reshape(1, -1)
                bins[fi, corr] = (((times_del / p0) % 1) * ngate).astype(np.int)      

        return bins

    
    def noisefunc(self, data):
        ngate = data.shape[-1]

        on = [19, 20, 21]
        off = np.delete(range(ngate), on)
        
        data[np.isnan(data)] = 0.0
        profile_avg = data.mean(axis=1)
        on_avg = profile_avg[..., on] - profile_avg[..., np.newaxis, off].mean(-1)
        
        data_no_ns = data.copy()

        #Now recover the data without any noise injection
        data_no_ns[..., on] -= on_avg[:, np.newaxis]

        data_on = data[..., on] - np.median(data[..., np.newaxis, off], axis=-1)

        return data_no_ns, data_on
                        

    def fringestop(self, reverse=False, uf=1.0, vf=1.0):
        """ Gets a time and freq dependent phase from 
        fringestop_phase and apply it to the data. 

        Parameters
        ----------
        reverse : boolean
             If True this will apply the conjugate phase

        Returns
        -------
        Fringestopped data array (nfreq, ncorr, ntimes)
        """
        data = self.data.copy()

        ha = np.deg2rad(self.RA[np.newaxis, np.newaxis, :] - self.RA_src)
        dec = np.deg2rad(self.dec)

        u, v = self.get_uv()
        phase = self.fringestop_phase(ha, np.deg2rad(chime_lat), dec, uf*u, vf*v)
        
        if reverse==True:
            self.data = data * np.conj(phase)
            print "Reverse fringestopping"
        else:
            self.data = data * phase


    def fringestop_phase(self, ha, lat, dec, u, v):
        """Return the phase required to fringestop. All angle inputs are radians. 

        Parameter
        ---------
        ha : array_like
             The Hour Angle of the source to fringestop too.
        lat : array_like
             The latitude of the observatory.
        dec : array_like
             The declination of the source.
        u : array_like
             The EW separation in wavelengths (increases to the E)
        v : array_like
             The NS separation in wavelengths (increases to the N)
        
        Returns
        -------
        phase : np.ndarray
        The phase required to *correct* the fringeing. Shape is
        given by the broadcast of the arguments together.
        """
        
        uhdotn = np.cos(dec) * np.sin(-ha)
        vhdotn = np.cos(lat) * np.sin(dec) - np.sin(lat) * np.cos(dec) * np.cos(-ha)
        phase = uhdotn * u + vhdotn * v
        
        return np.exp(2.0J * np.pi * phase)

    def dedispersed_timestream(self, dm, times, sample_t=1/625.0, onm='bonus'):
        """
        """

        for nu in range(self.nfreq):

            for corr in range(self.ncorr):
                ind = np.where(times[nu, corr] != 0.0)[0]

                if len(ind) < 1: continue
                
                tmin = times[nu, corr, ind].min()
                tmax = times[nu, corr, ind].max()
                times[nu, corr] -= tmin
                
        del_T = times.max()
        nb = np.int(del_T / sample_t)
        nt = times.shape[-1]

        print self.data.shape, del_T.shape
        spec, bincount = self.fold_real(dm, del_T, 
                                times, ntrebin=nt, ngate=nb)

        del times

        arr_dd = spec / bincount

        del spec, bincount

        arr_dd[mask] = 0.0
        arr_dd[np.isnan(arr_dd)] = 0.0        

        # Fill Stokes I
        arr_dd[:, 0] = arr_dd[:, 0] + arr_dd[:, 3]
        
        # Fill abs(U + iV)
        #arr_dd[:, 1] = (arr_dd[:, 1] + 1j*arr_dd[:, 2])
        arr_dd[:, 1] = np.angle(arr_dd[:, 1] + 1j*arr_dd[:, 2]) 
        times_ordered = np.linspace(0, sample_t * nb, nb)

        print "dim here"
        plot_ddtimestream(arr_dd, times_ordered, 'stand.png')

        #return arr_dd[:, :2].mean(0)[:, 0], times_ordered#, arr_dd
        return arr_dd[:, :, 0], times_ordered
    
mask = [142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 553,
       554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566,
       567, 568, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594,
       595, 596, 597, 598, 599, 631, 632, 633, 634, 635, 636, 637, 638,
       639, 640, 641, 642, 643, 644, 677, 678, 679, 680, 681, 682, 683,
       684, 685, 686, 687, 688, 689, 690, 691, 754, 755, 756, 757, 758,
       759, 760, 762, 763, 786, 787, 789, 808, 809, 846, 882, 895, 975]

mask += range(105, 185)    

# Accounting for new bad channels. 
mask = [142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 754, 755, 756, 757, 758, 759, 760, 762, 763, 786, 787, 789, 808, 809, 846, 882, 895, 975, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 848, 859, 866, 867, 874, 875, 883, 890, 891]

class RFI_Clean(PulsarPipeline):

    always_cut = range(111,138) + range(889,893) + range(856,860) + \
        range(873,877) + range(583,600) + range(552,569) + range(630,645) +\
        range(675,692) + range(753, 768) + range(783, 798) + [267,268,273] + \
        [882, 808, 809, 810, 846, 847, 855, 876, 879, 878, 864, 896, 897, 798, 1017, 978, 977]

    def frequency_clean(self, threshold=1e6, broadband_only=True):
        """
        Does an RFI cut in frequency by normalizing by nu**4 to flatten bandpass and then cutting 
        above some threshold. 
        
        Parameters
        ----------
        data : array of np.complex128
                pulsar data array. 
        threshold :
                threshold at which to cut spectral RFI. For autocorrelations threshold=1 is reasonable, for
                cross-correlations 0.25 seems to work.
                
        Returns
        -------
        RFI-cleaned data

        """
        if broadband_only==False:
            data = abs(self.data.copy())
            freq = (self.freq)[:, np.newaxis]
            data_freq = data.mean(axis=-1)
            data_freq_norm = data_freq / data_freq.mean() * freq**(4) / freq[-1]**4
            mask = data_freq_norm > threshold

            self.data[np.where(mask)[0], :] = 0.0 

        self.data[self.always_cut, :] = 0.0
        
    def time_clean(self, threshold=5):
        """
        Does an RFI cut in time by normalizing my nu**4 to flatten bandpass and then cutting 
        above some threshold. SHOULD NOT USE THIS AT THE MOMENT, IT INTERFERES WITH FOLDING
        
        Parameters
        ----------
        data : array of np.complex128
                pulsar data array. 
        threshold :
                threshold at which to cut transient RFI. Threshold of 5 seems reasonable.
                
        Returns
        -------
        RFI-cleaned data

        """
        data = self.data
        data_time = data.mean(axis=0)
        data_time_norm = abs(data_time - np.median(data_time))
        data_time_norm /= np.median(data_time_norm)
        mask = data_time_norm > threshold
        self.data[:, np.where(mask)[0]] = 0.0 

        print "Finished cutting temporal RFI"


def derotate_far(data, RM):
    """
    Undoes Faraday rotation
    """
    freq = np.linspace(800, 400, data.shape[0]) * 1e6
    phase = np.exp(-2j * RM * (3e8 / freq)**2)
    
    if len(data.shape)==3:
        phase = phase[:, np.newaxis]

    return data * phase[:, np.newaxis]

def inj_fake_noise(data, times, cad=5):
    t_ns, p0 = np.linspace(times[0], times[-1], 
              ((times[-1] - times[0]) / cad).astype(int), retstep=True)
    tt = times.repeat(len(t_ns)).reshape(-1, len(t_ns))

    ind_ns = abs(tt - t_ns).argmin(axis=0)
    
    for corr in range(data.shape[1]):
        for k in range(6):
            data[:, corr, ind_ns[:-1]+k] += 0.02 * np.median(data[:, corr], axis=-1)[:, np.newaxis]

    return data, p0

def opt_subtraction(data, phase_axis=-1, time_axis=-2, weights=None):
    """ Performs optimal on/off subtraction on folded data

    Parameters
    ----------
    data : array_like
         Data array of visibilites, with freqs, times, phase
    phase_axis : np.int
         Axis with pulsar phase
    time_axis : np.int
         Time axis
    absval : bool
         Use time avg abs value as weight
    weights : 
         A weighting for each phase bin at each frequency
         Must have the same dimension as the data. MUST
         ALREADY BE AVERAGED IN TIME!

    Returns
    -------
    Dynamic spectrum
    """

    if weights != None:
        data_ = weights
        assert len(data_.shape) == len(data.shape)
    else:
        data_ = np.mean(abs(data.copy()), axis=time_axis, keepdims=True)

    data_sub_avg = data_ - np.mean(data_, axis=phase_axis, keepdims=True)

#   data_sub_avg = np.mean(data_sub, axis=time_axis, keepdims=True)
    weights = data_sub_avg / data_sub_avg.sum()

    return (data * weights).mean(axis=phase_axis)

def common_phasebins(PulsarPipeline, p1, p2, 
                     ngate1, ngate2, on1, on2):

    t1 = PulsarPipeline.phase_bins(p1, 0.0, ngate1)[0]
    t2 = PulsarPipeline.phase_bins(p2, 0.0, ngate2)[0]


    return np.where(t1==on1)[0], np.where(t2==on2)[0]

def plot_ddtimestream(arr_dd, times_ordered, onm):
    fig = plt.figure(figsize=(14, 10))

    arr = arr_dd[:, :, 0]
    arr[mask] = 0.0

    arr_freq_avg = arr[:, 0].sum(0) / np.where(arr[:, 0]!=0, 1, 0).sum(axis=0)

    nt = len(times_ordered)

    arr = arr.reshape(256, 4, 4, -1).sum(1)
    arr = arr[..., :nt//4 *4].reshape(256, 4, -1, 4).sum(-1)
    
    np.save('arrbo', arr[:, (0, 3)].sum(1))

    fig.add_subplot(311)
    plt.imshow((arr.real - np.mean(arr.real, axis=-1, keepdims=True))[:, (0, 3)].sum(1)\
            , aspect='auto', interpolation='nearest', 
                  vmin=0, vmax=np.std(arr[:, 0].real)*1)
    plt.xlabel("Stokes I")

    fig.add_subplot(312)
    plt.imshow(arr[:, 1]
            , aspect='auto', interpolation='nearest')
    plt.xlabel("Stokes xy")

    fig.add_subplot(313)
    plt.plot(times_ordered, arr_freq_avg)
    plt.xlabel("Time [s]")

    plt.savefig(onm)
    print onm
    
def plot_spectra(arr, onm='out.png', dd_timestream=False):
     fig = plt.figure(figsize=(14, 14))

     if dd_timestream is True:
         plt.plot(arr[0])
         plt.savefig(onm)

         return None

     print "plitting"

     arr[np.isnan(arr)] = 0.0
     arr[mask] = 0.0

     arrf = np.real(arr - np.mean(arr, axis=-1,
                               keepdims=True)).mean(-2)

     arrt = np.real(arr - np.mean(arr, axis=-1,
                               keepdims=True)).mean(0)


     fig.add_subplot(221)

     plt.imshow(arrt, interpolation='nearest',
            aspect='auto', cmap='RdBu')#, vmax=0.03, vmin=-0.01)

     plt.ylabel('times')
     plt.xlabel('phase')

     fig.add_subplot(222)
     plt.imshow(arrf, interpolation='nearest',
            aspect='auto', cmap='RdBu', vmax=arrf.max() / 5.0, vmin=arrf.min() / 5.0)
    
     plt.ylabel('freq')
     plt.xlabel('phase')
     
     fig.add_subplot(212)
     plt.plot(arrf.mean(0))

     plt.savefig(onm)

     x = arrf.mean(0)
     on = np.argmax(x)
     off = np.delete(range(len(x)), on)

     print (np.max(x)-np.median(x[off]))/ np.std(x[off])

def plot_RM(arr_ds, onm='out.png'):
   
    fig = plt.figure(figsize=(14, 14))
    plt.plot(arr_ds.mean(-1), '.')
    
    sig = np.median(arr_ds.mean(-1))
    #plt.ylim(-50 * sig, +50 * sig)
#    plt.ylim(-50, 50)
    plt.savefig(onm)


def im_RM(arr_ds):

    fig = plt.figure(figsize=(14, 14))

    vm = np.median(arr_ds)

    plt.imshow(arr_ds.real, aspect='auto', interpolation='nearest', vmax=1e14, vmin=-1e14)

    plt.colorbar()

    plt.savefig('itis.png')

src_dict = {   'B0329+54' : (0.7144900329176275, 26.833057, 53.5),
               'B1929+10' : (4.41443433443**(-1), 3.180082, 292.0),
               'B2016+28' : (0.55795, 14.176, 304.0),
               'J0341+57' : (1.888, 100.425, 55.25),
               'B0531+21' : (29.6591013546**(-1), 56.791, 83.5), # For 140509998.74 
               'B0834+06' : (1.2737682915785, 12.889, 129.25),
               'B1937+21' : (1.557806**-1, 71.040, 129.25)
               
          }
