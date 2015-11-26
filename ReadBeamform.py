import os

import numpy as np 

try:
    # do *NOT* use on-disk cache; blue gene doesn't work; slower anyway
    # import pyfftw
    # pyfftw.interfaces.cache.enable()
    from pyfftw.interfaces.scipy_fftpack import (rfft, rfftfreq,
                                                 fft, ifft, fftfreq, fftshift)
    _fftargs = {'threads': int(os.environ.get('OMP_NUM_THREADS', 2)),
                'planner_effort': 'FFTW_ESTIMATE'}
except(ImportError):
    print("Consider installing pyfftw: https://github.com/hgomersall/pyFFTW")
    # use FFT from scipy, since unlike numpy it does not cast up to complex128
    from scipy.fftpack import rfft, rfftfreq, fft, ifft, fftfreq, fftshift
    _fftargs = {}


class ReadBeamform:
     """ A class to read, correlate, and process CHIME vdif data. 
     """

     def __init__(self, pmin=0, pmax=1e7):
          # Dumb way of setting total number of 
          # packets to read from each file. 1e7 should get to end.
          self.pmax = pmax 
          self.pmin = pmin
          self.nfr = 8 # Number of links 
          self.npol = 2
          self.nfq = 8 # Number of frequencies in each frame
          self.nfreq = 1024 # Total number of freq
          self.nperpacket = 625 # Time samples per packet
          self.frame_size = 5032 # Frame size in bytes
          self.freq = np.linspace(800, 400, 1024) * 1e6 # Frequency array in Hz
          self.dt = 1. / (800e6) # Sample rate pre-channelization
          self.dispersion_delay_constant = 4149. # u.s * u.MHz**2 * u.cm**3 / u.pc
          self.ntint = 2**13


     @property
     def header_dict(self):
          """ Dictionary with header info. Each entry has a
          length-three list ordered [word_number, bit_min, bit_max]. 
          i.e. the 8b word number in the 32-byte VDIF header followed 
          by the bit range within the word.
          """
          header_dict = {'time'    : [0, 0, 29],
                         'epoch'   : [1, 24, 29],
                         'frame'   : [1, 0, 23],
                         'link'    : [3, 16, 19],
                         'slot'    : [3, 20, 25],
                         'station' : [3, 0, 15],
                         'eud2'    : [5, 0, 32]
                         }

          return header_dict

     def bit_manip(self, x, k, l):
          """ Select only bits k from the right to
          l from the left.
          """
          return (x / 2**k) % 2**(l - k)

     def parse_header(self, header):
          """ Take header binary parse it 

          Returns
          -------
          station : int 
               polarization state (0 or 1)
          link : int
               freq index, increases packet to packet 
          slot : int
               node number 
          frame : int
               frame index
          time : int 
               time after reference epoch in seconds
          count : int
               fpga count               
          """
          # Should be 8 words long
          head_int = np.fromstring(header, dtype=np.uint32) 

          hdict = self.header_dict

          t_ind = hdict['time']
          frame_ind = hdict['frame']
          stat_ind = hdict['station']
          link_ind = hdict['link']
          slot_ind = hdict['slot']
          eud2_ind = hdict['eud2']

          station = self.bit_manip(head_int[stat_ind[0]], stat_ind[1], stat_ind[2])
          link = self.bit_manip(head_int[link_ind[0]], link_ind[1], link_ind[2])
          slot = self.bit_manip(head_int[slot_ind[0]], slot_ind[1], slot_ind[2])
          frame = self.bit_manip(head_int[frame_ind[0]], frame_ind[1], frame_ind[2])
          time = self.bit_manip(head_int[t_ind[0]], t_ind[1], t_ind[2])
          count = self.bit_manip(head_int[eud2_ind[0]], eud2_ind[1], eud2_ind[2])

          return station, link, slot, frame, time, count

     def open_pcap(self, fn):
          """ Reads in pcap file with dpkt package
          """
          f = open(fn)

          return dpkt.pcap.Reader(f)

     def str_to_int(self, raw):
          """ Read in data from packets as signed 8b ints

          Parameters
          ----------
          raw : binary
               Binary data to be read in 

          Returns
          -------
          data : array_like
               np.float32 arr [Re, Im, Re, Im, ...]
          """
          raw = np.fromstring(raw, dtype=np.uint8)

          raw_re = (((raw >> 4) & 0xf).astype(np.int8) - 8).astype(np.float32)
          raw_im = ((raw & 0xf).astype(np.int8) - 8).astype(np.float32)

          data = np.zeros([2*len(raw_re)], dtype=np.float32)
          data[0::2] = raw_re
          data[1::2] = raw_im

          return data

     def freq_ind(self, slot_id, link_id, frame):
          """ Get freq index (0-1024) from slot number,
          link number, and frame.
          """
          link_id = link_id[:, np.newaxis]
          frame = frame[np.newaxis]

          return slot_id + 16 * link_id + 128 * frame


     def read_file(self, fn):
          """ Get header and data from a pcap file 

          Parameters
          ----------
          fn : np.str 
               file name

          Returns
          -------
          header : array_like
               (nt, 5) array, see self.parse_header
          data : array_like
               (nt, ntfr * 2 * nfq)
          """
          pcap = self.open_pcap(fn)

          header = []
          data = []

          k = 0

          for ts, buf in pcap:
               k += 1

               if (k >= self.pmax):
                    break
               if (k < self.pmin):
                    continue

               eth = dpkt.ethernet.Ethernet(buf)
               ip = eth.data
               tcp = ip.data
               
               # Instead of tcp, open the file, read in 5032 bytes
               # after an open

               header.append(self.parse_header(tcp.data[:32]))
               data.append(self.str_to_int(tcp.data[32:])[np.newaxis])

          if len(header) >= 1:
               
               data = np.concatenate(data).reshape(len(header), -1)
               header = np.concatenate(header).reshape(-1, 6)

               return header, data

     def read_file_dat(self, fn):
          """ Get header and data from .dat file
   
          Parameters  
          ----------
          fn : np.str 
               file name                                                                 

          Returns
          -------                                                                                                                                          
          header : array_like
               (nt, 6) array, see self.parse_header
          data : array_like 
               (nt, ntfr * 2 * nfq)                                                                                                                                                     
          """
          fo = open(fn)

          header = []
          data = []
          
          for k in range(np.int(self.pmax)):

               data_str = fo.read(self.frame_size)

               if len(data_str) == 0:
                    print "Fin File"
                    break

               header.append(self.parse_header(data_str[:32]))
               data.append(self.str_to_int(data_str[32:])[np.newaxis])

          if len(header) >= 1:

               data = np.concatenate(data).reshape(len(header), -1)
               header = np.concatenate(header).reshape(-1, 6)

               return header, data

     def J2000_to_unix(self, t_j2000):
          """ Takes seconds since J2000 and returns 
          a unix time
          """
          J2000_unix = 946728000.0

          return t_j2000 + J2000_unix

     def rebin_time(self, arr, trb):
          """ Rebin data array in time
          """
          nt = arr.shape[0]
          rbshape = (nt//trb, trb, ) + arr.shape[1:]

          arr = arr[:nt // trb * trb].reshape(rbshape)

          return arr.mean(1)

     def get_times(self, header, seq=True):
          """ Takes two time columns of header (seconds since
          J2000 and packet number) and constructs time array in
          seconds

          Parameters
          ----------
          header : 

          seq : boolean
               If True, use fpga sequence number. Else use vdif timestamp
          """
          times = header[:, -3] / np.float(self.nperpacket) + header[:, -2].astype(np.float)

          if seq is True:
               seq = header[:, -1] 
               times = (seq - seq[0]) / 625.0**2 + times[0]

          return self.J2000_to_unix(times)

     def get_fft_freq(self, ntint, dm):
          dtsample = 2 * self.nfreq * self.dt

          fcoh = self.freq - fftfreq(
                    ntint, dtsample)[:, np.newaxis]
     
          _fref = self.freq[np.newaxis]

          dang = (self.dispersion_delay_constant * dm * fcoh *
                              (1./_fref - 1./fcoh)**2) 

          dd_coh = np.exp(-1j * dang).astype(np.complex64)

          return dd_coh


     def correlate_xy(self, data_pol0, data_pol1, header, indpol0, indpol1):

          seq0 = header[indpol0, -1]
          seq1 = header[indpol1, -1]

          XYreal = []
          XYimag = []
          
          seq_xy = []

          data_rp0 = data_pol0.real
          data_ip0 = data_pol0.imag

          data_rp1 = data_pol1.real
          data_ip1 = data_pol1.imag

          for t0, tt in enumerate(seq0):

               t1 = np.where(seq1 == tt)[0]

               if len(t1) < 1:
                    continue

               seq_xy.append(tt)

               xyreal = data_rp0[t0] * data_rp1[t1] + data_ip0[t0] * data_ip1[t1]
               xyimag = data_ip0[t0] * data_rp1[t1] - data_rp0[t0] * data_ip1[t1]

               XYreal.append(xyreal)
               XYimag.append(xyimag)

          times = header[indpol0, -3] / np.float(self.nperpacket) 
          times += header[indpol0, -2].astype(np.float)

          tt_xy = (seq_xy - seq_xy[0]) / 625.0**2 + times[0]

          return XYreal, XYimag, self.J2000_to_unix(tt_xy)

     def correlate_and_fill(self, data, header, trb=1):
          """ Take header and data arrays and reorganize
          to produce the full time, pol, freq array

          Parameters
          ----------
          data : array_like
               (nt, ntfr * 2 * self.nfq) array of nt packets
          header : array_like
               (nt, 5) array, see self.parse_header
          ntimes : np.int
               Number of packets to use

          Returns 
          -------
          arr : array_like (duhh) np.float64
               (ntimes * ntfr, npol, nfreq) array of autocorrelations
          tt : array_like 
               Same shape as arr, since each frequency has its own time vector
          """

          slots = set(header[:, 2])

          print "Data has", len(slots), "slots: ", slots


          data = data[:, ::2] + 1j * data[:, 1::2]

          data_corr = data.real**2 + data.imag**2
          data_corr = data_corr.reshape(-1, self.nperpacket, 8).mean(1)

          data_real = data.real.reshape(-1, self.nperpacket, 8).transpose((0, 2, 1))
          data_imag = data.imag.reshape(-1, self.nperpacket, 8).transpose((0, 2, 1))

          arr = np.zeros([data_corr.shape[0] / self.nfr / 2 / len(slots) + 256
                                   , 2*self.npol, self.nfreq], np.float64)

          tt = np.zeros([data_corr.shape[0] / self.nfr / 2 / len(slots) + 256
                                   , 2*self.npol, self.nfreq], np.float64)

          for qq in range(self.nfr):

               for ii in slots:

                    fin = ii + 16 * qq + 128 * np.arange(8)

                    indpol0 = np.where((header[:, 0]==0) & \
                                            (header[:, 1]==qq) & (header[:, 2]==ii))[0]

                    indpol1 = np.where((header[:, 0]==1) & \
                                            (header[:, 1]==qq) & (header[:, 2]==ii))[0]
                    
                    inl = min(len(indpol0), len(indpol1))

                    if inl < 1:
                         continue

                    indpol0 = indpol0[:inl]
                    indpol1 = indpol1[:inl]
                    
                    XYreal, XYimag, tt_xy = self.correlate_xy(
                                 data[indpol0], data[indpol1], header, indpol0, indpol1)

                    XYreal = np.concatenate(XYreal, axis=0).reshape(-1, self.nperpacket, 8)
                    XYimag = np.concatenate(XYimag, axis=0).reshape(-1, self.nperpacket, 8)


                    arr[:len(indpol0), 0, fin] = data_corr[indpol0]
                    arr[:len(indpol1), 3, fin] = data_corr[indpol1]

                    arr[:len(XYreal), 1, fin] = XYreal.mean(1)
                    arr[:len(XYimag), 2, fin] = XYimag.mean(1)

                    tt[:len(tt_xy), 1, fin] = np.array(tt_xy).repeat(8).reshape(-1, 8)
                    tt[:len(tt_xy), 2, fin] = tt[:len(tt_xy), 1, fin].copy()

                    if (len(indpol0) >= 1) and (len(indpol0) < arr.shape[0]): 
                         tt[:len(indpol0), 0, fin] = self.get_times(\
                                         header[indpol0]).repeat(8).reshape(-1, 8)

                    if (len(indpol1) >= 1) and (len(indpol1) < arr.shape[0]):
                         tt[:len(indpol1), 3, fin] = self.get_times(\
                                         header[indpol1]).repeat(8).reshape(-1, 8)

          return arr, tt


     def correlate_and_fill_cohdd(self, data, header, p0, dm, ngate=64, trb=1):
          """ Take header and data arrays and reorganize
          to produce the full time, pol, freq array after
          coherently dedispersing timestream

          Parameters
          ----------
          data : array_like
               (nt, ntfr * 2 * self.nfq) array of nt packets
          header : array_like
               (nt, 5) array, see self.parse_header
          dm : np.float
               dispersion measure pc cm**-3
          ntimes : np.int
               Number of packets to use

          Returns 
          -------
          arr : array_like (duhh) np.float64
               (ntimes * ntfr, npol, nfreq) array of autocorrelations
          tt : array_like 
               Same shape as arr, since each frequency has its own time vector
          """

          slots = set(header[:, 2])

          print "Data has", len(slots), "slots: ", slots

          data = data[:, ::2] + 1j * data[:, 1::2]

          fold_arr = np.zeros([self.nfreq, self.npol**2, data.shape[0]//ngate, ngate], np.float32)

          # Precalculate the coh dedispersion shift phases
          dd_coh = self.get_fft_freq(self.ntint, dm)

          for qq in range(self.nfr):

               for ii in slots:
                    fin = ii + 16 * qq + 128 * np.arange(8)

                    indpol0 = np.where((header[:, 0]==0) & \
                                            (header[:, 1]==qq) & (header[:, 2]==ii))[0]

                    indpol1 = np.where((header[:, 0]==1) & \
                                            (header[:, 1]==qq) & (header[:, 2]==ii))[0]
                    
                    inl = min(len(indpol0), len(indpol1))
                    inm = max(len(indpol0), len(indpol1))

                    if inl < 1:
                         continue

                    seq0 = header[indpol0, -1]
                    seq1 = header[indpol1, -1]

                    frames0 = (seq0 - seq0[0]) / self.nperpacket
                    frames1 = (seq1 - seq1[0]) / self.nperpacket
                    
                    data0 = np.zeros([frames0.max()+1, self.nperpacket * 8], dtype=data.dtype)
                    data1 = np.zeros([frames1.max()+1, self.nperpacket * 8], dtype=data.dtype)

                    #seq0 = np.linspace(seq0[0], seq0[-1], len(data0) * self.nperpacket)                    
                    #seq1 = np.linspace(seq1[0], seq1[-1], len(data1) * self.nperpacket)                    

                    seq0 = np.arange(seq0[0], seq0[-1] + 1)
                    seq1 = np.arange(seq1[0], seq1[-1] + 1)

                    # Make sure we can safely FFT for coherent dedispersion
                    data0_ = data[indpol0]#.reshape(-1, 8)
                    data1_ = data[indpol1]#.reshape(-1, 8)

                    data0[frames0] = data0_
                    data1[frames1] = data1_

                    data0.shape = (-1, 8)
                    data1.shape = (-1, 8)

                    dropped_pack0 = np.where(np.diff(seq0)!=self.nperpacket)[0]
                    dropped_pack1 = np.where(np.diff(seq1)!=self.nperpacket)[0]

#                    print len(dropped_pack0), len(dropped_pack1)
#                    print seq0.shape, seq1.shape, data0.shape, data1.shape

                    """for dp in dropped_pack0:
                        fill_pack = np.arange(seq0[dp] + self.nperpacket, seq0[dp+1], self.nperpacket)
                        print dp, seq0[dp+1] - seq0[dp], fill_pack
                        seq0 = np.insert(seq0, dp+1, fill_pack)
                        data0 = np.insert(data1, dp+1, 
                                          np.repeat(0.0, len(fill_pack))[:, np.newaxis], axis=0)

                    for dp in dropped_pack1:
                        fill_pack = np.arange(seq1[dp] + self.nperpacket, seq1[dp+1], self.nperpacket)
                        seq1 = np.insert(seq1, dp+1, fill_pack)
                        data1 = np.insert(data1, dp+1, 
                                          np.repeat(0.0, len(fill_pack))[:, np.newaxis], axis=0)

#                    print seq0.shape, seq1.shape, data0.shape, data1.shape
                    

                    if not (np.diff(seq0)==self.nperpacket).all():
                         print "Dropped packet, shouldn't fft"
                         continue
                    """ 


                    data0 = data0[:self.ntint]
                    data1 = data1[:self.ntint]

                    data_dechan0 = fft(data0, axis=0, overwrite_x=True)
                    data_dechan1 = fft(data1, axis=0, overwrite_x=True)

                    data_dechan0 *= dd_coh[:, fin]
                    data_dechan1 *= dd_coh[:, fin]

                    data0 = ifft(data_dechan0, axis=0, overwrite_x=True)
                    data1 = ifft(data_dechan1, axis=0, overwrite_x=True)

                    data_corr0 = data0.real**2 + data0.imag**2
                    data_corr1 = data1.real**2 + data1.imag**2

                    indpol0 = indpol0[:inl]
                    indpol1 = indpol1[:inl]

                    XYreal, XYimag, timesxy = self.correlate_xy(
                                 data0, data1, header, indpol0, indpol1)

                    XYreal = np.concatenate(XYreal, axis=0)
                    XYimag = np.concatenate(XYimag, axis=0)

                    times0 = self.get_times(header[indpol0], seq=False)[0] \
                                        + (seq0 - seq0[0]) / 625.0**2
                    times1 = self.get_times(header[indpol1], seq=False)[0] \
                                        + (seq0 - seq0[0]) / 625.0**2

                    print times0[:10], np.diff(times0)[0:5]  

                    bins0 = (((times0 / p0) % 1) * ngate).astype(np.int)      
                    bins1 = (((times1 / p0) % 1) * ngate).astype(np.int)  

                    print bins0.shape, data_corr0.shape
                    icount[fin, 0, ti] = np.bincount(bins0, data_corr0 != 0., ngate)    

                    icount[fin, 1, ti] = np.bincount(binsxy, XYreal != 0., ngate)    

                    icount[fin, 2, ti] = np.bincount(binsxy, XYimag != 0., ngate)  

                    icount[fin, 3, ti] = np.bincount(bins1, data_corr1 != 0., ngate)   

                    fold_arr[fi, 0, ti, :] = np.bincount(bins0, 
                                        weights=data_corr0, minlength=ngate)

                    fold_arr[fi, 1, ti, :] = np.bincount(binsxy, 
                                        weights=XYreal, minlength=ngate)

                    fold_arr[fi, 2, ti, :] = np.bincount(binxy, 
                                        weights=XYimag, minlength=ngate)

                    fold_arr[fi, -1, ti, :] = np.bincount(bins1, 
                                        weights=data_corr1, minlength=ngate)


          return fold_arr, icount


     def correlate_and_reorg(self, header, data):
          """ 
          """
          assert data.shape[-1] == 8

          seq = list(set(header[:, -1]))
          seq.sort()


          npackets = (seq[-1] - seq[0]) / 625

#          times = (seq - seq[0]) / 625.**2 #+ self.get_times(header[0])

          seq_f = np.arange(seq[0], seq[-1], 625)

          Arr = np.zeros([npackets, 2, self.nfreq])

          for pp in range(self.npol):
               for qq in range(self.nfr):
                    for ii in range(16):
                         for tt in range(len(seq_f)):
                              ind = np.where((header[:, 0]==pp) & (header[:, 1]==qq) & \
                                                (header[:, 2]==ii) & (header[:, -1]==seq_f[tt]))[0]

                              fin = ii + 16 * qq + 128 * np.arange(8)
                         
                              if len(ind) != 1:
                                   continue
                         
                              Arr[tt, pp, fin] = data[ind]

          return Arr

     def fill_arr(self, header, data, ntimes=None, trb=1):
          """ Take header and data arrays and reorganize
          to produce the full time, pol, freq array

          Parameters
          ----------
          header : array_like
               (nt, 5) array, see self.parse_header
          data : array_like
               (nt, ntfr * 2 * self.nfq) array of nt packets
          ntimes : np.int
               Number of packets to use

          Returns 
          -------
          arr : array_like (duhh) np.float64
               (ntimes * ntfr, npol, nfreq) array of autocorrelations
          """

          ntfr = data.shape[-1] // (2 * self.nfq)

          data_corr = data[:ntimes, 0::2]**2 + data[:ntimes, 1::2]**2
          ntimes = data_corr.shape[0]

          header = header[:ntimes]

          data_corr = data_corr[:ntimes // (self.nfr*self.npol) * self.nfr 
                                 ].reshape(-1, self.nfr, self.npol, ntfr, self.nfq)

          data_corr = data_corr.transpose((0, 3, 2, 1, 4)
                                 ).reshape(-1, self.npol, self.nfr * self.nfq)

          pos = np.arange(self.nfr)
          f_ind = self.freq_ind(header[0, 2], np.arange(self.nfq), pos).reshape(-1)

          arr = np.zeros([data_corr.shape[0], self.npol, self.nfreq], np.float64)
          arr[..., f_ind] = data_corr

          del data_corr

          return self.rebin_time(arr, trb)

     def corrbin(self, data):
          """ Take data, square and sum to produce
          two autocorrelations.
          """

          data_corr = data[:, 0::2]**2 + data[:, 1::2]**2

          data_corr = data_corr.reshape(-1, 625, 8).mean(1)

          return data_corr

def MJD_to_unix(MJD):
     
     return (MJD + 2400000.5 - 2440587.5) * 86400.0

def unix_to_MJD(t_unix):
     
     return (t_unix / 86400.0) + 2440587.5 - 2400000.5

