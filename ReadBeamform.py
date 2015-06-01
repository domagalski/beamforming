import numpy as np 
import dpkt


class ReadBeamform:

     def __init__(self, pmin=0, pmax=1000):
          self.pmax = pmax # Quick fix. Effectively no upper limit.
          self.pmin = pmin
          self.nfr = 8 # Number of links 
          self.npol = 2
          self.nfq = 8 # Number of frequencies in each frame
          self.nfreq = 1024 # Total number of freq

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
                         'station' : [3, 0, 15]
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
          """
          # Should be 8 words long
          head_int = np.fromstring(header, dtype=np.uint32) 

          hdict = self.header_dict

          t_ind = hdict['time']
          frame_ind = hdict['frame']
          stat_ind = hdict['station']
          link_ind = hdict['link']
          slot_ind = hdict['slot']

          station = self.bit_manip(head_int[stat_ind[0]], stat_ind[1], stat_ind[2])
          link = self.bit_manip(head_int[link_ind[0]], link_ind[1], link_ind[2])
          slot = self.bit_manip(head_int[slot_ind[0]], slot_ind[1], slot_ind[2])
          frame = self.bit_manip(head_int[frame_ind[0]], frame_ind[1], frame_ind[2])
          time = self.bit_manip(head_int[t_ind[0]], t_ind[1], t_ind[2])

          return station, link, slot, frame, time

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
          data[::2] = raw_re
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

               if ((k >= self.pmax) or (k < self.pmin)):
                    continue

               eth = dpkt.ethernet.Ethernet(buf)
               ip = eth.data
               tcp = ip.data

               header.append(self.parse_header(tcp.data[:32]))
               data.append(self.str_to_int(tcp.data[32:])[np.newaxis])

          data = np.concatenate(data).reshape(len(header), -1)
          header = np.concatenate(header).reshape(-1, 5)

          return header, data

     def get_times(self, header, data_bin, os=0):
          nframes = header.shape[0]
          t_sec = nframes / 625.0 / self.npol / self.nfr
          
          return np.linspace(os * t_sec, (os+1) * t_sec, data_bin.shape[0])

     def rebin_time(self, arr, trb):
          """ Rebin data array in time
          """
          nt = arr.shape[0]
          rbshape = (nt//trb, trb, ) + arr.shape[1:]

          arr = arr[:nt // trb * trb].reshape(rbshape)

          return arr.mean(1)


     def h_index(self, data, header, trb=1):
          data_corr = data[:, 0::2]**2 + data[:, 1::2]**2

          data_corr = data_corr.reshape(-1, 625, 8).mean(1)

          arr = np.zeros([data_corr.shape[0] / self.nfr / 2 + 1
                                   , self.npol, self.nfreq], np.float32)

          for pp in range(self.npol):
               for qq in range(self.nfr):
                    ind = np.where((header[:, 0]==pp) & (header[:, 1]==qq))[0]
                    fin = header[0, 2] + 16 * qq + 128 * np.arange(8)

                    arr[:len(ind), pp, fin] = data_corr[ind]

          del data_corr

          return self.rebin_time(arr, trb) 


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


