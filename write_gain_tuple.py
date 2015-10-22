import sys

import h5py
import numpy as np
import json

def construct_gain_mat(gx, gy, nfreq=1024, nfeed=256):
     """ Take gains for x and y feeds and construct 
     full gain matrix in CHIME i.d. ordering

     Parameters:
     ----------
     gx : 
     
     gy : 
     
     Returns:
     -------
     gain_mat : array_like
        (nfreq, nfeed) complex array with gains
     """
     gain_mat = np.zeros([nfreq, nfeed], np.complex128)

     nf4 = nfeed/4

     gain_mat[:, :nf4] = gx[:, :nf4]
     gain_mat[:, nf4:2*nf4] = gy[:, :nf4]

     gain_mat[:, 2*nf4:3*nf4] = gx[:, nf4:]
     gain_mat[:, 3*nf4:] = gy[:, nf4:]

     return gain_mat

def getjs(gains):
     """ Produce dictionary with gains in json 
     dumpable format

     Parameters
     ----------
     gains : tuple
          [[[ant0 f0 re, ant0 f0 im], 
            [ant0 f1 re, ...], ..., 
            [ant255 f1023 re, ant255 f1023 im]]]
     """
     return json.JSONEncoder().encode({
                         "type": "beamform_gains", 
                         "gains": gains
                         })

def array_to_tuple(Gains, nfreq=1024, nfeed=256, norm=True):
     """ Take gains array and return tuple to be written to File

     Parameters
     ----------
     Gains : np.array
          (nfreq, nfeed) complex128 array
     nfreq : int
          number of frequencies
     nfeed : int
          number of feeds 
     norm : bool
          normalize each gain to length 1

     Returns
     -------
     tup_full : tuple
          [[[ant0 f0 re, ant0 f0 im], 
            [ant0 f1 re, ...], ..., 
            [ant255 f1023 re, ant255 f1023 im]]]
     """
     if norm is True:
          Gains /= np.abs(Gains)

     Gains[np.isnan(Gains)] = 0.0
     
     tup_full = []

     for ff in range(nfeed):
          tup = []

          for nu in range(nfreq):
               tup.append([Gains[nu, ff].real, Gains[nu, ff].imag])

          tup_full.append(tup)

     return tup_full

def tuple_to_array(gains_tuple, nfreq=1024, nfeed=256):

     Gains = np.zeros([nfreq, nfeed], np.complex128)
     
     for ff in range(nfeed):
          for nu in range(nfreq):
               Gains[nu, ff] = gains_tuple[ff][nu][0] + 1j * gains_tuple[ff][nu][1]
     
     return Gains

gn = sys.argv[1]
fn = sys.argv[2]

g = h5py.File(gn, 'r')

gx = g['gainsx']
gy = g['gainsy']

gains_mat = construct_gain_mat(gx, gy)

f = open(fn, 'w')

gains_tup = array_to_tuple(gains_mat)

gain_dict = getjs(gains_tup)

f.write(gain_dict)

f.close()
