import sys

import h5py
import numpy as np
import json

gn = sys.argv[1]
fn = sys.argv[2]

g = h5py.File(gn,'r')

gx = g['gainsx'][:]
gy = g['gainsy'][:]

Gains = np.concatenate((gx, gy), axis=-1)

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


f = open(fn, 'w')

gains = array_to_tuple(Gains, nfreq=1024, nfeed=256)

gain_dict = getjs(gains)

f.write(gain_dict)

f.close()