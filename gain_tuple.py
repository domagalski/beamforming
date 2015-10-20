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

gains = list(np.arange(25 * 10 * 2))

def getjs(gains):
     """ Produce dictionary with gains in json 
     dumpable format
     """
     return json.JSONEncoder().encode({
                         "type": "beamform_gains", 
                         "gains": gains
                         })

def array_to_tuple(Gains, nfreq=1024, nfeed=256, norm=True):

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

gains = array_to_tuple(Gains, nfreq=10, nfeed=12)

gain_dict = getjs(gains)

json.dump(gain_dict, f)

f.close()