import os

import numpy as np
import h5py 
import glob

import ReadBeamform as rbf

npol = 2
nfreq = 1024
trb = 1 # Factor to rebin data array by
nfiles = 100 # Total number of files to use
f_start = 3840
f_step = 1 # step between files
n_save = 20 # Save down every n_save files

f_dir = '/drives/0/baseband/20150710T195202Z_chime_beamformed/'
f_dir = '/drives/0/baseband/20150711T160938Z_chime_beamformed/'
f_dir = '/drives/0/baseband/20150716T194720Z_chime_beamformed/' # Solar both pol one feed
#f_dir = '/drives/0/baseband/20150717T172614Z_chime_beamformed/'
f_dir = '/drives/0/baseband/20150717T194826Z_chime_beamformed/'
#f_dir = '/drives/0/baseband/20150718T154633Z_chime_beamformed/' # One pol b0329+54 has ~10 beams on the same cylinder
#f_dir = '/drives/0/baseband//20150722T144418Z_chime_beamformed/' # Synthetic data from Andre
f_dir = '/drives/0/baseband/20150722T151521Z_chime_beamformed/' # B0329+54 multiple 
f_dir = '/drives/0/baseband//20150723T002219Z_chime_beamformed/' # Virgo A
f_dir = '/home/connor/'

outfile = './testout'

flist = glob.glob(f_dir + '/*dat')
flist.sort()

arr = []

tt_tot = []

print "------------"
print "Frame range:"
print "------------"

RB = rbf.ReadBeamform(pmax=1e8)

def write_h5(outfile, arr, tt_tot):
     f = h5py.File(outfile, 'w')
     f.create_dataset('arr', data=arr)
     f.create_dataset('times', data=tt_tot)
     f.close()
     
     print "Wrote to: ", outfile


for ii in range(f_start, f_start + nfiles, f_step):

     k = ii - f_start + 1
     
     try:
          print "reading %s" % flist[ii]
     except IndexError:
          break
          
     if os.path.isfile(flist[ii]) is False:
          break

     read_arrs = RB.read_file_dat(flist[ii])
     
     if read_arrs == None:
          print "exiting"
          break

     header, data = read_arrs

     data = RB.corrbin(data)

     v, tt = RB.h_index(data, header)

     arr.append(v[:(len(v)//trb)*trb].reshape(-1, trb, 2, 1024).mean(1))
     
     del v, data
     
     tt_tot.append(tt)

     del header

     if (ii == f_start + nfiles -1) or ((k % n_save) == 0):
          if trb == 1:
               arr = np.concatenate(arr)
               tt_tot = (np.concatenate(tt_tot, axis=0))
               write_h5(outfile + np.str(ii) + '.hdf5', arr, tt_tot)

               arr = []
               tt_tot = []

               continue

          elif trb > 1:
               print "rebinning by :", trb
               arr = np.concatenate(arr)
               arr = arr[:(len(arr) // trb) * trb].reshape(-1, trb, npol, nfreq)
               nonz = np.where(arr != 0.0, 1, 0.0)

               arr = arr.sum(1) / nonz.sum(1)
               arr[np.isnan(arr)] = 0.0
               tt_tot = (np.concatenate(tt_tot, axis=0))[trb/2::trb]
               write_h5(outfile + np.str(ii) + '.hdf5', arr, tt_tot)

               arr = []
               tt_tot = []
          else:
               raise Exception("Needs to be integer > 0")

     
