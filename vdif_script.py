import os

import numpy as np
import h5py 
import glob

import ReadBeamform as rbf

ntrb = 1

nfiles = 1000

f_dir = '/drives/0/baseband/20150710T195202Z_chime_beamformed/'
f_dir = '/drives/0/baseband/20150711T160938Z_chime_beamformed/'

flist = glob.glob(f_dir + '/*dat')
flist.sort()

arr = []
t_tot = []
h_tot = []

print "Frame range:"
print "------------"


RB = rbf.ReadBeamform(pmax=1e6)

for ii in range(100):
     print "reading %s" % flist[ii]

     if os.path.isfile(flist[ii]) is False:
          break
 
     read_arrs = RB.read_file_dat(flist[ii])
     
     if read_arrs == None:
          print "exiting"
          break

     h, d = read_arrs
     v, tt = RB.h_index(d, h, trb=ntrb)

     trb = 10

     arr.append(v[:(len(v)//trb)*trb].reshape(-1, trb, 2, 1024).mean(1))

#     tt = RB.get_times(h)
     
     del v, d
     
     h_tot.append(tt)

     del h

arr = np.concatenate(arr)
h_tot = np.concatenate(h_tot, axis=0)
#t_tot = RB.get_times(h_tot)

print "here"

outfile = '/drives/0/liamscratch/b0329.hdf5'

f = h5py.File(outfile, 'w')
f.create_dataset('arr', data=arr)
#f.create_dataset('times', data=t_tot)
f.create_dataset('header', data=h_tot)
f.close()

print "Wrote to: ", outfile

