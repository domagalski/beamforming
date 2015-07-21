import os

import numpy as np
import h5py 
import glob

import ReadBeamform as rbf

ntrb = 1

nfiles = 1000

f_dir = '/drives/0/baseband/20150710T195202Z_chime_beamformed/'
f_dir = '/drives/0/baseband/20150711T160938Z_chime_beamformed/'
f_dir = '/drives/0/baseband/20150716T194720Z_chime_beamformed/' # Solar both pol one feed
f_dir = '/drives/0/baseband/20150717T172614Z_chime_beamformed/'
f_dir = '/drives/0/baseband/20150717T194826Z_chime_beamformed/'
f_dir = '/drives/0/baseband/20150718T154633Z_chime_beamformed/' # One pol has ~10 beams on the same cylinder

outfile = '/drives/0/liamscratch/b0329+54_10feeds.hdf5'

flist = glob.glob(f_dir + '/*dat')
flist.sort()

arr = []

tt_tot = []

print "Frame range:"
print "------------"


RB = rbf.ReadBeamform(pmax=1e8)

for ii in range(0, 100, 20):
     print "reading %s" % flist[ii]

     if os.path.isfile(flist[ii]) is False:
          break
 
     read_arrs = RB.read_file_dat(flist[ii])
     
     if read_arrs == None:
          print "exiting"
          break

     h, d = read_arrs
     v, tt = RB.h_index(d, h, trb=ntrb)

     trb = 1

     #v = v[:(len(v)//trb)*trb].reshape(-1, trb, 2, 1024)
     #nonz = np.where(v==0, 0, 1).sum(1)

     arr.append(v[:(len(v)//trb)*trb].reshape(-1, trb, 2, 1024).mean(1))
     #arr.append(v.sum(1) / nonz)
     
     del v, d
     
     tt_tot.append(tt)

     del h

arr = np.concatenate(arr)
tt_tot = (np.concatenate(tt_tot, axis=0))[::trb]
#t_tot = RB.get_times(h_tot)

print "here"

f = h5py.File(outfile, 'w')
f.create_dataset('arr', data=arr)
f.create_dataset('times', data=tt_tot)
f.close()

print "Wrote to: ", outfile

