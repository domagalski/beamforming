import os

import numpy as np
import h5py 
import glob

import ReadBeamform as rbf

trb = 1 # Factor to rebin data array by
nfiles = 10000 # Total number of files to use
f_start = 0
f_step = 50 # step between files

f_dir = '/drives/0/baseband/20150710T195202Z_chime_beamformed/'
f_dir = '/drives/0/baseband/20150711T160938Z_chime_beamformed/'
f_dir = '/drives/0/baseband/20150716T194720Z_chime_beamformed/' # Solar both pol one feed
#f_dir = '/drives/0/baseband/20150717T172614Z_chime_beamformed/'
f_dir = '/drives/0/baseband/20150717T194826Z_chime_beamformed/'
#f_dir = '/drives/0/baseband/20150718T154633Z_chime_beamformed/' # One pol b0329+54 has ~10 beams on the same cylinder
#f_dir = '/drives/0/baseband//20150722T144418Z_chime_beamformed/' # Synthetic data from Andre
#f_dir = '/drives/0/baseband//20150722T144820Z_chime_beamformed/'
#f_dir = '/drives/0/baseband//20150722T150516Z_chime_beamformed/'
f_dir = '/drives/0/baseband/20150722T151521Z_chime_beamformed/' # B0329+54 multiple 
#f_dir = '/drives/0/baseband//20150722T211314Z_chime_beamformed/'
#f_dir = '/drives/0/baseband//20150722T221328Z_chime_beamformed/'
f_dir = '/drives/0/baseband//20150723T002219Z_chime_beamformed/' # Virgo A

outfile = '/drives/0/liamscratch/virA.hdf5'

flist = glob.glob(f_dir + '/*dat')
flist.sort()

arr = []

tt_tot = []

print "------------"
print "Frame range:"
print "------------"


RB = rbf.ReadBeamform(pmax=1e8)

for ii in range(f_start, f_start + nfiles, f_step):
     print "reading %s" % flist[ii]

     if os.path.isfile(flist[ii]) is False:
          break
 
     read_arrs = RB.read_file_dat(flist[ii])
     
     if read_arrs == None:
          print "exiting"
          break

     header, data = read_arrs
     v, tt = RB.h_index(data, header)

     arr.append(v[:(len(v)//trb)*trb].reshape(-1, trb, 2, 1024).mean(1))
     
     del v, data
     
     tt_tot.append(tt)

     del header

arr = np.concatenate(arr)
tt_tot = (np.concatenate(tt_tot, axis=0))[::trb]
#t_tot = RB.get_times(h_tot)

print "here"

f = h5py.File(outfile, 'w')
f.create_dataset('arr', data=arr)
f.create_dataset('times', data=tt_tot)
f.close()

print "Wrote to: ", outfile

