import numpy as np 
import h5py

import ReadBeamform as rbf

fn = 'b0329_3bit_v2.pcap'
outfile = fn + 'out.hdf5'

nframes = 1e5 + 1 

RB = rbf.ReadBeamform(pmax=nframes)

h, d = RB.read_file(fn)

print h

arr = RB.h_index(d, h, trb=1)
times = RB.get_times(h, arr[0])

del d, h

f = h5py.File(outfile, 'w')
f.create_dataset('arr', data=arr)
f.create_dataset('times', data=times)
f.close()
