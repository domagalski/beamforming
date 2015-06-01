import numpy as np
import h5py 

import ReadBeamform as rbf

npack = 5000
ntrb = 5000

fn = 'b0329_3bit_v2.pcap'
outfile = 'b0329_test_times.hdf5'
arr = []
t_tot = []

print "Frame range:"
print "------------"
for ii in range(20):
     print "    ", npack*ii, npack*(ii+1)
     RB = rbf.ReadBeamform(pmin=npack*ii, pmax=npack*(ii+1))
     h, d = RB.read_file(fn)
     arr.append(RB.fill_arr(h, d, trb=ntrb))
     t_tot.append(RB.get_times(h, arr[0]))

arr = np.concatenate(arr)
t_tot = np.concatenate(t_tot)

f = h5py.File(outfile, 'w')
f.create_dataset('arr', data=arr)
f.create_dataset('times', data=t_tot)
f.close()

