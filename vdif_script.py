import numpy as np
import h5py 

import ReadBeamform as rbf

npack = 6e4
ntrb = 1

fn = 'b0329_3bit_v2.pcap'
#fn = 'b0329_4bit_v3.pcap'
fn = '/mnt/gamelan_test/beamforming/test_4bit.pcap'
fn = './onefeed2_0bit.pcap'
fn = './sun_2feed_0bit.pcap'
fn = './b0329+54_4bit.pcap'
fn = '/home/connor/sun_3feed_b0bit.pcap'
fn = '/home/connor/sun_2feed_0bit.pcap'
fn = '/home/connor/sun_2feeds2_0bit.pcap'
fn = './twofeed_sun22_0bit.pcap'
fn = './sun_2feed_0_july6.pcap'
fn = './virA_allfeed5_4bit_july6.pcap'
fn = './sun_2feed2_0_july6.pcap'
fn = 'sun_11.135.67.203.0bit_july6.pcap'
fn = 'b0329.0bit_july10.pcap'

arr = []
t_tot = []

print "Frame range:"
print "------------"

for ii in range(20):

     print "    ", npack*ii, npack*(ii+1)

     RB = rbf.ReadBeamform(pmin=npack*ii, pmax=npack*(ii+1))
     read_arrs = RB.read_file(fn)
     
     if read_arrs == None:
          print "Exiting"
          break

     h, d = read_arrs

     v = RB.h_index(d, h, trb=ntrb)
#    arr.append(v[:len(v)//100 * 100].reshape(-1, 100, 2, 1024).mean(1))
     arr.append(v)
     del d
     t_tot.append(RB.get_times(h))
     del h

arr = np.concatenate(arr)
t_tot = np.concatenate(t_tot)

print "here"

#f = h5py.File(outfile, 'w')
f = h5py.File('./b0329j11.hdf5', 'w')
f.create_dataset('arr', data=arr)
f.create_dataset('times', data=t_tot)
f.close()

