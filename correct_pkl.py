import numpy as np
import h5py
import glob
import pickle

import misc_data_io as misc

slot = np.array([4, 2, 16, 14, 3, 1, 15, 13, 8, 6, 12, 10, 7, 5, 11, 9])
slot_id = slot.repeat(16)

fpga_dict = {'10161420452671572' : 8,
           '10362099439906844' : 6,
           '13547143200256028' : 1,
           '1354900185165844'  : 15,
           '13759032190185500' : 11,
           '13774983532556316' : 5,
           '15780492741586972' : 12,
           '15808512172929116' : 14, 
           '20310480648056916' : 13,
           '2269693859475484'  : 4,
           '2516293578010644'  : 7,
           '29389679799218268' : 16,
           '4521493673160724'  : 2,
           '6801312918188124'  : 3,
           '9053112731873364'  : 10,
           '9192752332517468'  : 9}


fpga_list = ['10161420452671572',
           '10362099439906844',
           '13547143200256028',
           '1354900185165844',
           '13759032190185500',
           '13774983532556316',
           '15780492741586972',
           '15808512172929116',
           '20310480648056916',
           '2269693859475484',
           '2516293578010644',
           '29389679799218268',
           '4521493673160724',
           '6801312918188124',
           '9053112731873364',
           '9192752332517468'] 

fn = '/home/chime/gains_jun19/gains_15780492741586972.pkl'

def read_pkl(fn):
     f = open(fn)   

     return pickle.load(f)

def write_pkl(fnout, data):
     output = open(fnout, 'w')
     pickle.dump(data, output)
     output.close()

def phase_mult(data_pkl, phase, inp):
 
     data_pkl[inp][1][0] *= np.exp(-1j * phase)
     data_pkl[inp][1][0] = np.round(data_pkl[inp][1][0].real)\
         + 1j * np.round(data_pkl[inp][1][0].imag)

     assert abs(data_pkl[inp][1][0].real).all() < 32768
     return list(data_pkl[inp][1][0])

def apply_gain(data_pkl, gains, N):
     phase_arr = np.angle(gains)

     for ii in range(N):
          data_pkl[ii][1][0] = (phase_mult(data_pkl, phase_arr[:, ii], ii))

     return data_pkl

def remove_gain(data, gain_pkl, nfeeds=256):
     
     for ii in range(nfeeds):
          for jj in range(ii, nfeeds):
               data[:, misc.feed_map(ii, jj, nfeeds)] \
                   /= (gain_pkl[:, ii] * np.conj(gain_pkl[:, jj]))[..., np.newaxis]

     return data

def apply_pkl_gain(data, gain_pkl, nfeeds=256):

     for ii in range(nfeeds):
          for jj in range(ii, nfeeds):
                      data[:, misc.feed_map(ii, jj, nfeeds)] \
                   *= (gain_pkl[:, ii] * np.conj(gain_pkl[:, jj]))[..., np.newaxis]

     return data

def this(infile0, infile1, data, freq=range(1024)):
     gain_mat0 = gain_pkl_mat(infile0)
     gain_mat1 = gain_pkl_mat(infile1)

     data_rm = remove_gain(data, gain_mat0[freq])      
     data_ng = apply_pkl_gain(data_rm, gain_mat1[freq])

     return data_ng

def construct_gain_mat(infile, nch, nfreq=1024, nfeed=256):
     gain_mat = np.zeros([nfreq, nfeed], np.complex128)

     nf = nfreq // nch

     for nu in range(nch):
          g = h5py.File(infile + np.str(nu) + '.hdf5', 'r')
          gx = g['ax'][:]
          gy = g['ay'][:]

          print range(nf * nu, nf*(nu+1))

          gain_mat[nf*nu:nf*(nu+1), :64] = gx[:, :64, 8:14].mean(-1)
          gain_mat[nf*nu:nf*(nu+1), 128:128+64] = gx[:, 64:, 8:14].mean(-1)

          gain_mat[nf*nu:nf*(nu+1), 64:128] = gy[:, :64, 8:14].mean(-1)
          gain_mat[nf*nu:nf*(nu+1), 128+64:] = gy[:, 64:, 8:14].mean(-1)

          print nu

     return gain_mat

def avg_channels(gain_mat, left=14, right=16):
     # Note hack to make them fit. 16::16 is one shorter than 14::16
     phi_left = np.angle(gain_mat[range(1024)[left::16][:-1], :])
     phi_right = np.angle(gain_mat[range(1024)[right::16], :])

     phi_avg = 0.5 * phi_left + 0.5 * phi_right
     
     frq_z = right - 1

     gain_mat[range(1024)[frq_z::16][:-1]] = np.exp(1j * phi_avg)

     return gain_mat
          

def gain_pkl_mat(infile):
     GGpkl = np.zeros([1024, 256], np.complex128)

     for fpga_name in fpga_list:
          data_pkl = read_pkl(infile + fpga_name + '.pkl')
          x=fpga_dict[fpga_name]

          feeds = np.where(slot_id==x)[0]#[::-1]                                                           
          feeds = feeds[ch_map]

          for i in range(16):
               GGpkl[:, feeds[i]] = data_pkl[i][1][0]
               print feeds[i]

     return GGpkl

infile = './check_gains/gains_'
#infile = './datah5/gains_'

ch_map = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]

#g = h5py.File('Jun11Gains.hdf5', 'r')
#Gains = g['gains'][:]

Gains = construct_gain_mat('transj22_cygx', 64)
Gains = avg_channels(Gains)

if __name__=='__main__':

     for fpga_name in fpga_list:
          data_pkl = read_pkl(infile + fpga_name + '.pkl')
          x=fpga_dict[fpga_name]
     
          feeds = np.where(slot_id==x)[0]#[::-1]
          feeds = feeds[ch_map]
          g = Gains[:, feeds]

          print feeds

          data_pkl = apply_gain(data_pkl, g, 16)     

          # Write pickle
          outfile =  '/home/chime/gains_jun19/gains_' + fpga_name + '.pkl'
          outfile = './check_gains_out/gains_' + fpga_name + '.pkl'
          write_pkl(outfile, data_pkl)
          print "=================================="

          print "Wrote to ", outfile 

          print "=================================="

          print ""

"""

GGpkl = np.zeros([1024, 256], np.complex128)

for fpga_name in fpga_list:
     data_pkl = read_pkl(infile + fpga_name + '.pkl')
     x=fpga_dict[fpga_name]

     feeds = np.where(slot_id==x)[0]#[::-1]                                                           
     feeds = feeds[ch_map]

     for i in range(16):
        GGpkl[:, feeds[i]] = data_pkl[i][1][0]
        print feeds[i]


     if feeds[0] in xfeeds:
        print feeds
        g = Gains[:, feeds]

        feeds2 = feeds   

"""


