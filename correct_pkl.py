import numpy as np
import h5py
import glob
import pickle

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

fn = '/home/chime/gains_15780492741586972.pkl'

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

infile = '/home/chime/gains/gains_'
#infile = './datah5/gains_'

ch_map = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]

g = h5py.File('May16Gains.hdf5', 'r')
Gains = g['gains'][:]

for fpga_name in fpga_list:
     data_pkl = read_pkl(infile + fpga_name + '.pkl')
     x=fpga_dict[fpga_name]
     
     feeds = np.where(slot_id==x)[0]#[::-1]
     feeds = feeds[ch_map]
     g = Gains[:, feeds]

     print feeds

     data_pkl = apply_gain(data_pkl, g, 16)     

     # Write pickle
     outfile =  './datah5/gains_' + fpga_name + '.pkl'
     write_pkl(outfile, data_pkl)
     print "=================================="

     print "Wrote to ", outfile 

     print "=================================="

     print ""







