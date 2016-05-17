# Code to take gain solutions and generate
# .pkl files to be fed to Pathfinder F-engine. 
# Because CHIME pathfinder has a lower-triangle X-engine
# and a conjugate-FFT F-engine, the final visibilities
# end up with the phase you'd expect, however the 
# intermediate steps can be tricky. 
#
# For this reason, note that if phase_solver_code.remove_fpga_gains
# is used with triu=False, then correct_pkl.phase_mult_remove_original_phase
# the line "data_pkl[inp][1][0] *= np.exp(+1j * phase)" should 
# indeed have +1j, and not -1j. This worked between Oct 2015 - May 2016
#


import numpy as np
import h5py
import glob
import pickle

import misc_data_io as misc
import ch_util.ephemeris as eph
import phase_solver_code as pc
reload(pc)

slot = np.array([4, 2, 16, 14, 3, 1, 15, 13, 8, 6, 12, 10, 7, 5, 11, 9])
slot_id = slot.repeat(16)

fpga_dict = {'8': 8,
           '6' : 6,
           '1'  : 1,
           '3' : 3,
           '11' : 11,
           '5' : 5,
           '12' : 12,
           '14' : 14,
           '13' : 13,
           '4'  : 4,
           '7'  : 7,
           '16' : 16,
           '2'  : 2,
           '15' : 15,
           '10'  : 10,
           '9'  : 9}

fpga_dict2  = {'0008': 16,
              '0014': 14,
              '0016': 10,
              '0018': 6,
              '0019': 3,
              '0023': 15,
              '0024': 8,
              '0025': 11,
              '0026': 2,
              '0027': 13,
              '0030': 4,
              '0031': 5,
              '0032': 12,
              '0055': 9,
              '0063': 7,
              '0068': 1}


fpga_list = ['10161420452671572',
           '10362099439906844',
           '6897813178691612',
           '20522120145121364',
           '13759032190185500',
           '13774983532556316',
           '15780492741586972',
           '15808512172929116',
           '20310480648056916',
           '2269693859475484',
           '2516293578010644',
           '29389679799218268',
           '4521493673160724',
           '31710750469369940',
           '7019893205381148',
           '9192752332517468']

fpga_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']


def read_pkl(fn):
     f = open(fn)   

     return pickle.load(f)

def write_pkl(fnout, data):
     output = open(fnout, 'w')
     pickle.dump(data, output)
     output.close()

def phase_mult_remove_original_phase(data_pkl, phase, inp):
     phase[np.isnan(phase)] = 0.0

     # Force the pkl carrier to be purely Real                                                                                                                        
     data_pkl[inp][1][0] *= np.exp(-1j * np.angle(data_pkl[inp][1][0]))

     # Now remove instrumental phase from carrier pkl                                                                                                                 
     data_pkl[inp][1][0] *= np.exp(+1j * phase)


     phase[np.isnan(phase)] = 0.0

     # Force the pkl carrier to be purely Real
     data_pkl[inp][1][0] *= np.exp(-1j * np.angle(data_pkl[inp][1][0]))

     # Now remove instrumental phase from carrier pkl 
     data_pkl[inp][1][0] *= np.exp(+1j * phase)
     data_pkl[inp][1][0] = np.round(data_pkl[inp][1][0].real)\
         + 1j * np.round(data_pkl[inp][1][0].imag)

     # Gains cannot be larger than 2**15 - 1
     assert abs(data_pkl[inp][1][0].real).all() < 32767
     assert abs(data_pkl[inp][1][0].imag).all() < 32767

     return list(data_pkl[inp][1][0])

def phase_mult(data_pkl, phase, inp):
     
     data_pkl[inp][1][0] *= np.exp(-1j * phase)

     # Seems to have been changed from -phi to +phi already
     data_pkl[inp][1][0] = np.round(data_pkl[inp][1][0].real)\
         + 1j * np.round(data_pkl[inp][1][0].imag)

     assert abs(data_pkl[inp][1][0].real).all() < 32768
     return list(data_pkl[inp][1][0])

def apply_gain(data_pkl, gains, N):
     phase_arr = np.angle(gains)

     for ii in range(N):
          data_pkl[ii][1][0] = (phase_mult_remove_original_phase(data_pkl, phase_arr[:, ii], ii))

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


def construct_gain_mat(gx, gy, nch, nfreq=1024, nfeed=256):
     """ Take gains for x and y feeds and construct 
     full gain matrix in CHIME i.d. ordering

     Parameters:
     ----------
     gx : 
     
     gy : 
     
     nch : int
         Number of chunks

     Returns:
     -------
     gain_mat : array_like
        (nfreq, nfeed) complex array with gains
     """
     gain_mat = np.zeros([nfreq, nfeed], np.complex128)

     nf = nfreq // nch

     for nu in range(nch):


          gain_mat[nf*nu:nf*(nu+1), :64] = gx[nf*nu:nf*(nu+1), :64]#.mean(-1)
          gain_mat[nf*nu:nf*(nu+1), 128:128+64] = gx[nf*nu:nf*(nu+1), 64:]#.mean(-1)

          gain_mat[nf*nu:nf*(nu+1), 64:128] = gy[nf*nu:nf*(nu+1), :64]#.mean(-1)
          gain_mat[nf*nu:nf*(nu+1), 128+64:] = gy[nf*nu:nf*(nu+1), 64:]#.mean(-1)


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
     """ Read in gain*pkl files and construct an ordered
     gain matrix out of them.
     """

     GGpkl = np.zeros([1024, 256], np.complex128)

     for fpga_name in fpga_list:
          data_pkl = read_pkl(infile + fpga_name + '.pkl')
          x = fpga_dict[fpga_name]

          feeds = np.where(slot_id==x)[0]#[::-1]                                                           
          feeds = feeds[ch_map]
          
          for i in range(16):
               GGpkl[:, feeds[i]] = data_pkl[i][1][0]

     return GGpkl

def check_gain_solution(infile_pkl, infile_h5, feeds, src, freq=305, transposed=True):

    dfs = pc.fringestop_and_sum(infile_h5, feeds,
                   freq, src, transposed=transposed,
                                return_unfs=True, meridian=False)[-2]

    dfs = dfs[0].transpose()
    ntimes = dfs.shape[0]

#    Gpkl = gain_pkl_mat('./inp_gains/gains_slot')
#    Gpkl = Gpkl[freq]

    f = h5py.File(infile_h5, 'r')                                                                                              

    if transposed is True:
         g = f['gain_coeff'][freq, :, 0]
         Gh5 = g['r'] + 1j * g['i']
    else:
         g = f['gain_coeff'][0, freq]
         Gh5 = g['r'] + 1j * g['i']
    
#    for i in range(len(feeds)):
#         for j in range(i, len(feeds)):
#              dfs[:, misc.feed_map(i, j, 256)] *= np.exp(-1j * np.angle(Gh5[i] * np.conj(Gh5[j])))
              #dfs[:, misc.feed_map(i, j, 256)] *= np.exp(1j * np.angle(Gpkl[i] * np.conj(Gpkl[j])))

    return dfs, Gh5

def fs_and_correct_gains(fn_h5, fn_gain, src, freq=305, \
                              del_t = 900, transposed=True,\
                              remove_fpga_phase=True, remove_instr=True,
                              remove_fpga=False):

    dfs = pc.fs_from_file(fn_h5, freq, src,
             del_t=1800, transposed=transposed)

    ntimes = dfs.shape[0]

    fg = h5py.File(fn_gain, 'r')

    gx = fg['gainsx']
    gy = fg['gainsy']

    gain_mat = construct_gain_mat(gx, gy, 64)[freq]

    f = h5py.File(fn_h5, 'r')
    feeds = range(256)

    # Skip loading the gains and applying them if both are False
    if (remove_fpga is False) and \
         (remove_instr is False) and (remove_fpga_phase is False):
         return dfs

    if transposed is True:
         g = f['gain_coeff'][freq, :, 0]
         Gh5 = g['r'] + 1j * g['i']
    else:
         g = f['gain_coeff'][0, freq]
         Gh5 = g['r'] + 1j * g['i']

    print np.isnan(Gh5).sum() / np.float(len(Gh5.flatten()))
    print "doing the big loop"
    print gain_mat.sum()

    for fi, nu in enumerate(freq):

         for i in range(len(feeds)):

              for j in range(i, len(feeds)):

                   # If solved with triu=False then 
                   # remove_fpga should be +1j and instr should be -1j
                   if remove_fpga is True:
                        dfs[:, misc.feed_map(i, j, 256)] \
                             /= np.conj((Gh5[fi, i]) * np.conj(Gh5[fi, j]))

                   if remove_fpga_phase is True:
                        # Remove fpga phases written in .h5 file
                        dfs[:, misc.feed_map(i, j, 256)] *= \
                            np.exp(+1j * np.angle(Gh5[fi, i] * np.conj(Gh5[fi, j])))

                   if remove_instr is True:
                        # Apply gains solved for 
                        dfs[:, misc.feed_map(i, j, 256)] \
                            *= np.exp(-1j * np.angle(gain_mat[fi, i]\
                                       * np.conj(gain_mat[fi, j])))
    
    print "Summed h5 gains: ", Gh5.sum()

    return dfs

"""
def check_gain_solution(infile_pkl, infile_h5, freq=305, transposed=True):
    Gpkl = gain_pkl_mat(infile_pkl)

    f = h5py.File(infile_h5, 'r')
     
    if transposed is True:
        r = andata.Reader(fn)
        r.freq_sel = freq
        X = r.read()
        times = r.time

        g = f['gain_coeff'][0]               
    else:
        f = h5py.File(fn, 'r')  
        times = f['index_map']['time'].value['ctime']
     
        g = f['gain_coeff'][..., 0]
        
    src_trans = eph.transit_times(src, times[0])

     t_range = np.where((times < src_trans + del_t) & (times > src_trans - del_t))[0]
     inp = pc.gen_inp()[0]

     data = X.vis[:, :, t_range[0]:t_range[-1]]
     freq = X.freq

     times = times[t_range[0]:t_range[-1]]

     ra = eph.transit_RA(times)

     dfs = fringestop_pathfinder(data.copy(), ra, freq, inp, src)

     # Order of real / imag seems to be switched. Must correct for this.
     Gh5 = g['i'] + 1j * g['r'] 
"""

def compare_pkl_to_h5(fn, input_pkls, trans=False):
     f = h5py.File(fn, 'r')

     if trans is False:
          g = f['gain_coeff'][0]
     else:
          g = f['gain_coeff'][..., 0]

     Gains = g['r'] + 1j * g['i']
     
     phases_h5 = np.angle(Gains)
     
     Gain_pkl = gain_pkl_mat(input_pkls)

     for ii in range(256):
          print np.degrees(np.angle(Gain_pkl[305, ii])) + np.degrees(np.angle(Gains[305, ii]))
     
     return data_pkl

ch_map = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
#fn = '/mnt/gamelan/untransposed/20150903T172424Z_pathfinder_corr/00000002_0000.h5'  
#compare_pkl_to_h5(fn, './inp_gains/gains_', trans=False)
    

def generate_pkl_files(Gains, input_pkls):
     """ Take a Gains matrix and original 
     gain .pkl files and update pkl phases. 

     Parameters
     ----------
     Gains : np.ndarray[nfreq, nfeed]
          Complex gain matrix solved for by phase_solver_code
     input_pkls : string
          string giving path name of pkls (e.g. './inp_pkls/gain_slot')
      
     """

     for fpga_name in fpga_list:
          data_pkl = read_pkl(input_pkls + fpga_name + '.pkl')
          x=fpga_dict[fpga_name]
     
          feeds = np.where(slot_id==x)[0]
          feeds = feeds[ch_map]
          g = Gains[:, feeds]

          data_pkl = apply_gain(data_pkl, g, 16)     

          # Write pickle
          outfile = './outp_gains/gains_slot' + fpga_name + '.pkl'
          write_pkl(outfile, data_pkl)

          print "=================================="
          print "Wrote to %s" % outfile 
          print "=================================="

          print ""


ch_map = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]

if __name__=='__main__':

      infile = './inp_gains/gains_'
      #infile = './datah5/gains_'

      #g = h5py.File('Jun11Gains.hdf5', 'r')
      g = h5py.File('gains_jul8.hdf5', 'r')
      Gains = g['gains'][:]

      #Gains = construct_gain_mat('transj22_cygx', 64)
      #Gains = avg_channels(Gains)  

      for fpga_name in fpga_list:
          data_pkl = read_pkl(infile + fpga_name + '.pkl')
          x=fpga_dict[fpga_name]
     
          feeds = np.where(slot_id==x)[0]#[::-1]
          feeds = feeds[ch_map]
          g = Gains[:, feeds]

          data_pkl = apply_gain(data_pkl, g, 16)     

          # Write pickle
          outfile =  '/home/chime/gains_jun19/gains_' + fpga_name + '.pkl'
          outfile = './outp_gains/gains_' + fpga_name + '.pkl'
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


