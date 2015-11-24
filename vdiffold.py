# Liam Connor 4 November 2015 
# Scipt to read in CHIME .vdif format files then 
# dedisperse and fold on a given pulsar

import os
import sys

import numpy as np
import h5py 
import glob

import ReadBeamform2 as rbf
import ch_pulsar_analysis2 as chp
import ch_util.ephemeris as eph
import source_dict

accumulate = True
reamalgamate = True
dd_timestream = False
fold_full = True

npol = 2
nfreq = 1024
trb = 2000 # Factor to rebin data array by
f_start = 4250
nfiles = 2000 #25 # Total number of files to us
f_step = 1 # step between files
n_save = 30 # Save down every n_save files

f_dir = '/drives/E/*/' + sys.argv[1]
outfile = './proc_data/' + sys.argv[2]

ngate = 256
psr = sys.argv[3]

p0, dm = source_dict.src_dict[psr]

print "Using period %f and DM %f" % (p0, dm)

flist = glob.glob(f_dir + '/*dat')
flist.sort()

arr = []
tt_tot = []

print "------------"
print "Frame range:"
print "------------"

RB = rbf.ReadBeamform(pmax=1e7)

def write_h5(outfile, arr, tt_tot):
     f = h5py.File(outfile, 'w')
     f.create_dataset('arr', data=arr)
     f.create_dataset('times', data=tt_tot)
     f.close()
     
     print "Wrote to: ", outfile

def amalgamate(outfile):
     list_corr = glob.glob(outfile + '*.hdf5')
     list_corr = list_corr[:]
     list_corr.sort()

     Arr = []
     TT = []

     for fnm in list_corr:
          ff = h5py.File(fnm, 'r')

          arr = ff['arr'][:]
#          tt = ff['times'][:]

          Arr.append(arr)
#          TT.append(tt)
     
     Arr = np.concatenate(Arr, axis=-2)
#     TT = np.concatenate(TT, axis=-2)

     print "Writing to %s" % (outfile + 'full.hdf5')

     g = h5py.File(outfile + 'full.hdf5', 'w')
     g.create_dataset('arr', data=Arr)
#     g.create_dataset('times', data=TT)
     g.close()

if reamalgamate is True and fold_full is True:
     amalgamate(outfile)

header_acc = []
data_acc = []

k = 0

for ii in range(f_start, f_start + nfiles, f_step):
     
     k += 1

     fnumber = "%07d" % (ii, )
     fname = f_dir + fnumber + '.dat'
     fname = glob.glob(fname)[0]

     try:
          print "reading %s number %d" % (fname, k)

     except IndexError:
          print "Index issue"
          continue
          
     if os.path.isfile(fname) is False:
          continue

     read_arrs = RB.read_file_dat(fname)

     if read_arrs == None:
          print "exiting"
          break

     header, data = read_arrs

     # In case packets straddle multiple files, don't "correlate_and_fill"
     # until you have 3 files
     if accumulate == True:

          header_acc.append(header)
          data_acc.append(data)

          if (k % 3) == 0:

               header_acc = np.concatenate(header_acc, axis=0)
               data_acc = np.concatenate(data_acc, axis=0)

               v, tt = RB.correlate_and_fill(data_acc, header_acc)

               print "Time %f and RA %f \n" % (tt[0, 0, 0], eph.transit_RA(tt[0, 0, 0]))

               arr.append(v)
               tt_tot.append(tt)

               header_acc = []
               data_acc = []

     else:
          v, tt = RB.correlate_and_fill(data, header)

          print "Time %f and RA %f \n" % (tt[0, 0, 0], eph.transit_RA(tt[0, 0, 0]))

          arr.append(v)
          tt_tot.append(tt)   
          
     del header, data

     if (ii == range(f_start, f_start + nfiles, f_step)[-1]) or ((k % n_save) == 0):

          print "Beginning fold \n"

          arr = np.concatenate(arr).transpose()
          times = np.concatenate(tt_tot).transpose()

          # Instance class with full array, but a dummy time vector
          PulsarPipeline = chp.PulsarPipeline(arr, times[0, 0])

          # Fold whole array, include all polarizations
          
          if fold_full is True:
               print "....... Folding data ....... \n"
               folded_spec, icount = PulsarPipeline.\
                            fold_real(dm, p0, times, ngate=ngate, ntrebin=trb)

               del arr
          
               psr_spec_full = folded_spec / icount
               psr_spec_full[np.isnan(psr_spec_full)] = 0.0

               del folded_spec, icount

               write_h5(outfile + np.str(ii) + '.hdf5', psr_spec_full, times)#times[:, :, ::trb])
          
          if dd_timestream is True:
               print "....... Calculating dedispersed timestream ....... \n"

               dedis_timestream, ddtimes = PulsarPipeline.dedispersed_timestream(dm, times)
               write_h5(outfile + np.str(ii) + 'ddts.hdf5', psr_spec_full, ddtimes)

          arr = []
          tt_tot = []

del arr, tt_tot



def dedisperse_ts():
     freq = np.linspace(800, 400, 1024)
     
     bins_per_sec = nbins / tt_tot

     for fi in range(1024):
          tau = (4.148808e3 * dm * (freq[fi]**(-2) - (400.0)**(-2)))
          tau_int = np.int(tau * bins_per_sec)
          arr[fi] = np.roll(arrf[i], tau_int, axis=-1)

def plot_spectra(arr):
     arr[mask] = 0.0
     arr[np.isnan(arr)] = 0.0

     arrf = real(arr - np.mean(arr, axis=-1, 
                               keepdims=True)).mean(-2)

     arrt = real(arr - np.mean(arr, axis=-1, 
                               keepdims=True)).mean(0)

     subplot(121)

     imshow(arrt, interpolation='nearest',
            aspect='auto', cmap='RdBu', vmax=0.03, vmin=-0.01)
     ylabel('times')
     xlabel('phase')

     subplot(122)
     imshow(arrf, interpolation='nearest',
            aspect='auto', cmap='RdBu', vmax=0.03, vmin=-0.01)
     ylabel('freq')
     xlabel('phase')
