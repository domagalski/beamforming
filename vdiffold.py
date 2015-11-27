# Liam Connor 4 November 2015 
# Scipt to read in CHIME .vdif format files then 
# dedisperse and fold on a given pulsar

import os
import sys

import numpy as np
import h5py 
import glob

import ReadBeamform as rbf
import ch_pulsar_analysis2 as chp
import ch_util.ephemeris as eph
import source_dict

accumulate = True
reamalgamate = True
dd_timestream = False
fold_full = False
coherent_dd = False
plot_spec = True

npol = 2
nfreq = 1024
trb = 1500 # Factor to rebin data array by
f_start = 4950
nfiles = 40 #25 # Total number of files to us
f_step = 1 # step between files
n_save = 5 # Save down every n_save files

f_dir = '/drives/E/*/' + sys.argv[1]
outfile = './proc_data/' + sys.argv[2]

ngate = 350
psr = sys.argv[3]

p0, dm = source_dict.src_dict[psr]

print "Using period %f and DM %f" % (p0, dm)

flist = glob.glob(f_dir + '/*dat')
flist.sort()

folded_spec_coh = []
icount_coh = []

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
          Arr.append(arr)
     
     Arr = np.concatenate(Arr, axis=-2)

     print "Writing to %s" % (outfile + 'full.hdf5')

     g = h5py.File(outfile + 'full.hdf5', 'w')
     g.create_dataset('arr', data=Arr)
     g.close()

     return Arr

header_acc = []
data_acc = []

k = 0

for ii in range(f_start, f_start + nfiles, f_step):
     
     k += 1

     fnumber = "%07d" % (ii, )
     fname = f_dir + fnumber + '.dat'

     fname = glob.glob(fname)[0]

     if os.path.isfile(fname) is False:
          continue

     try:
          print "reading %s number %d" % (fname, k)

     except IndexError:
          print "Index issue"
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

               if coherent_dd:
                    spec, count = RB.correlate_and_fill_cohdd(data_acc, header_acc, p0, dm)
                    folded_spec_coh.append(spec)
                    icount_coh.append(count)
                    
               else:
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

          if coherent_dd is True:
               folded_spec_coh = np.concatenate(folded_spec_coh)
               icount_coh = np.concatenate(icount_coh)

               psr_spec_full_coh = folded_spec_coh / icount_coh
               psr_spec_full_coh[np.isnan(psr_spec_full_coh)] = 0.0

               write_h5(outfile + np.str(ii) + '.hdf5', psr_spec_full_coh, [])

               folded_spec_coh, icount_coh = [], []

          # Fold whole array, include all polarizations          
          if fold_full is True and coherent_dd is False:

               print "Beginning fold \n"

               arr = np.concatenate(arr).transpose()
               times = np.concatenate(tt_tot).transpose()

               # Instance class with full array, but a dummy time vector
               PulsarPipeline = chp.PulsarPipeline(arr, times[0, 0])

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

               dedis_timestream, ddtimes = PulsarPipeline.\
                         dedispersed_timestream(dm, times)
               print 'hear'
               chp.plot_spectra(dedis_timestream, outfile + '.png', dd_timestream=True)
              
               write_h5(outfile + np.str(ii) + 'ddts.hdf5', dedis_timestream, ddtimes)

          arr = []
          tt_tot = []

del arr, tt_tot

if reamalgamate is True and fold_full is True:
     Arr = amalgamate(outfile)

     if (plot_spec is True):
          if fold_full is True:
               arrI = Arr[:, 0] + Arr[:, -1] 
               chp.plot_spectra(arrI, outfile + '.png', dd_timestream=True)
