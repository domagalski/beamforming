# Liam Connoaccur 4dm November 2015 
# Scipt to read in CHIME .vdif format files then 
# dedisperse and fold on a given pulsar

import os
import sys

import numpy as np
import h5py 
import glob

import ReadBeamform as rbf

import ch_pulsar_analysis as chp
#import ch_util.ephemeris as eph
import source_dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

assert len(sys.argv) == 6, "Expecting %s %s %s %s %s" % \
    ("fdir", "outfile", "pulsar", "file_number_start", "number_of_files")

accumulate = True
reamalgamate = True
reamalgamate_first = False # In case data already folded

timestream_save = False # Don't do anything pulsar related
dd_timestream = False # Produce dedispersed timestream, among other things
fold_full = True
coherent_dd = False
voltage_beam = False

make_highres_plot = False

plot_spec = True

npol = 2
nfreq = 1024
trb = 2000 # Factor to rebin data array by
f_start = np.int(sys.argv[4])
nfiles = np.int(sys.argv[5]) # Total number of files to us
f_step = 1 # step between files
n_save = 50 # Save down every n_save files

f_dir = '/drives/H/*/' + sys.argv[1]
outfile = '/drives/H/0/liamfolded/proc_data/' + sys.argv[2]

ngate = 64
psr = sys.argv[3]

if psr is not None:
     p0, dm, ra = source_dict.src_dict[psr]

     print "Using period %f and DM %f \n" % (p0, dm)

print "Accumulate : %r" % accumulate 
print "Reamalgamate : %r" % reamalgamate
print "Reamalg first : %r" % reamalgamate_first
print "Fold full : %r" % fold_full
print "Save every : %d files" % n_save

flist = glob.glob(f_dir + '/*dat')
flist.sort()

folded_spec_coh = []
icount_coh = []
dddis_full = []

arr = []
tt_tot = []

print "------------"
print "Frame range:"
print "------------"

RB = rbf.ReadBeamform()

def write_h5(outfile, arr, tt_tot):
     f = h5py.File(outfile, 'w')
     f.create_dataset('arr', data=arr)
     f.create_dataset('times', data=tt_tot)
     f.close()
     
     print "Wrote to: ", outfile

def amalgamate(outfile, taxis=-2):
     list_corr = glob.glob(outfile + '*.hdf5')
     list_corr = list_corr[:]
     list_corr.sort()

     Arr = []
     times_full = []

     for fnm in list_corr:
          ff = h5py.File(fnm, 'r')
          arr = ff['arr'][:]
          print arr.shape
          if len(arr.shape) < 3:
               continue

          Arr.append(arr)

     Arr = np.concatenate(Arr, axis=taxis)

     print "Writing to %s" % (outfile + 'full.hdf5')

     g = h5py.File(outfile + 'full.hdf5', 'w')
     g.create_dataset('arr', data=Arr)
     g.create_dataset('times', data=times_full)
     g.close()

     return Arr

if reamalgamate_first:
     if (dd_timestream or timestream_save):
          taxis=0
     else:
          taxis=-2

     Arr = amalgamate(outfile, taxis=taxis)

     if (plot_spec is True):
          #if fold_full is True:
          arrI = Arr[:, 0] + Arr[:, -1]
          chp.plot_spectra(arrI, outfile + '.png', dd_timestream=dd_timestream)

header_acc = []
data_acc = []

k = 0

for ii in range(f_start, f_start + nfiles, f_step):

     k += 1

     fnumber = "%07d" % (ii, )
     fname = f_dir + fnumber + '.dat'

     try:
          fname = glob.glob(fname)[0]
     except IndexError:
          print glob.glob(f_dir + '*')[0]

     if os.path.isfile(fname) is False:
          continue

     try:
          print "reading %s number %d" % (fname, k)

     except IndexError:
          print "Index issue"
          continue
          
     read_arrs = RB.read_file_dat(fname, voltage_beam=voltage_beam)

     if read_arrs == None:
          print "exiting"
          break

     header, data = read_arrs

     print "Unix time: %f %f" % (RB.get_times(header)[1][0], 0)

     if make_highres_plot is True:# and k==1:
          print "Making high res plot"
          arr_highres = RB.reorg_array(header, data)#, rbtime=625)
          nto = len(arr_highres)
          arr_highres = arr_highres[:nto//RB.nperpacket*RB.nperpacket]
#          arr_highres = arr_highres[:nto//25*25]

          arr_highres = np.abs(arr_highres.reshape(-1, RB.nperpacket, npol, nfreq))**2
#          arr_highres = np.abs(arr_highres.reshape(-1, 25, npol, nfreq))**2

          arr_highres = arr_highres.sum(1)
          np.save('dd' + np.str(k), arr_highres)
          continue

     times_o = RB.get_times(header, False)

#     print "RA: %d %f" % (times_o[0], eph.transit_RA(times_o[0]))

     # In case packets straddle multiple files, don't "correlate_and_fill"
     # until you have 3 files
     if accumulate == True:

          header_acc.append(header)
          data_acc.append(data)

          if (k % 3) == 0:
               
               header_acc = np.concatenate(header_acc, axis=0)
               data_acc = np.concatenate(data_acc, axis=0)

               if coherent_dd:
                    spec, count = RB.correlate_and_fill_cohdd(
                          data_acc, header_acc, p0, dm)

                    folded_spec_coh.append(spec)
                    icount_coh.append(count)
                    
               else:
                    if voltage_beam is True:
                         v, tt = RB.correlate_and_fill(data_acc, 
                                header_acc)
                    elif voltage_beam is False:
                         v, tt = RB.correlate_and_fill_incoherent(data_acc,
                                 header_acc)

                    arr.append(v)
                    tt_tot.append(tt)

                    
               header_acc = []
               data_acc = []

     else:
          v, tt = RB.correlate_and_fill(data, header)
          
#          print "Time %f and RA %f \n" % (tt[0, 0, 305], eph.transit_RA(tt[0, 0, 305]))

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
               chp.plot_spectra(psr_spec_full_coh[:, 0], 'blah.png')
               folded_spec_coh, icount_coh = [], []

          if timestream_save or fold_full or dd_timestream:
               arr = np.concatenate(arr)
               times = np.concatenate(tt_tot)

          if timestream_save is True:
               write_h5(outfile + np.str(ii) + '.hdf5', arr, times)

          # Fold whole array, include all polarizations          
          if fold_full is True and coherent_dd is False:

               print "Beginning fold \n"

               # Instance class with full array, but a dummy time vector
               PulsarPipeline = chp.PulsarPipeline(arr.transpose(), times[:, 0, 0])

               print "....... Folding data ....... \n"
               folded_spec, icount = PulsarPipeline.\
                            fold_real(dm, p0, times.transpose(), ngate=ngate, ntrebin=trb)
               
               psr_spec_full = folded_spec / icount
               psr_spec_full[np.isnan(psr_spec_full)] = 0.0

               del folded_spec, icount, PulsarPipeline

               write_h5(outfile + np.str(ii) + '.hdf5', psr_spec_full, times)#times[:, :, ::trb])
          
          if dd_timestream is True:
               print "....... Calculating dedispersed timestream ....... \n"

               # Instance class with full array, but a dummy time vector
               PulsarPipeline = chp.PulsarPipeline(arr.transpose(), times[:, 0, 0])

               print dm, np.diff(times[:, 0, 0])[:10]

               dedis_timestream, ddtimes = PulsarPipeline.\
                         dedispersed_timestream(dm, times.transpose(), 
                                onm=outfile + np.str(ii) + '.png')

               dedis_timestream -= np.mean(dedis_timestream, axis=-1, keepdims=True)

               dddis_full.append(dedis_timestream)

#               chp.plot_ddtimestream(dedis_timestream, ddtimes, './CRIMSON.png')
#               chp.plot_spectra(dedis_timestream, outfile + np.str(ii) + '.png', dd_timestream=True)

#               chp.plot_spectra(dedis_timestream, 'firenacht.png', dd_timestream=True)
              
               write_h5(outfile + '_ddts_' + np.str(ii) + '.hdf5', dedis_timestream, ddtimes)

          arr = []
          tt_tot = []

if dd_timestream is True:
     dddis_full = np.concatenate(dddis_full, axis=-1)
     print dddis_full.shape
     chp.plot_spectra(dddis_full, outfile + '.png', dd_timestream=True)
     write_h5(outfile + '_ddts_' + np.str(ii) + '.hdf5', dedis_timestream, ddtimes)

del arr, tt_tot

if reamalgamate is True and (fold_full is True or timestream_save is True):
     if timestream_save is True:
          Arr = amalgamate(outfile, taxis=0)
     else:
          Arr = amalgamate(outfile, taxis=-2)

     if plot_spec is True:
          if fold_full is True:
               arrI = Arr[:, 0] + Arr[:, -1] 
               chp.plot_spectra(arrI, outfile + '.png', dd_timestream=False)
