import numpy as np
import h5py
import argparse

import phase_solver_code as psolv 
import correct_pkl 

parser = argparse.ArgumentParser(description="This script RFI-cleans, fringestops, and folds the pulsar data.")
parser.add_argument("fn", help="datafile")
parser.add_argument("src", help="Name of src to calibrate off")
parser.add_argument("outfile", help="outfile prefix")
parser.add_argument("-input_pkls", help="outfile prefix", default='./inp_gains/gains_')
parser.add_argument("-nfreq", help="Number of frequencies in acquisition", default=1024, type=int)
parser.add_argument("-nfeed", help="Number of feeds in acquisition", default=256, type=int)

args = parser.parse_args()

name = args.src
src = eph.CygA # Need to figure out how to turn str into variable

nfreq = args.nfreq
nfeed = arg.nfeed

# Assumes a standard layout
xfeeds = range(nfeed/4) + range(2 * nfeed/4, 3 * nfeed/4)
yfeeds = range(nfeed/4, 2 * nfeed/4) + range(3 * nfeed/4, 4 * nfeed/4)

xcorrs = []
ycorrs = []

for ii in range(nfeed/2):
     for jj in range(ii, nfeed/2):
          xcorrs.append(misc.feed_map(xfeeds[ii], xfeeds[jj], nfeed))
          ycorrs.append(misc.feed_map(yfeeds[ii], yfeeds[jj], nfeed))


corrinputs = tools.get_correlator_inputs(\
                datetime.datetime(2015, 6, 1, 0, 0, 0), correlator='K7BP16-0004')

# Now rearrange to match the correlation indices in the h5 files
corrinput_real = psolv.rearrange_list(corrinputs, nfeeds=256)

 
"""
This was used before I just hardcoded the list in. 

R = andata.Reader(fn0)
R.prod_sel = 0
R.freq_sel = 305
R.time_sel = [0, 2]
and_obj = R.read()
corrinput_real = rearrange_inp(and_obj, corrinputs, nfeeds=256)
"""

inpx = []
inpy = []

for i in range(nfeed/2):
    inpx.append(corrinput_real[xfeeds[i]])
    inpy.append(corrinput_real[yfeeds[i]])

gx = psolv.solve_untrans(fn, xcorrs, inpx, src)
gy = psol.solve_untrans(fn, ycorrs, inpy, src)

g = h5py.File(outfile, 'w')
g.create_dataset('gainsx', data=gx)
g.create_dataset('gainsy', data=gy)
g.close()

print "Wrote down gains to: ", oufile

Gains = correct_pkl.construct_gain_mat(gx, gy, 64)

correct_pkl.do_it_all(Gains, args.input_pkls)






