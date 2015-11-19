import os

import numpy as np
import h5py
import argparse
import datetime

import phase_solver_code as psolv 
import correct_pkl 
from ch_util import tools
import ch_util.ephemeris as eph
import misc_data_io as misc

parser = argparse.ArgumentParser(description="This script RFI-cleans, fringestops, and folds the pulsar data.")
parser.add_argument("fn", help="datafile")
parser.add_argument("src", help="Name of src to calibrate off")
parser.add_argument("-input_pkls", help="outfile prefix", default='./inp_gains/gains_slot')
parser.add_argument("-nfreq", help="Number of frequencies in acquisition", default=1024, type=int)
parser.add_argument("-nfeed", help="Number of feeds in acquisition", default=256, type=int)
parser.add_argument("-trans", help="Is the data already transposed", default=0, type=int)
parser.add_argument("-doall", help="do only the last bit", default=1, type=int)
parser.add_argument("-do_pkl_stuff", help="do only the last bit", default=0, type=int)

args = parser.parse_args()

name = args.src

# Find datetime string in input .h5 file
tstring = args.fn[args.fn.index('201') : args.fn.index('201') + 15]

outfile = './solutions/' + tstring + name + '.hdf5'

g = h5py.File(outfile, 'a')

if args.doall == 1:

    src_dict = {'CasA': eph.CasA, 'TauA': eph.TauA, 'CygA': eph.CygA}
    trans_dict = {0: False, 1: True}

    src = src_dict[name]

    nfreq = args.nfreq
    nfeed = args.nfeed

    corrinput_real, inpx, inpy, xcorrs, ycorrs, xfeeds, yfeeds  = psolv.gen_inp()

    gx = psolv.solve_untrans(args.fn, xcorrs, xfeeds, inpx, src, transposed=trans_dict[args.trans])
    g.create_dataset('gainsx', data=gx)

    print "================"
    print "== Starting y =="
    print "================"

    gy = psolv.solve_untrans(args.fn, ycorrs, yfeeds, inpy, src, transposed=trans_dict[args.trans])
    g.create_dataset('gainsy', data=gy)
    g.close()


g = h5py.File(outfile, 'r')

gx = g['gainsx'][:]
gy = g['gainsy'][:]

g.close()

outfile_tuple = './solutions/' + tstring + '_gainsoltup.dat'

os.system('python write_gain_tuple.py ' + outfile + ' ' + outfile_tuple)

print "Wrote down gains to: ", outfile

if args.do_pkl_stuff == 1:
    Gains = correct_pkl.construct_gain_mat(gx, gy, 64)

    os.system('scp "%s:%s" "%s"' % ("gamelan", "/home/chime/ch_acq/gains_slot*pkl", "./inp_gains/") )

    correct_pkl.do_it_all(Gains, args.input_pkls)

    os.system('scp -r "%s" chime@"%s:%s"' % ("/home/connor/code/beamforming/beamforming/outp_gains/gains_slot*pkl", "gamelan", "/home/chime/") )

