# Script to RFI-clean, fringestop, then solve 
# gains off of a point-source transit. 
#
# Example usage:
# python solver_script.py /mnt/gong/archive/20160604T115702Z_pathfinder_corr/00154100_0000.h5 CygA -solve_gains 1 -gen_pkls 1
# which would isolate the Cygnus A transit in file 00154100_0000.h5
# and fringestop, solve for gains, write those gains to an hdf5
# file, and then generate corresponding *pkl files

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

parser = argparse.ArgumentParser(description="This script RFI-cleans, fringestops, solve gains")

parser.add_argument("fn", help="datafile")
parser.add_argument("src", help="Name of src to calibrate off")
parser.add_argument("-input_pkls", help="outfile prefix", default='./inp_gains/gains_slot')
parser.add_argument("-nfreq", help="Number of frequencies in acquisition", default=1024, type=int)
parser.add_argument("-nfeed", help="Number of feeds in acquisition", default=256, type=int)
parser.add_argument("-trans", help="Is the data already transposed", default=1, type=int)
parser.add_argument("-solve_gains", help="Get phase soltuion", default=1, type=int)
parser.add_argument("-gen_pkls", help=
          "do only the last bit, assuming gain sol is already calculated", default=0, type=int)

args = parser.parse_args()

name = args.src

# Find datetime string in input .h5 file

#tstring = args.fn[args.fn.index('201') : args.fn.index('201') + 15]

tstring = psolv.make_outfile_name(args.fn)

outfile = './solutions/' + tstring + name + 'toonie.hdf5'

g = h5py.File(outfile, 'a')

if args.solve_gains == 1:

    src_dict = {'CasA': eph.CasA, 'TauA': eph.TauA, 'CygA': eph.CygA}
    trans_dict = {0: False, 1: True}

    src = src_dict[name]

    nfreq = args.nfreq
    nfeed = args.nfeed

    corrinput_real, inpx, inpy, xcorrs, ycorrs, xfeeds, yfeeds  = psolv.gen_inp()

    # Get gain solution from only x-polarization correlations
    gx = psolv.solve_ps_transit(
        args.fn, xcorrs, xfeeds, inpx, src, transposed=trans_dict[args.trans])

    g.create_dataset('gainsx', data=gx)

    print "============================"
    print "==       Starting y       =="
    print "============================"

    # Get gain solution from only y-polarization correlations
    gy = psolv.solve_ps_transit(
        args.fn, ycorrs, yfeeds, inpy, src, transposed=trans_dict[args.trans])

    g.create_dataset('gainsy', data=gy)
    g.close()


g = h5py.File(outfile, 'r')

gx = g['gainsx'][:]
gy = g['gainsy'][:]

g.close()

outfile_tuple = './solutions/' + tstring + '_gainsoltup.dat'

os.system('python write_gain_tuple.py ' + outfile + ' ' + outfile_tuple)

print "Wrote down gains to: ", outfile

if args.gen_pkls == 1:
    Gains = correct_pkl.construct_gain_mat(gx, gy, 64)

    os.system('scp "%s:%s" "%s"' % ("chime@glock", 
           "/home/chime/ch_acq/gains_slot*pkl", "./inp_gains/") )

    correct_pkl.generate_pkl_files(Gains, args.input_pkls)

    print "Sending to chime@glock:~/ch_acq/"
    os.system('scp /home/connor/code/beamforming/beamforming/outp_gains/gains_slot*.pkl chime@glock:~/ch_acq/')
#    os.system('scp -r "%s" chime@"%s:%s"' % ("\
#    /home/connor/code/beamforming/beamforming/outp_gains/gains_slot*pkl", \
#    "glock", "/home/chime/") )

