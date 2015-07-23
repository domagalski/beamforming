import numpy as np
import h5py
import argparse

import ch_pulsar_analysis as chp

parser = argparse.ArgumentParser(description="")
parser.add_argument("data", help="pulsar data to fold")
parser.add_argument("p0", help="pulsar period", type=np.float32)
parser.add_argument("dm", help="dispersion measure", type=np.float32)
args = parser.parse_args()

f = h5py.File(args.data, 'r')

# Read in two datasets and transpose to look like andata
arr = (f['arr'][:]).transpose((2, 1, 0))
times = (f['times'][:]).transpose((2, 1, 0))

p0 = args.p0
dm = args.dm

PP = chp.PulsarPipeline(arr, times[0, 0])

# Fold the two polarization separately
spec0, icount0 = PP.fold2(dm, p0, times[:, 0])
spec1, icount1 = PP.fold2(dm, p0, times[:, 1])

psr_spec0 = spec0 / icount0
psr_spec1 = spec1 / icount1

# Zero out the inevitable NaNs
psr_spec0[np.isnan(psr_spec0)] = 0.0
psr_spec1[np.isnan(psr_spec1)] = 0.0

psr_spec = psr_spec0[:, 0] + psr_spec1[:, 0]

print psr_spec.shape


