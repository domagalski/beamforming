import numpy as np
import h5py
import argparse

import ch_pulsar_analysis as chp

parser = argparse.ArgumentParser(description="")
parser.add_argument("data", help="pulsar data to fold")
parser.add_argument("p0", help="pulsar period", type=np.float32)
parser.add_argument("dm", help="dispersion measure", type=np.float32)
parser.add_argument("--ngate", help="number of phase bins", default=32, type=int)
parser.add_argument("--trb", help="number of phase bins", default=1000, type=int)
parser.add_argument("--outfile", help="if used, will save down hdf5 file to outfile", default=None)
args = parser.parse_args()

f = h5py.File(args.data, 'r')

# Read in two datasets and transpose to look like andata
arr = (f['arr'][:]).transpose((2, 1, 0))
times = (f['times'][:]).transpose((2, 1, 0))

nt = arr.shape[-1]

p0 = args.p0
dm = args.dm

PP0 = chp.PulsarPipeline(arr[:, 0, np.newaxis], times[0, 0])
PP1 = chp.PulsarPipeline(arr[:, 1, np.newaxis], times[0, 0])

# Fold the two polarization separately
spec0, icount0 = PP0.fold2(dm, p0, times[:, 0], ngate=args.ngate, ntrebin=args.trb)
spec1, icount1 = PP1.fold2(dm, p0, times[:, 1], ngate=args.ngate, ntrebin=args.trb)

psr_spec0 = spec0 / icount0
psr_spec1 = spec1 / icount1

# Zero out the inevitable NaNs
psr_spec0[np.isnan(psr_spec0)] = 0.0
psr_spec1[np.isnan(psr_spec1)] = 0.0

psr_spec = psr_spec0 + psr_spec1
times_psr = times[:, 0, :nt // args.trb * args.trb]

# Now take middle of timestamp for rebinning
times_psr = times_psr.reshape(1024, -1, args.trb)[..., args.trb/2]

print "========================================"
print "Finished folding and summing frequencies"
print "========================================"

if args.outfile != None:
    g = h5py.File(args.outfile, 'w')
    g.create_dataset('psr_spec', data=psr_spec)
    g.create_dataset
    g.close()
    print "Wrote to: ", args.outfile


