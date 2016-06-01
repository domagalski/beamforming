import sys

import numpy as np
import h5py

from ch_util import ephemeris as eph

def print_timestamp(fn):
    src_dict = {'CasA': eph.CasA, 'TauA': eph.TauA, 'CygA': eph.CygA, 'VirA': eph.VirA,
            '1929': 292.0, '0329': 54.0}

    try:
        f = h5py.File(fn, 'r')
    except IOError:
        print "File couldn't be opened"
        return 

    times = f['index_map']['time'].value['ctime'][:]

    print "-------------------------------------"
    print "start time %s in PT\n" % (eph.unix_to_datetime(times[0]))

    print "RA range %f : %f" % (np.round(eph.transit_RA(times[0]), 1), 
                                np.round(eph.transit_RA(times[-1]),1))
    print "-------------------------------------"

    srcz = []
    for src in src_dict:
        if eph.transit_times(src_dict[src], times[0])[0] < times[-1]:
            print "%s is in this file" % src
            srcz.append(src)

    return srcz

if __name__ == '__main__':
    fn = sys.argv[1]

    print_timestamp(fn)
