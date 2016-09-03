import os
import sys

import time
import numpy as np
import glob
import datetime

import ch_util.ephemeris as eph
import print_timestamp as pt
from phase_solver_code import make_outfile_name

run_ph_solver = True

def reorder_dir(basedir):
    """ Find all directories in the 
    basedir and order them in order them 
    in creation time. Return most recent directory. 
    """
    dirlist = []
    dirtimes = []

    try:
        basedir_list = os.listdir(basedir)
    except OSError:
        print "A literal OSError"
        
    if basedir_list is None:
        print "%s is empty" % basedir
        return 

    for fn in basedir_list:

        if 'pathfinder_corr' in fn:
            dirlist.append(fn)
            dirtimes.append(os.path.getctime(basedir + fn))

    dirlist_sort, dirtimes_sort = zip(*sorted(zip(dirlist, dirtimes)))

    return dirlist_sort[-1], dirtimes_sort[-1]
    

def find_trans(fdir, src_search):
    """ Search fdir for a given source "src_search", 
    return the most recent transit file.
    """
    flist = glob.glob(fdir + '/*h5')
    flist.sort()

    for file in flist[::-1]:
        print "\nFile: %s \n" % file
        src_list = pt.print_timestamp(file)

        print src_list
        if src_list is None:
            print "%s is NoneType" % file
            continue 

        if src_search in src_list:
            trans_file = file

            print "Transit in %s" % trans_file

            return trans_file


def main():
    assert len(sys.argv) == 2, "src"

    basedir = '/mnt/gong/archive/'

    if run_ph_solver is True:
        while True:
            dir_recent = reorder_dir(basedir)[0]
  
            src_search = sys.argv[1]
            print dir_recent

            # Search the most recent directory for the source "src_search"
            trans_file = find_trans(basedir + dir_recent, src_search)

            if trans_file is None:
                print "Trans file doesn't exist in this directory, sleeping for 1hr"
                time.sleep(3600)
                continue

            tstring = make_outfile_name(trans_file)

            outfile = './solutions/' + tstring + src_search + '2.hdf5'
            
            # might be useful to just make sure gains are swapped
#            os.system('python solver_script.py %s %s -solve_gains 0 -gen_pkls 1' \
#                          % (trans_file, src_search))

            if os.path.exists(outfile):
                print "%s already exists, taking a nap. Will check again in 1 hour." % outfile
                time.sleep(3600)
                continue

            print "feed this", trans_file

            snm = time.time()

            os.system('python solver_script.py %s %s -solve_gains 1 -gen_pkls 1' \
                          % (trans_file, src_search))
            print "Going to sleep for a whole damn day"
            print eph.unix_to_datetime(time.time() - 8 * 3600)

            time.sleep(12 * 3600)


if __name__=='__main__':
    # Make sure there's a way to debug!
    import logging
    logging.basicConfig(level=logging.DEBUG, filename='./calib.log')

    try:
        main()
    except:        logging.exception("Oops:")
