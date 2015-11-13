import os

import numpy as np
import time

import phase_solver_code as pc

time_yesterday = time.time() - 24 * 3600.0

src = CasA

# Find previous day's transit file for CasA
fn = pc.find_transit(time_yesterday)

os.system('python ph_solver_script.py %s %s' % (fn, src))
