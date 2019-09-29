## notebook style -- full width
from IPython.core.display import display, HTML
display(HTML("applying full width style...<style>.container { width:80% !important; }</style>"))



def spa(path=None):
    import sys

    if path is not None:
        sys.path.append(path)
    else:
        raise Exception('provide a path to append to sys.path.append')

# this estabilishes the path to the development version of handybeam:

import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
print(f'current_dir is "{current_dir}"')
handybeam_dir="..\..\handybeam-core-code"
handybeam_dir2=os.path.normpath(os.path.join(current_dir, handybeam_dir))
print(f'expecting handybeam module at "{handybeam_dir2}"')
sys.path.append(handybeam_dir2)

## Imports
import importlib
import sys
import numpy as np
import matplotlib.pyplot as plt



import handybeam
import handybeam.world
import handybeam.tx_array_library
import handybeam.tx_array
import handybeam.visualise
import handybeam.samplers.rectilinear_sampler
import handybeam.samplers.clist_sampler
import handybeam.samplers.arcYZ
from handybeam.solver import Solver
# matplotlib.rcParams['figure.figsize'] = [15, 10]
from handybeam.misc import HandyDict
import scipy.signal.windows
import strathclyde

