## notebook style -- full width
from IPython.core.display import display, HTML

display(HTML("applying full width style...<style>.container { width:100% !important; }</style>"))

def spa(path=None):
    import sys
    if path is not None:
        sys.path.append(path)
    else:
        raise Exception('provide a path to append to sys.path.append')

spa("../handybeam_core_repo/")

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
from handybeam.solver import Solver
# matplotlib.rcParams['figure.figsize'] = [15, 10]
from handybeam.misc import HandyDict
import scipy.signal.windows
import strathclyde

