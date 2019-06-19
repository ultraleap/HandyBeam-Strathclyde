## notebook style -- full width
from IPython.core.display import display, HTML

display(HTML("applying full width style...<style>.container { width:100% !important; }</style>"))

## Imports
import importlib
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../handybeam_core_repo/")

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
