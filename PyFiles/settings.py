import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from pathlib import Path
from utils import *

TRAIN_PATH = "gs://leap-persistent/jbusecke/data/climatebench/train_val/"
TEST_PATH = "gs://leap-persistent/jbusecke/data/climatebench/test/"

CWD_PYFILE = os.getcwd()
PYFILES_PATH = Path(CWD_PYFILE)
MODEL_NN_SAVE_PATH = PYFILES_PATH.parent / "Notebooks" / "Models" / "Neural_Network"

CHLIST = ["#7E2954", "#bbbbbb", "#2E2585", "#5DA899", "#DCCD7D", "#C26A77"]
