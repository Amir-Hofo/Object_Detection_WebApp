# import torch, torchvision, matplotlib, PIL
# for lib in [torch, torchvision, matplotlib, PIL]:
#   print(lib.__name__, '-->', lib.__version__)
# del matplotlib, PIL

import torch, torchvision
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision import models
import streamlit as st
import io
import warnings

warnings.filterwarnings("ignore")

# pip install pip-chill
# pip-chill >> requirements.txt