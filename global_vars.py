# all the common libraries
import gc
from PIL import Image
import numpy as np
import json


# all the models are imported here
from models.clip import *
from models.blip import *
from models.align import *
from models.clip4clip import *
from models.xclip import *
from models.blip2 import *


# all the common variables are imported here

models_config = json.load(open('config.json', 'r'))
device = "cuda" if torch.cuda.is_available() else "cpu"
