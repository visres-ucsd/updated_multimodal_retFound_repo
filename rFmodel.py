# Basic Imports
from copy import copy
import datetime
from glob import glob
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import urllib.request
import numpy as np
from constants import *

# PyTorch related imports....
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision

# retfound specific import.....
import models_vit


# Impoving code reproducability...
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)
cudnn.benchmark = True

# setting up gpu details
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

print("PyTorch version : {} | Device type : {}".format(torch.__version__, device))

class classificationModel(nn.Module):
    def __init__(self,
                 base_model = "vit_large_patch16"):
        
        super().__init__()            
        # instantiate model...(a lot of training control variables are specified in the constants.py file)
        self.model = models_vit.__dict__[base_model](img_size = input_shape,
                                                num_classes =2,
                                                drop_path_rate = dropout,
                                                global_pool=True)
        
        self.layer_1 = nn.Linear(1024, 1)
        self.activation_func = nn.Sigmoid()

        # loading model....
        checkpoint = torch.load(ret_found_ext_og_weights, map_location = device)
        msg = self.model.load_state_dict(checkpoint['model'], strict=False)
        #print("Loading status : ",msg)




    def forward(self, x):
        
        x = self.model.forward_features(x)
        x = self.layer_1(x)
        x = self.activation_func(x)

        return x


# Using only pre-trained models for our experiments...
# Add the model as the experiments progress....
# Using heavy models for best classification accuracy....
def build_model():
    
    final_model = classificationModel()
    

    return final_model

