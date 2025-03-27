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

class classificationMultiModal(nn.Module):
    def __init__(self,
                 base_model = "vit_large_patch16",
                 logits_flag = False):
        
        super().__init__()            
        # instantiate model...(a lot of training control variables are specified in the constants.py file)
        self.model_fundus = models_vit.__dict__[base_model](img_size = input_shape,
                                                            num_classes =2,
                                                            drop_path_rate = dropout,
                                                            global_pool=False)

        self.model_oct =  models_vit.__dict__[base_model](img_size = input_shape,
                                                            num_classes =2,
                                                            drop_path_rate = dropout,
                                                            global_pool=False)
        
        self.layer_cat = nn.Linear(2048, 1)
        #self.layer_2 = nn.Linear(512,1)
        self.activ = nn.ReLU()
        self.activation_func = nn.Sigmoid()

        # loading model....
        checkpoint_fundus = torch.load(r_found_fun_og_weights, map_location = device)
        checkpoint_oct    = torch.load(r_found_oct_og_weights, map_location = device)

        msg_f = self.model_fundus.load_state_dict(checkpoint_fundus['model'], strict=False)
        msg_o = self.model_oct.load_state_dict(checkpoint_oct['model'], strict=False)
        #print("Loading status : ",msg)

        self.logits_flag = logits_flag




    def forward(self, f_img, o_img):
        
        x_fun = self.model_fundus.forward_features(f_img)
        x_oct = self.model_oct.forward_features(o_img)

        x_cat = torch.cat((x_fun, x_oct), dim = 1)
        x = self.layer_cat(x_cat)
        #x = self.activ(x)
        #x = self.layer_2(x)

        #if not self.logits_flag:
        #    x = self.activation_func(x)

        return x


# Using only pre-trained models for our experiments...
# Add the model as the experiments progress....
# Using heavy models for best classification accuracy....
def build_model(logits_flag = False):
    
    final_model = classificationMultiModal(logits_flag = logits_flag)
    

    return final_model

