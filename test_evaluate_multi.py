# Basic Imports
from copy import copy
import datetime
from glob import glob
import json
import math
import multiprocessing
import os
import sys
from pathlib import Path
import random
import pickle
import urllib.request
import numpy as np
import pandas as pd
from multi_modal_constants import *
from tqdm.auto import tqdm


# PyTorch related imports....
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import time
from tempfile import TemporaryDirectory
import models_vit
import models_mae
from multi_modal_rFmodel import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from multi_modal_data_loader import *
from focal_loss_imp import *
import pytorch_warmup as warmup
from sklearn.model_selection import KFold


pre_proc_func = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])


multimodal_weights = torch.load("/tscc/nfs/home/vejoshi/oct_fundus_project/ret_found_exp/RETFound_MAE/experiments/fundus_oct/retFound/retFound_shape_224_dp_0.2_lr_0.0005_dense_run_0.pt",
                                map_location = "cuda")

multimodal_model = build_model(logits_flag = "True")
multimodal_model.load_state_dict(multimodal_weights, strict=False)
multimodal_model = multimodal_model.to("cuda")
multimodal_model.eval()


# load either single modality label set does not matter.....
with open(dataset_dir + "fundus_labels_binary_version_csv_algined.pickle", 'rb') as handle:
    label_dict = pickle.load(handle)


with open("/tscc/nfs/home/vejoshi/oct_fundus_project/oct_fundus_dataset/testing_multi_modal_file_pairs_maximal_repeat_match_csv_aligned.pickle", 'rb') as handle:
    testing_multi_modal_pair_list = pickle.load(handle)

activation_func = nn.Sigmoid()
test_files = []
test_cnt = {"healthy" : 0,  "glaucoma" : 0}
test_cnt_fundus  = {"healthy" : {"overall_cnt": 0, "OS" : 0, "OD" : 0},
                    "glaucoma" : {"overall_cnt": 0, "OS" : 0, "OD" : 0}}

for pid in testing_multi_modal_pair_list:
    # looping all pairs....
    for pr in testing_multi_modal_pair_list[pid]:

        # fundus id followed by oct id.....
        fd_img = dataset_dir + "fundus_images_csv_aligned/" + pr[0]
        eye_type = pr[0].split("_")[1]
        oc_img = dataset_dir + "oct_images_csv_aligned/" + pr[1]
        test_files.append((fd_img, oc_img))
        test_cnt[label_dict[pr[0]]] += 1
        test_cnt_fundus[label_dict[pr[0]]][eye_type]+=1


print("Total number of test instances : ",len(test_files))
print("Testing split   : Healthy : {} | Glaucoma : {}".format(test_cnt["healthy"] / len(test_files),
                                                               test_cnt["glaucoma"] / len(test_files)))


print("Eye split :")
print("Healthy  cnt : Left eye = {} | Right eye = {}".format(test_cnt_fundus["healthy"]["OS"], test_cnt_fundus["healthy"]["OD"]))
print("Glaucoma cnt : Left eye = {} | Right eye = {}".format(test_cnt_fundus["glaucoma"]["OS"], test_cnt_fundus["glaucoma"]["OD"]))
print("################")

testing_data = TestMultiModalGenerateDataset(pair_files = test_files,
                                             labels_dict = label_dict,
                                             transform = pre_proc_func)
test_loader = DataLoader(testing_data,
                         batch_size = 16,
                         shuffle=True,
                         num_workers=4)


print("Testing loop.....")
prob_values = []
pred_values = []
gt_values = []
cnt = 0

oct_df = pd.read_csv("/tscc/nfs/home/vejoshi/oct_fundus_project/oct_fundus_dataset/V2_oct_training_server_in_directory_train_val_test.csv")
oct_df["match_col"] = oct_df["ImageID"].apply(lambda x : str(x).split(".")[0])
fundus_df = pd.read_csv("/tscc/nfs/home/vejoshi/oct_fundus_project/oct_fundus_dataset/V2_fundus_training_server_in_directory_train_val_test.csv")


data_dict = {"oct_date" : [],
             "fundus_date" : [],
             "patientId" : [],
             "eye" : [],
             "fundus_name" : [],
             "oct_name" : [],
             "modelPrediction_prob" : [],
             "thresholded_op_0.5" : [],
             "groundTruth" : []}

for f_img, o_img, labels_index, fundus_path, oct_path in tqdm(test_loader, position=0, leave=True):
    with torch.set_grad_enabled(False):
        f_img = f_img.to("cuda")
        o_img = o_img.to("cuda")
        outputs = multimodal_model(f_img = f_img,
                               o_img = o_img)

        preds = torch.sigmoid(outputs)
        pred_label  = (preds > 0.5)*1
        prob_values.extend([i[0] for i in preds.detach().cpu().numpy()])
        pred_values.extend([i[0] for i in pred_label.detach().cpu().numpy()])
        gt_values.extend(labels_index)

    oct_date_list = []
    for i in oct_path:
        eyeId = i.split("/")[-1].split("_")[2]
        if oct_df[oct_df["match_col"] == eyeId]["ExamDate"].nunique() > 1:
            print(eyeId)

        oct_date_list.append(oct_df[oct_df["match_col"] == eyeId]["ExamDate"].iloc[0])

    fundus_date_list = []
    for i in fundus_path:
        diff_value = i.split("/")[-1].split("_")[2]
        # all variations are taken on the same date....
        entries = fundus_df[fundus_df["diff"] == int(diff_value[1:])]["date"].iloc[0]
        fundus_date_list.append(entries)



    # csv data points....
    data_dict["oct_date"].extend(oct_date_list)
    data_dict["fundus_date"].extend(fundus_date_list)
    data_dict["patientId"].extend([i.split("/")[-1].split("_")[0] for i in fundus_path])
    data_dict["eye"].extend([i.split("/")[-1].split("_")[1] for i in fundus_path])
    data_dict["fundus_name"].extend([i.split("/")[-1].split(".")[0] for i in fundus_path])
    data_dict["oct_name"].extend([i.split("/")[-1].split(".")[0] for i in oct_path])
    data_dict["modelPrediction_prob"].extend([i[0] for i in preds.detach().cpu().numpy()])
    data_dict["thresholded_op_0.5"].extend(["healthy" if i[0] == 1 else "glaucoma" for i in pred_label.detach().cpu().numpy()])
    data_dict["groundTruth"].extend(["healthy" if i == 1 else "glaucoma" for i in labels_index])

gt_values = np.array(gt_values)
pred_values = np.array(pred_values)
prob_values = np.array(prob_values)

# precision :
prec = precision_score(y_true = gt_values,
                              y_pred = pred_values,
                              average = "binary",
                              pos_label = 0,
                              zero_division = 0.0)
recall = recall_score(y_true = gt_values,
                              y_pred = pred_values,
                              average = "binary",
                              pos_label = 0,
                              zero_division = 0.0)

f1 = (2*(prec*recall))/(prec + recall)
auc = roc_auc_score(y_true  = gt_values, y_score = prob_values)

print("Multimodal scores : ")
print("Precision : {:.3f} | Recall : {:.3f} | F1 Score : {:.3f} | AUC Score : {:.3f}".format(prec, recall, f1, auc))
print("#####################################################################")

final_results = pd.DataFrame(data=data_dict)
final_results.to_csv("./multi_modality_retFound_fundus_oct_results_csv_aligned.csv")

