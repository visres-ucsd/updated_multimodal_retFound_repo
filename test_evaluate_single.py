# Basic Imports........
import os
import numpy as np
import pandas as pd
import pickle
from tqdm.auto import tqdm
import sys
sys.path.append("/tscc/nfs/home/vejoshi/oct_fundus_project/fundus_oct_multi_modal_classifier/")
from constants import *


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
from rFmodel import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from data_loader import *
from focal_loss_imp import *
import pytorch_warmup as warmup
from sklearn.model_selection import KFold
import pandas as pd



# Fundus Evaluation
fundus_model_weights = torch.load("/tscc/nfs/home/vejoshi/oct_fundus_project/ret_found_exp/RETFound_MAE/experiments/fundus/retFoundretFound_shape_224_dp_0.1_lr_0.0005_csv_aligned_run_0.pt",
                                  map_location = "cuda")
fundus_model = build_model()
fundus_model.load_state_dict(fundus_model_weights, strict=False)
fundus_model = fundus_model.to("cuda")
fundus_model.eval()


oct_model_weights = torch.load("/tscc/nfs/home/vejoshi/oct_fundus_project/ret_found_exp/RETFound_MAE/experiments/oct/retFoundretFound_shape_224_dp_0.1_lr_0.0005_csv_aligned_run_0.pt",
                               map_location = "cuda")
oct_model = build_model()
oct_model.load_state_dict(oct_model_weights, strict=False)
oct_model = oct_model.to("cuda")
oct_model.eval()


pre_proc_func = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

# test patient ids loaded
with open(test_ids_paths, 'rb') as handle:
    test_ids_list = pickle.load(handle)

# fundus loading image paths & labels....
fundus_df = pd.read_csv("/tscc/nfs/home/vejoshi/oct_fundus_project/oct_fundus_dataset/V2_fundus_training_server_in_directory_train_val_test.csv")
fundus_img_path = dataset_dir + "fundus_images_csv_aligned/"
label_dict_fundus = {}
label_path_fundus = dataset_dir + "fundus_labels_binary_version_csv_algined.pickle"
with open(label_path_fundus, 'rb') as handle:
    label_dict_fundus = pickle.load(handle)


print("Fundus Modality Testing performance..............................")
print("Total number of patient ids       : ",len(label_dict_fundus))
print("Number of loaded test patient ids : ",len(test_ids_list))
test_cnt_fundus  = {"healthy"  : {"overall_cnt": 0, "OS" : 0, "OD" : 0},  
                    "glaucoma" : {"overall_cnt": 0, "OS" : 0, "OD" : 0}}

# Extracting the test patient ids only....
test_files_fundus  = []
for i in label_dict_fundus.keys():
    pid = i.split("_")[0]
    eye_type = i.split("_")[1]

    if pid in test_ids_list:
        test_files_fundus.append(os.path.join(fundus_img_path,i))
        test_cnt_fundus[label_dict_fundus[i]]["overall_cnt"] += 1
        test_cnt_fundus[label_dict_fundus[i]][eye_type] += 1


# Displaying % split....
print("Total number of test instances : ",len(test_files_fundus))
print("Testing split   : Healthy : {} | Glaucoma : {}".format(test_cnt_fundus["healthy"]["overall_cnt"] / len(test_files_fundus),
                                                               test_cnt_fundus["glaucoma"]["overall_cnt"] / len(test_files_fundus)))


print("Eye split :")
print("Healthy  cnt : Left eye = {} | Right eye = {}".format(test_cnt_fundus["healthy"]["OS"], test_cnt_fundus["healthy"]["OD"]))
print("Glaucoma cnt : Left eye = {} | Right eye = {}".format(test_cnt_fundus["glaucoma"]["OS"], test_cnt_fundus["glaucoma"]["OD"]))
print("################")

testing_data_fundus = TestGenerateDataset(image_files = test_files_fundus,
                                    labels_dict = label_dict_fundus,
                                    img_res = 224,
                                    transform = pre_proc_func)
test_dataloader_fundus = DataLoader(testing_data_fundus,
                                  batch_size = batch_size,
                                  shuffle=True,
                                  num_workers=4)

print("Testing loop.....")
prob_values = []
pred_values = []
gt_values = []
cnt = 0
pandas_fundus_dict = {"patientId" : [],
                      "date" : [],
                      "eye" : [],
                      "imageName" : [],
                      "modelPrediction_prob" : [],
                      "thresholded_op_0.5" : [],
                      "groundTruth" : []}

for inputs, labels_index, file_path in tqdm(test_dataloader_fundus, position=0, leave=True):
    inputs = inputs.to("cuda")
    preds  = fundus_model(inputs)
    pred_label  = (preds > 0.5)*1
    prob_values.extend([i[0] for i in preds.detach().cpu().numpy()])
    pred_values.extend([i[0] for i in pred_label.detach().cpu().numpy()])
    gt_values.extend(labels_index)
    
    # Extracting the dates via the diff value..... (looping the entire batch of test samples) 
    dates_list = []
    for fp in file_path:
        diff_value = fp.split("/")[-1].split("_")[2]
        # all variations are taken on the same date....
        entries = fundus_df[fundus_df["diff"] == int(diff_value[1:])]["date"].iloc[0]
        dates_list.append(entries)
        

    # csv data points.....
    pandas_fundus_dict["date"].extend(dates_list)
    pandas_fundus_dict["patientId"].extend([i.split("/")[-1].split("_")[0] for i in file_path])
    pandas_fundus_dict["eye"].extend([i.split("/")[-1].split("_")[1] for i in file_path])
    pandas_fundus_dict["imageName"].extend([i.split("/")[-1].split(".")[0] for i in file_path])
    pandas_fundus_dict["modelPrediction_prob"].extend([i[0] for i in preds.detach().cpu().numpy()])
    pandas_fundus_dict["thresholded_op_0.5"].extend(["healthy" if i[0] == 1 else "glaucoma" for i in pred_label.detach().cpu().numpy()])
    pandas_fundus_dict["groundTruth"].extend(["healthy" if i == 1 else "glaucoma" for i in labels_index])


fundus_results = pd.DataFrame(data=pandas_fundus_dict)
fundus_results.to_csv("./single_modality_retFound_fundus_results_csv_aligned.csv")

# Calculating metrics....
gt_values = np.array(gt_values)
pred_values = np.array(pred_values)
prob_values = np.array(prob_values)

# Measures taken wrt to the glaucoma class....
# precision :
fundus_prec = precision_score(y_true = gt_values,
                              y_pred = pred_values,
                              average = "binary",
                              pos_label = 0,
                              zero_division = 0.0)
fundus_recall = recall_score(y_true = gt_values,
                              y_pred = pred_values,
                              average = "binary",
                              pos_label = 0,
                              zero_division = 0.0)

fundus_f1 = (2*(fundus_prec*fundus_recall))/(fundus_prec + fundus_recall)
fundus_auc = roc_auc_score(y_true  = gt_values, y_score = prob_values)

print("Fundus scores : ")
print("Precision : {:.3f} | Recall : {:.3f} | F1 Score : {:.3f} | AUC Score : {:.3f}".format(fundus_prec, fundus_recall, fundus_f1, fundus_auc))
print("#####################################################################")



print("\n\n\n")
print("OCT Modality Testing performance....................")


oct_df = pd.read_csv("/tscc/nfs/home/vejoshi/oct_fundus_project/oct_fundus_dataset/V2_oct_training_server_in_directory_train_val_test.csv")
oct_df["match_col"] = oct_df["ImageID"].apply(lambda x : str(x).split(".")[0])

# oct loading image paths & labels....
oct_img_path = dataset_dir + "oct_images_csv_aligned/"
label_dict_oct = {}
label_path_oct = dataset_dir + "oct_labels_binary_version_csv_algined.pickle"
with open(label_path_oct, 'rb') as handle:
    label_dict_oct = pickle.load(handle)


print("OCT Modality Testing performance..............................")
print("Total number of patient ids       : ",len(label_dict_oct))
print("Number of loaded test patient ids : ",len(test_ids_list))
test_cnt_oct  = {"healthy" : {"overall_cnt": 0, "L" : 0, "R" : 0},
                    "glaucoma" : {"overall_cnt": 0, "L" : 0, "R" : 0}}
test_files_oct  = []
pandas_oct_dict = {"date" : [],
                      "patientId" : [],
                      "eye" : [],
                      "imageName" : [],
                      "modelPrediction_prob" : [],
                      "thresholded_op_0.5" : [],
                      "groundTruth" : []}

for i in label_dict_oct.keys():
    pid = i.split("_")[0]
    mat = i.split("_")[2]
    
    if pid in test_ids_list:
        eye_name = oct_df[oct_df["match_col"] == mat]["Eye"].iloc[0]
        test_files_oct.append(os.path.join(oct_img_path,i))
        test_cnt_oct[label_dict_oct[i]]["overall_cnt"] += 1
        test_cnt_oct[label_dict_oct[i]][eye_name] += 1

print("Total number of test instances : ",len(test_files_oct))
print("Testing split   : Healthy : {} | Glaucoma : {}".format(test_cnt_oct["healthy"]["overall_cnt"] / len(test_files_oct),
                                                               test_cnt_oct["glaucoma"]["overall_cnt"] / len(test_files_oct)))


print("Eye split :")
print("Healthy  cnt : Left eye = {} | Right eye = {}".format(test_cnt_oct["healthy"]["L"], test_cnt_oct["healthy"]["R"]))
print("Glaucoma cnt : Left eye = {} | Right eye = {}".format(test_cnt_oct["glaucoma"]["L"], test_cnt_oct["glaucoma"]["R"]))
print("################")


testing_data_oct = TestGenerateDataset(image_files = test_files_oct,
                                       labels_dict = label_dict_oct,
                                       img_res = 224,
                                       transform = pre_proc_func)
test_dataloader_oct = DataLoader(testing_data_oct,
                                  batch_size = batch_size,
                                  shuffle=True,
                                  num_workers=4)


print("Testing loop.....")
# cleaning computation from before....
prob_values = []
pred_values = []
gt_values = []
cnt = 0
for inputs, labels_index, file_path in tqdm(test_dataloader_oct, position=0, leave=True):
    inputs = inputs.to("cuda")
    preds  = oct_model(inputs)
    pred_label  = (preds > 0.5)*1
    prob_values.extend([i[0] for i in preds.detach().cpu().numpy()])
    pred_values.extend([i[0] for i in pred_label.detach().cpu().numpy()])
    gt_values.extend(labels_index)

    # csv data points.....
    pandas_oct_dict["patientId"].extend([i.split("/")[-1].split("_")[0] for i in file_path])
    eye_list = []
    date_list = []
    for i in file_path:
        pid_oct = i.split("/")[-1].split("_")[0]
        eyeId = i.split("/")[-1].split("_")[2]
        # in OCT dates vary according to patient IDs...
        entries = oct_df[(oct_df["PatientID"] == pid_oct) & (oct_df["match_col"] == eyeId)]["Eye"].iloc[0]
        eye_list.append(entries)
        
        date_list.append(oct_df[oct_df["match_col"] == eyeId]["ExamDate"].iloc[0])

    pandas_oct_dict["eye"].extend(eye_list)
    pandas_oct_dict["date"].extend(date_list)
    pandas_oct_dict["imageName"].extend([i.split("/")[-1].split(".")[0] for i in file_path])
    pandas_oct_dict["modelPrediction_prob"].extend([i[0] for i in preds.detach().cpu().numpy()])
    pandas_oct_dict["thresholded_op_0.5"].extend(["healthy" if i[0] == 1 else "glaucoma" for i in pred_label.detach().cpu().numpy()])
    pandas_oct_dict["groundTruth"].extend(["healthy" if i == 1 else "glaucoma" for i in labels_index])


gt_values = np.array(gt_values)
pred_values = np.array(pred_values)
prob_values = np.array(prob_values)
# precision :
oct_prec = precision_score(y_true = gt_values,
                              y_pred = pred_values,
                              average = "binary",
                              pos_label = 0,
                              zero_division = 0.0)
oct_recall = recall_score(y_true = gt_values,
                              y_pred = pred_values,
                              average = "binary",
                              pos_label = 0,
                              zero_division = 0.0)

oct_f1 = (2*(oct_prec*oct_recall))/(oct_prec + oct_recall)
oct_auc = roc_auc_score(y_true  = gt_values, y_score = prob_values)

print("OCT scores : ")
print("Precision : {:.3f} | Recall : {:.3f} | F1 Score : {:.3f} | AUC Score : {:.3f}".format(oct_prec, oct_recall, oct_f1, oct_auc))
print("#####################################################################")

oct_results = pd.DataFrame(data=pandas_oct_dict)
oct_results.to_csv("./single_modality_retFound_oct_results_csv_aligned.csv")
