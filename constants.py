# file to store all code constants for training

# file paths....
modality_type = "oct"

project_dir = "/tscc/nfs/home/vejoshi/oct_fundus_project/"
rFound_var  = "/tscc/nfs/home/vejoshi/oct_fundus_project/ret_found_exp/RETFound_MAE/pre_trained_weights/"
dataset_dir    = project_dir + "oct_fundus_dataset/"

r_found_oct_og_weights = rFound_var + "RETFound_oct_weights.pth"
r_found_fun_og_weights = rFound_var + "RETFound_cfp_weights.pth"

ret_found_ext_og_weights = r_found_fun_og_weights if modality_type == "fundus" else r_found_oct_og_weights

dataset_path   = dataset_dir + "fundus_images_csv_aligned/" if modality_type == "fundus" else dataset_dir + "oct_images_csv_aligned/"
label_ids      = dataset_dir + "multi_modal_fundus_train_val_image_names.pickle" if modality_type == "fundus" else dataset_dir + "multi_modal_oct_train_val_image_names.pickle"
label_path     = dataset_dir + "fundus_labels_binary_version_csv_algined.pickle" if modality_type == "fundus" else dataset_dir + "oct_labels_binary_version_csv_algined.pickle"
test_ids_paths = dataset_dir + "test_patient_ids_fundus_oct_csv_aligned.pickle"

# training constants
use_aug = True
training_nature = "supervised_only"
model_name = "retFound"
input_shape = (224,224,3)
unfreeze_perc = 1.0
frozen_epochs = 10 # keep the base model weights frozen for these many epochs otherwise the classification heads would damage the
last_lr = 1e-06
learning_rate = 5e-04
dropout = 0.1
warmup_epochs = 10
focal_weight = 2.8
l2_reg = 1e-05
pool_type = "max"
dense_1 = 8
dense_2 = 12
dense_3 = 24
batch_size = 64
decision_threshold = 0.5 # used by metrics
train_epochs = 100
num_train_samples_viz = 4
patience = 10
reduce_lr_patience = 3
lr_scale = 0.1
lab_smooth = 0.13
aug_prob = 0.6

# Label constants...
"""
label_mapping = {"healthy"  : [1,0,0],
                 "suspects" : [0,1,0],
                 "glaucoma" : [0,0,1]}
"""
label_mapping = {"glaucoma" : 0,
                 "healthy"  : 1}

num_classes = len(label_mapping)

# Save directory ##########################################################
# Creating directory to save runs & best weights.....
save_dir_name = "./experiments/test_" + modality_type + "/"
model_save_name = model_name + "_shape_" + str(input_shape[0]) + "_dp_" + \
str(dropout) + "_lr_" + str(learning_rate) + "_csv_aligned"
