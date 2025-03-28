# updated_multimodal_retFound_repo
New Structure Repository for the multimodal (fundus-oct) classification using RetFound

Explainations of all files :
1. attentionMapsRetFound.ipynb : File to understand per layer attention of vision trasformers on oct & fundus images.
2. constants.py : Most of the common variables such as file paths, loss weights, learning rate etc. used in model training are taken from this file.
3. data_loader.py : File that contains augmentation functions & data loading classes for both single + multiple model (eg RetFound + llava) cases.
4. focal_loss_imp.py : File that contains focal loss custom implementation for normal precision & automatic mixed precision cases.
5. gradCamVis.ipynb : GradCam implementation for model output visualisation.
6. models_mae & models_vit : Files from retFound repository to build the retFound model for pre-training & fine tuning respectively.
7. multi_modal_constants.py : Constants of training for multimodal setup.
8. multi_modal_data_loader.py : Data loader for multimodal setup.
9. multi_modal_rFmodel.py : Building the fundus-oct based multi-modal model.
10. multimodal_train_retfound.py : Multimodal training script.
11. rFmodel.py : Single modality model building.
12. test_evaluate_multi.py : Testing script for multi-modal model.
13. test_evaluate_single.py : Testing script for uni-modal model.
14. train_retfound.py : Training script for uni-modal model.
