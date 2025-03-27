# Data loader centric imports....
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from multi_modal_constants import *
import PIL
import numpy as np


class H_random_translate(object):
    """
    Class to perform random Horizontal translation with image wraping....

    Args:
        perc_motion : (float) maximum % of image width to move left or right (sampled randomly)
    """

    def __init__(self, perc_motion = 0.1):
        self.perc_motion = perc_motion

    def __call__(self, sample):

        # performing wraping...
        total_width = sample.size[1]*3
        max_height  = sample.size[0]
        new_im = PIL.Image.new('RGB', (total_width, max_height))

        # concating the same image on right & left....
        x_offset = 0
        for i in range(3):
            new_im.paste(sample, (x_offset,0))
            x_offset+= sample.size[1]

        # random ranges for horizontal translation 
        motion_limit = int(sample.size[1]*self.perc_motion)
        crop_coord = torch.randint(-1*motion_limit, motion_limit, (1,)).numpy()[0]
    
        # wrapping cropping...
        proc_img = transforms.functional.crop(new_im, top = 0, left = sample.size[1] + crop_coord, height = 224, width = 224) 
        
        return proc_img

# Data-loader class
class MultiModalGenerateDataset(Dataset):

    def __init__(self,
                 pair_files,
                 labels_dict,
                 img_res = 224,
                 augment = False,
                 apply_random_prob = 0.2,
                 shuffle = True,
                 transform = None,
                 split_flag = "training"):


        # loading image paths & labels....
        self.image_list = pair_files # fundus, oct
        
        # one hot encoded labels......
        self.labels = [label_mapping[labels_dict[i[0].split("/")[-1]]] for i in self.image_list]

        self.labels = np.array(self.labels)
        self.image_list = np.array(self.image_list)

        print("Number of image pairs loaded for {} split : {} images".format(split_flag, len(self.image_list)))
        print("Number of labels loaded for {} split      : {} labels".format(split_flag, len(self.labels)))

        # model specific pre-processing function.....
        self.transform = transform

        # other training constants.....
        self.img_res = (img_res, img_res)
        self.augment = augment
        self.apply_random_prob = apply_random_prob
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_size = len(self.image_list)

        # inverse of class proportion serves as the class weights 
        # more frequent the class is, less is the associated class weight....
        self.class_weights = {0 : (1/len(self.labels[self.labels[:] == 0]))*(self.total_size/ num_classes),
                              1 : (1/len(self.labels[self.labels[:] == 1]))*(self.total_size/ num_classes)}

        print("Class weights are : ")
        for ct,i in enumerate(label_mapping):
            freq = len(self.labels[self.labels[:] == ct])
            print("Class Name : {} | Frequency : {} | Weight : {}".format(i, freq, self.class_weights[ct]))


    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):

        # fetching images & label....
        img_path = self.image_list[idx][0]
        larger_scale = img_path.replace("fundus_images_csv_aligned", "fundus_images_csv_aligned_2_5_scale")
        fundus_img_inp =  PIL.Image.open(larger_scale).convert('RGB').resize(self.img_res)
        oct_img_inp    =  PIL.Image.open(self.image_list[idx][1]).convert('RGB').resize(self.img_res)

        class_label = self.labels[idx]

        # point to experiment further........
        perc = 0.8
        fundus_aug_transforms = None
        oct_aug_transforms = None
        if self.augment:

        
            # augmentation params computed for fundus images....
            """
            fundus_aug_transforms = transforms.Compose([transforms.RandomResizedCrop(size = 224, scale = (0.9,1.0)),
                                                        transforms.RandomHorizontalFlip(),
                                                        #transforms.Pad((50,50,50,50),padding_mode = "reflect"),
                                                        transforms.RandomAffine(degrees = (-5,5), translate = (0.1,0.1), interpolation=transforms.InterpolationMode.BILINEAR),])
                                                        #transforms.CenterCrop(size = self.img_res[0]),])
                                                        #transforms.ColorJitter(brightness = (1.0,1.2), contrast = (1.0,1.2), saturation = (1.0,1.2)),
                                                        #transforms.GaussianBlur(5, (1.0,3.0))])
            """
            fundus_aug_transforms = transforms.Compose([transforms.RandomAffine(degrees = (-30,30),translate = (0.1,0.1),interpolation=transforms.InterpolationMode.BILINEAR),
                                                     transforms.CenterCrop(size = int(fundus_img_inp.size[0]*perc)),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ColorJitter(brightness = 0.3),
                                                     transforms.GaussianBlur(5, (1.0,3.0)),
                                                     transforms.Resize(224)])

            # oct images augmentations.........
            oct_aug_transforms = transforms.Compose([H_random_translate(perc_motion = 0.25),
                                                     transforms.RandomAffine(degrees = (-5,5), 
                                                                             translate = (0.0,0.1), 
                                                                             interpolation=transforms.InterpolationMode.BILINEAR),
                                                     transforms.ColorJitter(brightness = 0.3)])


        
        # randomness to apply all augmentations....
        proc_img_fundus = fundus_img_inp
        proc_img_oct = oct_img_inp

        if fundus_aug_transforms is not None:
            toss = np.random.choice([0,1],p=[1-self.apply_random_prob, self.apply_random_prob])
            if toss == 1:
                proc_img_fundus = fundus_aug_transforms(fundus_img_inp)

        if oct_aug_transforms is not None:
            toss = np.random.choice([0,1],p=[1-self.apply_random_prob, self.apply_random_prob])
            if toss == 1:
                proc_img_oct = oct_aug_transforms(oct_img_inp)


        
        # returning fundus image, oct image, one hot encoded label, image path pair....
        # self transform is model specific compulsory pre-processing
        return self.transform(proc_img_fundus), self.transform(proc_img_oct), class_label, 0



class TestMultiModalGenerateDataset(Dataset):
    def __init__(self,
                 pair_files,
                 labels_dict,
                 img_res = 224,
                 transform = None):

        # loading image paths & labels....
        self.image_list = pair_files # fundus, oct

        # one hot encoded labels......
        self.labels = [label_mapping[labels_dict[i[0].split("/")[-1]]] for i in self.image_list]

        self.labels = np.array(self.labels)
        self.image_list = np.array(self.image_list)

        # model specific pre-processing function.....
        self.transform = transform

        # other training constants.....
        self.img_res = (img_res, img_res)
        self.batch_size = batch_size
        self.total_size = len(self.image_list)

    def __len__(self):
        return self.total_size


    def __getitem__(self, idx):

        # fetching images & label....
        larger_scale = self.image_list[idx][0].replace("fundus_images_csv_aligned", "fundus_images_csv_aligned_2_5_scale")
        fundus_img_inp =  PIL.Image.open(larger_scale).convert('RGB').resize(self.img_res)
        oct_img_inp    =  PIL.Image.open(self.image_list[idx][1]).convert('RGB').resize(self.img_res)
        class_label = self.labels[idx]

        return self.transform(fundus_img_inp), self.transform(oct_img_inp), class_label, self.image_list[idx][0], self.image_list[idx][1]
