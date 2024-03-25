#Referencing tutorials:
# - https://www.youtube.com/watch?v=iYisBtT6zvs
# - https://rumn.medium.com/how-to-create-a-custom-pytorch-dataset-with-a-csv-file-e64b89bc2dcc

#import the libraries relevent for data analysis
#will need to install pandas: conda install pandas
import pandas as pd

import PIL
import numpy as np

#import the libraries relevent for pytorch
#will need to install pytorch: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
import torch
import torchvision


class NWB_Dataset(torch.utils.data.Dataset):
    def __init__(self, file_config, unit=0, train=True):
        if(train):
            self.data_mode = "train"
        else:
            self.data_mode = "validation"
        
        self.file_config = file_config
        self.unit = unit
        self.data_transforms = {
            "train" : torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((224,224)),
                    torchvision.transforms.ToTensor(),
                    #may need to test without normalization (for now, values used are recommended for ResNet50 images)
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]
            ),
            "validation" : torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((224,224)),
                    torchvision.transforms.ToTensor(),
                    #may need to test without normalization (for now, values used are recommended for ResNet50 images)
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]
            )
        }

        self.dataframe = pd.read_csv(self.file_config[self.data_mode][self.unit])
        self.transform = self.data_transforms[self.data_mode]

    def __len__(self):
        return(len(self.dataframe))
    
    def __getitem__(self, idx):
        if(torch.is_tensor(idx)):
            idx = idx.tolist()
        
        #assuming the csv has 2 columns named path & label
        sample = {
            "image" : self.read_image(self.dataframe["image"].iloc[idx]),
            "label" : self.dataframe["label"].iloc[idx]
        }
        if(self.transform != None):
            sample["image"] = self.transform(sample["image"])
            sample["label"] = np.expand_dims(sample["label"], -1)

        return(sample["image"], sample["label"])

    def read_image(self, path):
        return(PIL.Image.open(path))
    
