"""
#EXAMPLE:
#Import the PyTorch libraries
import torch
import torchvision
#Define the ResNet50 model
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
#Change the final connection of the model to output a single value (spiking rate)
model.fc = torch.nn.Linear(in_features=2048, out_features=1)
"""

#Referencing tutorials: 
# - https://medium.com/@lucrece.shin/chapter-3-transfer-learning-with-resnet50-from-dataloaders-to-training-seed-of-thought-67aaf83155bc
# - https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch
# - https://www.youtube.com/watch?v=iYisBtT6zvs
# - https://www.geeksforgeeks.org/introduction-to-explainable-aixai-using-lime/
# - https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb

#import the libraries for the data analysis
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#import the libraries relevent for pytorch
#will need to install pytorch: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
import torch
import torchvision

#import LIME
#install LIME for explanable AI (XAI) toolset: pip install lime
import lime

#insstall scikit images for image segmentation (LIME Highlighting)
#need to install scikit image: pip install scikit-image
import skimage

#import custom dataset
from PytorchDataset import NWB_Dataset

#Get the data
file_config = {
    "train" : {
        0  : "data/train/unit0.csv",
        1  : "data/train/unit1.csv",
        2  : "data/train/unit2.csv",
        3  : "data/train/unit3.csv",
        4  : "data/train/unit4.csv",
        5  : "data/train/unit5.csv",
        6  : "data/train/unit6.csv",
        7  : "data/train/unit7.csv",
        8  : "data/train/unit8.csv",
        9  : "data/train/unit9.csv",
        10 : "data/train/unit10.csv",
        11 : "data/train/unit11.csv",
        12 : "data/train/unit12.csv",
        13 : "data/train/unit13.csv",
        14 : "data/train/unit14.csv"
    },
    "validation": {
        0  : "data/validation/unit0.csv",
        1  : "data/validation/unit1.csv",
        2  : "data/validation/unit2.csv",
        3  : "data/validation/unit3.csv",
        4  : "data/validation/unit4.csv",
        5  : "data/validation/unit5.csv",
        6  : "data/validation/unit6.csv",
        7  : "data/validation/unit7.csv",
        8  : "data/validation/unit8.csv",
        9  : "data/validation/unit9.csv",
        10 : "data/validation/unit10.csv",
        11 : "data/validation/unit11.csv",
        12 : "data/validation/unit12.csv",
        13 : "data/validation/unit13.csv",
        14 : "data/validation/unit14.csv"
    }
}
unit_used = 8
train_dataloader = torch.utils.data.DataLoader(
    NWB_Dataset(file_config, unit=unit_used, train=True),
    batch_size=16,
    shuffle=True
    #num_workers=2
)

#select the device to run the model on: GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#obtain a pretrained ResNet50 model
model = torchvision.models.resnet50(weights=None)#weights=torchvision.models.ResNet50_Weights.DEFAULT)
#freeze all the gradients to prevent them from changing
#for param in model.parameters():
#    param.requires_grad = False
#change the last layer of the model to return the average spike rate at each time bin (30 bins: -1 to 3 seconds after image onset)
model.fc = torch.nn.Linear(in_features=2048, out_features=1)
model = model.to(device) #ensure the model is sent to cuda after parameters are changed
#define the loss function: SmoothL1Loss for stable loss computation (L2 near 0 and L1 beyond +/-beta)
loss_function = torch.nn.SmoothL1Loss(beta=1.0)
#define the optimizer: adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#Define the model training process
model.train()
num_epochs = 100
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        #print(images)
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if((i+1)%5 == 0):
            #print the current epoch, batch (step), and current model loss
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")




validation_dataloader = torch.utils.data.DataLoader(
    NWB_Dataset(file_config, unit=unit_used, train=False),
    batch_size=32,
    shuffle=False
    #num_workers=2
)

model.eval()
for i, (images, labels) in enumerate(validation_dataloader):
    #print(images)
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    loss = loss_function(outputs, labels)

    for i in range(len(labels)):
        print(f"{labels[i].item():.3f}, {outputs[i].item():.3f}, {outputs[i].item()-labels[i].item():.3f}")
    print("-----")
    print(f"Loss: {loss.item():.4f}")

def preprocess_transform():
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize
    ])    

    return transf


def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    #probs = F.softmax(logits, dim=1)
    return logits.detach().cpu().numpy()

#Get the validation image
img, label = validation_dataloader[0]

#Evalute the model using LIME
lime_explainer = lime.lime_image.LimeImageExplainer()
explanation = lime_explainer.explain_instance(
    img.numpy(),
    batch_predict,
    top_labels=1,
    hide_color=0,
    num_samples=1000
)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
img_boundry1 = skimage.segmentation.mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry1)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
img_boundry2 = skimage.segmentation.mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry2)