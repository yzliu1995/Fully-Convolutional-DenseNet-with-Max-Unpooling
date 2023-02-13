# -*- coding: utf-8 -*-
"""DenseNet and DenseNet V2 implementation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KmHyLfDUnRSZOU9AR2UABnsAFLP8OdkJ

This file is developed by Joe Liu
"""

# Commented out IPython magic to ensure Python compatibility.
# Standard libraries
import os
import math
import numpy as np 
import time

# Imports for plotting
import matplotlib.pyplot as plt

from matplotlib.colors import to_rgba
import seaborn as sns
import matplotlib as mpl
import optparse
from mycolorpy import colorlist as mcp
import random

mpl.rcParams["legend.markerscale"] = 0.5

parser = optparse.OptionParser()
parser.add_option("-t", action="store_true", dest="threeClass", help = "whether 3-class segmentation or 4-class segmentation: 3-class segmentation")
parser.add_option("-f", action="store_false", dest="threeClass", help = "whether 3-class segmentation or 4-class segmentation: 4-class segmentation")


parser.add_option("-e", type="int", dest="epochs", help = "total number of epochs")

parser.add_option("-o", action="store_true", dest="model", help = "whether FCDenseNet or FCDenseNet V2: FCDenseNet")
parser.add_option("-u", action="store_false", dest="model", help = "whether FCDenseNet or FCDenseNet V2: FCDenseNet V2")

parser.add_option("-d", type="int", dest="device", help = "choose a device for GPU computing")

parser.add_option("-b", type="int", dest="batchsize", help = "batch size")

parser.add_option("-k", type="int", dest="numBlock", help = "number of dense blocks on each path")

parser.add_option("-l", "--file",
                  action="store", type="string", dest="filename", help = "file name for saving figures")

parser.add_option("-w", "--weights",
                  action="store", type="string", dest="weights", help = "file path for saving weights")

(options, args) = parser.parse_args()

# read in datasets
imgs_meds = np.load("./dataset/images_medseg.npy")
imgs_radios = np.load("./dataset/images_radiopedia.npy")

masks_meds = np.load("./dataset/masks_medseg.npy")
masks_radios = np.load("./dataset/masks_radiopedia.npy")

imgs_all = np.concatenate([imgs_meds, imgs_radios], axis = 0)
masks_all = np.concatenate([masks_meds, masks_radios], axis = 0)

selected = [i for i in range(imgs_all.shape[0]) if np.max(masks_all[i,...,0])*1 + np.max(masks_all[i,...,1])*1 + np.max(masks_all[i,...,2])*1 > 0 and np.sum(masks_all[i,...,3])/(masks_all[i,...,3].shape[0]*masks_all[i,...,3].shape[1])<= 0.9 and np.max(np.sum(masks_all[i,...,:2], axis = 2)+ np.sum(masks_all[i,...,3:4], axis = 2)) == 1]

print("Total # images: {}".format(len(selected)))

imgs = imgs_all[selected,...]
lbls = masks_all[selected,...]

for i in range(lbls.shape[0]):
  lbls[i,...,2] = np.ones((lbls.shape[1],lbls.shape[2]))- lbls[i,...,0] - lbls[i,...,1] - lbls[i,...,3]

print("min pixel value: {:.4f}, max pixel value: {:.4f}".format(np.min(imgs), np.max(imgs)))

pix_nums = np.array([np.sum(lbls[...,0]), np.sum(lbls[...,1]), np.sum(lbls[...,2]), np.sum(lbls[...,3])])

print("Total # images: {}".format(np.sum(pix_nums)/512/512))

img_pix_nums = np.array([len([i for i in range(lbls.shape[0]) if np.sum(lbls[i,..., c]) > 0])*lbls.shape[1]*lbls.shape[2] for c in range(4)])

print("Pixel count in million: {:.4f},{:.4f},{:.4f},{:.4f}".format(*pix_nums/(10**6)))

print("Image pixel count in million: {:.4f},{:.4f},{:.4f},{:.4f}".format(*img_pix_nums/(10**6)))

plt.imshow(np.sum(np.sum(lbls[...,:2], axis=3), axis=0), cmap = "hot")
plt.axis('off')
plt.savefig("./figures/Accumulation of labels.pdf", dpi=600)

def scaleCT(img):
  """
  scales the image by substracting the minimum pixel value from all pixel values 
  and then dividing them by the range of pixel values (i.e., maximum - minimum)
  """
  return((img-np.min(img))/(np.max(img) - np.min(img)))


# Create 3 subplots
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(np.squeeze(imgs[0,]), cmap="gray")
ax1.axis('off')
ax2.imshow(np.multiply(np.squeeze(scaleCT(imgs[0,])), (-1)*(np.squeeze(lbls[0,:,:,3]) - 1)), cmap="gray")
ax2.axis('off')
ax3.imshow(np.squeeze(lbls[0,:,:,0])*1 + np.squeeze(lbls[0,:,:,1]) * 2 + np.squeeze(lbls[0,:,:,2]) * 3, cmap = "coolwarm")
names = ["background", "GGO", "consolidations", "other lungs"]  
getLegend = lambda i: ax3.plot([],color = mcp.gen_color(cmap="coolwarm",n=4)[i],
                        label=names[i], ls="", marker="o")[0]
ax3.axis('off')
ax3.legend(handles=[getLegend(i) for i in range(4)],  fontsize = "xx-small", bbox_to_anchor=(1, 1.3))
plt.savefig("./figures/Dataset sample.pdf", dpi=600)

def compute_median_freq_balancing(lbl):
    """
    computes median frequency balancing
    """
    classes = lbl.shape[3]
    all_freq = []
    for i in range(int(lbl.shape[0])):
        counts = []
        for j in range(classes):
            counts.append(np.sum(lbl[i,...,j])/(lbl.shape[1]**2))
        freq = np.array(counts)
        freq = freq[None,...]
        all_freq.append(freq)
    freqs = np.concatenate(all_freq, axis = 0)
    FCs = np.array([np.sum(lbls[:int(lbl.shape[0]),...,i])  for i in range(lbls.shape[3])])/(np.apply_along_axis(lambda v: np.sum(v>0), 0, freqs) * (lbl.shape[1]**2))

    return np.median(FCs)/FCs

if options.threeClass:

  random.seed(2021)
  random.shuffle(selected)

  imgs = imgs_all[selected,...]
  lbls = masks_all[selected,...]

  lbls_multi = lbls

  lbls = np.concatenate([np.sum(lbls_multi[...,0:1,None]+lbls_multi[...,1:2, None], axis = 3), lbls_multi[...,2:4]], axis = 3)

else:
  random.seed(2021)
  random.shuffle(selected)

  imgs = imgs_all[selected,...]
  lbls = masks_all[selected,...]

# training
import torch
LR = 1e-4
LR_DECAY = 0.995
DECAY_EVERY_N_EPOCHS = 1
N_EPOCHS = options.epochs



import albumentations

import cv2

SOURCE_SIZE = 512
TARGET_SIZE = 256


train_augs = albumentations.Compose([
    albumentations.Rotate(limit=360, p=0.5, border_mode=cv2.BORDER_REPLICATE),
    albumentations.RandomSizedCrop((int(SOURCE_SIZE * 0.75), SOURCE_SIZE), 
                                   TARGET_SIZE, 
                                   TARGET_SIZE, 
                                   interpolation=cv2.INTER_NEAREST),
    albumentations.HorizontalFlip(p=0.5),

])

val_augs = albumentations.Compose([
    albumentations.Resize(TARGET_SIZE, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
])

from torchvision import transforms as T

class datasetCT:   
    def __init__(
            self, 
            images, 
            masks,
            augmentations=None
    ):
        self.images = images
        self.masks = masks
        self.augmentations = augmentations

    
    def __getitem__(self, i):
        if options.threeClass:
          image = np.multiply(np.squeeze((self.images[i,...]-np.min(self.images[i,...]))/(np.max(self.images[i,...])-np.min(self.images[i,...]))), (-1)*(np.squeeze(self.masks[i,...,2]) - 1))
        else:
          image = np.multiply(np.squeeze((self.images[i,...]-np.min(self.images[i,...]))/(np.max(self.images[i,...])-np.min(self.images[i,...]))), (-1)*(np.squeeze(self.masks[i,...,3]) - 1))
        
        if options.threeClass:
          mask = np.squeeze(self.masks[i,...,0])*1 + np.squeeze(self.masks[i,...,1]) * 2
        else:
          mask = np.squeeze(self.masks[i,...,0])*1 + np.squeeze(self.masks[i,...,1]) * 2 + np.squeeze(self.masks[i,...,2]) * 3
        
        if self.augmentations is not None:
            sample = self.augmentations(image=image, mask=mask)
            
            image, mask = sample['image'], sample['mask']
        
        if self.augmentations is None:
            image = image
            mask = mask
        
        t = T.Compose([T.ToTensor()])
        image = t(image).type(torch.cuda.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.LongTensor)
    
        return image, mask
    
    def __len__(self):
        return len(self.images)

from models import tiramisu
import utils.training as train_utils
import torch.nn as nn
import time

nsamples = imgs.shape[0]
trainSplit = 0.8
valSplit  = 0.2
batch_size = options.batchsize
numFolds = 5

# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

torch.cuda.set_device(options.device)

all_trainLosses = []
all_valLosses = []

all_trainAccs = []
all_valAccs = []

all_train_times = []
all_val_times = []

for i in range(numFolds):


  
  torch.cuda.manual_seed(0)  
  if options.model:

    if options.threeClass:
      model = tiramisu.FCDenseNet(in_channels=1, n_classes=3, down_blocks=(4,)*options.numBlock, up_blocks=(4,)*options.numBlock, bottleneck_layers=4, growth_rate=12).cuda()
    else:
      model = tiramisu.FCDenseNet(in_channels=1, n_classes=4, down_blocks=(4,)*options.numBlock, up_blocks=(4,)*options.numBlock, bottleneck_layers=4, growth_rate=12).cuda()
  else:
    if options.threeClass:
      model = tiramisu.FCDenseNetV2(in_channels=1, n_classes=3, down_blocks=(4,)*options.numBlock, up_blocks=(4,)*options.numBlock, bottleneck_layers=4, growth_rate=12).cuda()
    else:
      model = tiramisu.FCDenseNetV2(in_channels=1, n_classes=4, down_blocks=(4,)*options.numBlock, up_blocks=(4,)*options.numBlock, bottleneck_layers=4, growth_rate=12).cuda()
  
  model.apply(train_utils.weights_init)
  optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

 
  val_dset = datasetCT(imgs[round(valSplit*i*nsamples):round(valSplit*(i+1)*nsamples),...],
                     lbls[round(valSplit*i*nsamples):round(valSplit*(i+1)*nsamples),...],None) 
   
  train_dset = datasetCT(np.delete(imgs, range(round(valSplit*i*nsamples),round(valSplit*(i+1)*nsamples)), axis = 0),
                       np.delete(lbls, range(round(valSplit*i*nsamples),round(valSplit*(i+1)*nsamples)), axis = 0),
                       train_augs)

  print("validation samples: {} training samples: {}".format(val_dset.__len__(), train_dset.__len__()))
  
  if options.threeClass:
    
    class_weight = compute_median_freq_balancing(np.delete(lbls, range(round(valSplit*i*nsamples),round(valSplit*(i+1)*nsamples)), axis = 0))

    class_weight_copy = class_weight

    class_weight = np.array([class_weight_copy[2], class_weight_copy[0], class_weight_copy[1]])
    print("class weight: class 0: {:.4f}, class 1: {:.4f}, class 2: {:.4f}".format(*class_weight))
  
  else:
    class_weight = compute_median_freq_balancing(np.delete(lbls, range(round(valSplit*i*nsamples),round(valSplit*(i+1)*nsamples)), axis = 0))
    class_weight_copy = class_weight
    class_weight = np.array([class_weight_copy[3], class_weight_copy[0], class_weight_copy[1], class_weight_copy[2]])
    print("class weight: class 0: {:.4f}, class 1: {:.4f}, class 2: :{:.4f}, class 3: {:.4f}".format(*class_weight))



  class_weight = torch.FloatTensor(class_weight)
  criterion = nn.CrossEntropyLoss(weight=class_weight.cuda()).cuda()

  train_loader = torch.utils.data.DataLoader(
      train_dset, batch_size=batch_size, shuffle=True)
  val_loader = torch.utils.data.DataLoader(
      val_dset, batch_size=1, shuffle=False)
  

  trainLosses = []
  valLosses = []

  trainAccs = []
  valAccs = []

  train_times = []
  val_times = []


  epochs = []

  for epoch in range(1, N_EPOCHS+1):
      since = time.time()
 
      ### Train ###
      trn_loss, trn_err = train_utils.train(
          model, train_loader, optimizer, criterion, epoch)
      print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(
          epoch, trn_loss, 1-trn_err))    
      time_elapsed = time.time() - since  
      print('Train Time {:.0f}m {:.0f}s'.format(
          time_elapsed // 60, time_elapsed % 60))
    
      trainLosses.append(trn_loss.cpu().detach().numpy())
      trainAccs.append(1-trn_err.cpu().detach().numpy())
      epochs.append(epoch)
      train_times.append(time_elapsed)
      ### Test ###
      val_loss, val_err = train_utils.test(model, val_loader, criterion, epoch)    
      print('Val - Loss: {:.4f} | Acc: {:.4f}'.format(val_loss, 1-val_err))
      time_elapsed = time.time() - since  
      print('Total Time {:.0f}m {:.0f}s\n'.format(
          time_elapsed // 60, time_elapsed % 60))
      
      valLosses.append(val_loss.cpu().detach().numpy())
      valAccs.append(1-val_err.cpu().detach().numpy())

      val_times.append(time_elapsed)
      ### Checkpoint ###    
      train_utils.save_weights(model, epoch, val_loss, val_err, fold = i, WEIGHTS_PATH = options.weights)

      ### Adjust Lr ###
      train_utils.adjust_learning_rate(LR, LR_DECAY, optimizer, 
                                      epoch, DECAY_EVERY_N_EPOCHS)
  
  all_trainLosses.append(trainLosses)
  all_valLosses.append(valLosses)

  all_trainAccs.append(trainAccs)
  all_valAccs.append(valAccs)

  all_train_times.append(train_times)
  all_val_times.append(val_times)

  fig, ax1 = plt.subplots()

  color = 'tab:red'
  ax1.set_xlabel('Epoch (s)')
  ax1.set_ylabel('Loss', color=color)
  ax1.plot(epochs, trainLosses, color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:blue'
  ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
  ax2.plot(epochs, trainAccs, color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  plt.savefig("./figures/"+str(i)+"_"+options.filename+"_training accuracy and loss_denseNet.pdf", dpi = 600)


  fig, ax1 = plt.subplots()

  color = 'tab:red'
  ax1.set_xlabel('Epoch (s)')
  ax1.set_ylabel('Loss', color=color)
  ax1.plot(epochs, valLosses, color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:blue'
  ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
  ax2.plot(epochs, valAccs, color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  plt.savefig("./figures/"+str(i)+"_"+options.filename+"_validation accuracy and loss_denseNet.pdf", dpi = 600)

