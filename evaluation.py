"""
This file is developed by Joe Liu
"""

import os
import math
import numpy as np 
import time

# Imports for plotting
import matplotlib.pyplot as plt
import optparse
from matplotlib.colors import to_rgba
from mycolorpy import colorlist as mcp
import seaborn as sns
import matplotlib as mpl
import random
mpl.rcParams["legend.markerscale"] = 0.5

from models import tiramisu
import utils.training as train_utils
import torch.nn as nn
import time

from torchvision import transforms as T

parser = optparse.OptionParser()

# whether 3-class segmentation or 4-class segmentation
parser.add_option("-t", action="store_true", dest="threeClass", help = "whether 3-class segmentation or 4-class segmentation: 3-class segmentation")
parser.add_option("-f", action="store_false", dest="threeClass", help = "whether 3-class segmentation or 4-class segmentation: 4-class segmentation")

parser.add_option("-o", action="store_true", dest="model", help = "whether FCDenseNet or FCDenseNet V2: FCDenseNet")
parser.add_option("-u", action="store_false", dest="model", help = "whether FCDenseNet or FCDenseNet V2: FCDenseNet V2")
parser.add_option("-k", type="int", dest="numBlock", help = "number of dense blocks on each path")

parser.add_option("-w", "--weights",
                  action="store", type="string", dest="weights", help = "file path where weights were saved")

parser.add_option("-d", type="int", dest="device", help = "choose a device for GPU computing")

(options, args) = parser.parse_args()

# read in datasets
imgs_meds = np.load("./dataset/images_medseg.npy")
imgs_radios = np.load("./dataset/images_radiopedia.npy")

masks_meds = np.load("./dataset/masks_medseg.npy")
masks_radios = np.load("./dataset/masks_radiopedia.npy")

imgs_all = np.concatenate([imgs_meds, imgs_radios], axis = 0)
masks_all = np.concatenate([masks_meds, masks_radios], axis = 0)

selected = [i for i in range(imgs_all.shape[0]) if np.max(masks_all[i,...,0])*1 + np.max(masks_all[i,...,1])*1 + np.max(masks_all[i,...,2])*1 > 0 and np.sum(masks_all[i,...,3])/(masks_all[i,...,3].shape[0]*masks_all[i,...,3].shape[1])<= 0.9 and np.max(np.sum(masks_all[i,...,:2], axis = 2)+ np.sum(masks_all[i,...,3:4], axis = 2)) == 1]

imgs = imgs_all[selected,...]
lbls = masks_all[selected,...]

for i in range(lbls.shape[0]):
  lbls[i,...,2] = np.ones((lbls.shape[1],lbls.shape[2]))- lbls[i,...,0] - lbls[i,...,1] - lbls[i,...,3]

def scaleCT(img):
  """
  scales the image by substracting the minimum pixel value from all pixel values 
  and then dividing them by the range of pixel values (i.e., maximum - minimum)
  """
  return((img-np.min(img))/(np.max(img) - np.min(img)))

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

import torch
LR = 1e-4
LR_DECAY = 0.995
DECAY_EVERY_N_EPOCHS = 1
N_EPOCHS = 50
torch.cuda.manual_seed(0)

import albumentations

import cv2

SOURCE_SIZE = 512
TARGET_SIZE = 256


train_augs = albumentations.Compose([
    albumentations.Rotate(limit=360, p=0.9, border_mode=cv2.BORDER_REPLICATE),
    albumentations.RandomSizedCrop((int(SOURCE_SIZE * 0.75), SOURCE_SIZE), 
                                   TARGET_SIZE, 
                                   TARGET_SIZE, 
                                   interpolation=cv2.INTER_NEAREST),
    albumentations.HorizontalFlip(p=0.5),

])

val_augs = albumentations.Compose([
    albumentations.Resize(TARGET_SIZE, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
])

nsamples = imgs.shape[0]
trainSplit = 0.8
valSplit  = 0.2
numFolds = 5

# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

torch.cuda.set_device(options.device)

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

def sensitivity(pred, target, c):
  """
  TP/(TP+FN)
  
  """
  den = np.where(target == c)
  num = np.where(pred == c)
  TP = len(set(den[0]).intersection(num[0]))
  FN = len(den[0]) - TP 
  return TP/(TP + FN)


def specificity(pred, target, c):
  """
  TN/(TN+FP)

  """
  den = np.where(target != c)
  num = np.where(pred != c)
  TN = len(set(den[0]).intersection(num[0]))
  FP = len(den[0]) - TN 
  return TN /(FP + TN)


def dice(pred, target, c):
  """
  2 × TP/(2×TP+FP+FN)
  """
  den1 = np.where(target == c)
  num1 = np.where(pred == c)
  TP = len(set(den1[0]).intersection(num1[0]))
  FN = len(den1[0]) - TP 

  den2 = np.where(target != c)
  num2 = np.where(pred != c)
  TN = len(set(den2[0]).intersection(num2[0]))
  FP = len(den2[0]) - TN 

  return 2*TP/(2*TP+FP+FN)


def Gmean(pred, target, c):
  """
  sqrt(sensitivity × specificity)
  """
  return np.sqrt(sensitivity(pred, target, c)*specificity(pred, target, c))


def IoU(pred, target, c):
  """
  intersection/union
  """
  den = np.where(target == c)
  num = np.where(pred == c)
  inter = len(set(den[0]).intersection(num[0]))
  union =  len(set(den[0]).union(num[0]))

  return(inter/union)

def pixelA(pred, target):

  """
  percentage of pixels that are correctly classified
  """
  return len(np.where((pred-target) == 0)[0])/len(pred)


import glob

numFolds = 5

if options.threeClass:
  num_class = 3
  selSen = []
  selSpec = []
  selPixs = []
  selDic = []
  selGM = []
  selIOU = []
  for idx in range(numFolds):
    allWs = glob.glob(options.weights+str(idx)+"-weights"+"*")
    lossesV = [float(j.split("-")[3]) for j in allWs]
    print(min(lossesV))
    print(lossesV.index(min(lossesV)))

    print(glob.glob(options.weights+str(idx)+"-weights-"+str(lossesV.index(min(lossesV))+1)+"*")[0])
 
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

    model.load_state_dict(torch.load(glob.glob(options.weights+str(idx)+"-weights-"+str(lossesV.index(min(lossesV))+1)+"*")[0])['state_dict'])
    model.eval()

    val_img_fold = imgs[round(valSplit*idx*nsamples):round(valSplit*(idx+1)*nsamples),...]
    val_lbl_fold = lbls[round(valSplit*idx*nsamples):round(valSplit*(idx+1)*nsamples),...]
    val_dset = datasetCT(val_img_fold,
                        val_lbl_fold,None) 

    val_loader = torch.utils.data.DataLoader(
          val_dset, batch_size=1, shuffle=False)

    pixAs = []
    accuracies = {"0": [[],[],[],[],[],[]],
                  "1": [[],[],[],[],[],[]],
                  "2": [[],[],[],[],[],[]]
                  }
    k = 0

    img_idx = []
    from torch.autograd import Variable
    with torch.no_grad():
      for data, target in val_loader:
          k +=1
          data = Variable(data.cuda())
          target = Variable(target.cuda())
          output = model(data)
          # torch >=0.5
          pred = train_utils.get_predictions(output)
          p = np.squeeze(pred.detach().cpu().numpy()).flatten()
          t = np.squeeze(target.detach().cpu().numpy()).flatten()
          cs = np.unique(t)
          pxA = pixelA(p, t)
          pixAs.append(pxA)
          if(1 in cs):
              img_idx.append(k-1)

          for c in cs:
            sen = sensitivity(p, t, c)
            spec = specificity(p, t, c)
            dic = dice(p, t, c)
            gm = Gmean(p, t, c)
            iou = IoU(p,t,c)
            accuracies[str(c)][0].append(sen)
            accuracies[str(c)][1].append(spec)

            accuracies[str(c)][3].append(dic)
            accuracies[str(c)][4].append(gm)
            accuracies[str(c)][5].append(iou)
    print(k)

    
    selSen.append(np.median(accuracies[str(1)][0]))
    selSpec.append(np.median(accuracies[str(1)][1]))
    selPixs.append(np.median(pixAs))
    selDic.append(np.median(accuracies[str(1)][3]))
    selGM.append(np.median(accuracies[str(1)][4]))
    selIOU.append(np.median(accuracies[str(1)][5]))

    # Create 3 subplots
    query = accuracies[str(1)][5]
    qInx = query.index(max(query))
    val_dset_s = datasetCT(val_img_fold[img_idx[qInx]:(img_idx[qInx]+1),...],
                      val_lbl_fold[img_idx[qInx]:(img_idx[qInx]+1) ,...], None) 
    
    val_loader_s = torch.utils.data.DataLoader(
        val_dset_s, batch_size=1, shuffle=False)
    
    with torch.no_grad():
      for data, target in val_loader_s:
        data = Variable(data.cuda())
        target = Variable(target.cuda())
        output = model(data)
        # torch >=0.5
        pred = train_utils.get_predictions(output)
        p = np.squeeze(pred.detach().cpu().numpy())
        t = np.squeeze(target.detach().cpu().numpy())
        print(np.unique(t))
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(np.multiply(np.squeeze(scaleCT(val_img_fold[img_idx[qInx],])), (-1)*(np.squeeze(val_lbl_fold[img_idx[qInx],:,:,2]) - 1)), cmap="gray")
        ax1.axis('off')
        ax2.imshow(t, cmap = "coolwarm")
        ax2.axis('off')
        ax3.imshow(p, cmap = "coolwarm")
        ax3.axis('off')
        names = ["background", "GGO+consolidations", "other lungs"]  
        getLegend = lambda i: ax3.plot([],color = mcp.gen_color(cmap="coolwarm",n=num_class)[i],
                          label=names[i], ls="", marker="o")[0]
        ax3.axis('off')
        ax3.legend(handles=[getLegend(i) for i in range(num_class)],  fontsize = "xx-small", bbox_to_anchor=(1, 1.3))
        if options.model:
          plt.savefig("./figures/best result_"+str(idx)+"_" +str(options.numBlock) +"_fold_"+"FCdenseNet_3-class"+".pdf", dpi=600)
        else:
          plt.savefig("./figures/best result_"+str(idx)+"_" +str(options.numBlock)+"_fold_"+"FCdenseNet_V2_3-class"+".pdf", dpi=600)
  import pandas as pd
  d = {'Sensitivy': selSen, 'Specificity': selSpec, 'Dice': selDic, 'selGM': selGM, 'selIOU': selIOU, 'Pixel Accuracy': selPixs}
  df = pd.DataFrame(data=d)
  df.round(4)
  if options.model:
    df.to_csv("./results/"+str(options.numBlock)+"_densetNet binary class.csv",index=False)
  else:
    df.to_csv("./results/"+str(options.numBlock)+"_densetNet V2 binary class.csv",index=False)      

else:
  num_class = 4
  
  selSen = []
  selSpec = []
  selPixs = []
  selDic = []
  selGM = []
  selIOU = []

  selSen2 = []
  selSpec2 = []
  selPixs2 = []
  selDic2 = []
  selGM2 = []
  selIOU2 = []

  selSen3 = []
  selSpec3 = []
  selPixs3 = []
  selDic3 = []
  selGM3 = []
  selIOU3 = []

  for idx in range(numFolds):
    allWs = glob.glob(options.weights+str(idx)+"-weights"+"*")
    lossesV = [float(j.split("-")[3]) for j in allWs]
    print(min(lossesV))
    print(lossesV.index(min(lossesV)))

    print(glob.glob(options.weights+str(idx)+"-weights-"+str(lossesV.index(min(lossesV))+1)+"*")[0])
    
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

    model.load_state_dict(torch.load(glob.glob(options.weights+str(idx)+"-weights-"+str(lossesV.index(min(lossesV))+1)+"*")[0])['state_dict'])
    model.eval()
    val_img_fold = imgs[round(valSplit*idx*nsamples):round(valSplit*(idx+1)*nsamples),...]
    val_lbl_fold = lbls[round(valSplit*idx*nsamples):round(valSplit*(idx+1)*nsamples),...]
    val_dset = datasetCT(val_img_fold,
                        val_lbl_fold,None) 

    val_loader = torch.utils.data.DataLoader(
          val_dset, batch_size=1, shuffle=False)

    pixAs = []
    accuracies = {"0": [[],[],[],[],[],[]],
                  "1": [[],[],[],[],[],[]],
                  "2": [[],[],[],[],[],[]],
                  "3": [[],[],[],[],[],[]]
                  }
    k = 0

    img_idx = []
    from torch.autograd import Variable
    with torch.no_grad():
      for data, target in val_loader:
          k +=1
          data = Variable(data.cuda())
          target = Variable(target.cuda())
          output = model(data)
          # torch >=0.5
          pred = train_utils.get_predictions(output)
          p = np.squeeze(pred.detach().cpu().numpy()).flatten()
          t = np.squeeze(target.detach().cpu().numpy()).flatten()
          cs = np.unique(t)
          pxA = pixelA(p, t)
          pixAs.append(pxA)
          if 1 in cs:
              img_idx.append(k-1)

          for c in cs:
            sen = sensitivity(p, t, c)
            spec = specificity(p, t, c)
            dic = dice(p, t, c)
            gm = Gmean(p, t, c)
            iou = IoU(p,t,c)
            accuracies[str(c)][0].append(sen)
            accuracies[str(c)][1].append(spec)

            accuracies[str(c)][3].append(dic)
            accuracies[str(c)][4].append(gm)
            accuracies[str(c)][5].append(iou)
    print(k)

    
    selSen.append(np.median(accuracies[str(1)][0]))
    selSpec.append(np.median(accuracies[str(1)][1]))
    selPixs.append(np.median(pixAs))
    selDic.append(np.median(accuracies[str(1)][3]))
    selGM.append(np.median(accuracies[str(1)][4]))
    selIOU.append(np.median(accuracies[str(1)][5]))

    selSen2.append(np.median(accuracies[str(2)][0]))
    selSpec2.append(np.median(accuracies[str(2)][1]))
    selPixs2.append(np.median(pixAs))
    selDic2.append(np.median(accuracies[str(2)][3]))
    selGM2.append(np.median(accuracies[str(2)][4]))
    selIOU2.append(np.median(accuracies[str(2)][5]))

    selSen3.append(np.median(accuracies[str(3)][0]))
    selSpec3.append(np.median(accuracies[str(3)][1]))
    selPixs3.append(np.median(pixAs))
    selDic3.append(np.median(accuracies[str(3)][3]))
    selGM3.append(np.median(accuracies[str(3)][4]))
    selIOU3.append(np.median(accuracies[str(3)][5]))

    # Create 3 subplots
    query = accuracies[str(1)][5]
    qInx = query.index(max(query))
    val_dset_s = datasetCT(val_img_fold[img_idx[qInx]:(img_idx[qInx]+1),...],
                      val_lbl_fold[img_idx[qInx]:(img_idx[qInx]+1) ,...], None) 
    
    val_loader_s = torch.utils.data.DataLoader(
        val_dset_s, batch_size=1, shuffle=False)
    
    with torch.no_grad():
      for data, target in val_loader_s:
        data = Variable(data.cuda())
        target = Variable(target.cuda())
        output = model(data)
        # torch >=0.5
        pred = train_utils.get_predictions(output)
        p = np.squeeze(pred.detach().cpu().numpy())
        t = np.squeeze(target.detach().cpu().numpy())
        print(np.unique(t))
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(np.multiply(np.squeeze(scaleCT(val_img_fold[img_idx[qInx],])), (-1)*(np.squeeze(val_lbl_fold[img_idx[qInx],:,:,3]) - 1)), cmap="gray")
        ax1.axis('off')
        ax2.imshow(t, cmap = "coolwarm")
        ax2.axis('off')
        ax3.imshow(p, cmap = "coolwarm")
        ax3.axis('off')
        names = ["background", "GGO", "consolidations", "other lungs"]  
        getLegend = lambda i: ax3.plot([],color = mcp.gen_color(cmap="coolwarm",n=num_class)[i],
                          label=names[i], ls="", marker="o")[0]
        ax3.axis('off')
        ax3.legend(handles=[getLegend(i) for i in range(num_class)],  fontsize = "xx-small", bbox_to_anchor=(1, 1.3))
        if options.model:
          plt.savefig("./figures/best result_"+str(idx)+"_" +str(options.numBlock) +"_fold_"+"FCdenseNet_4-class"+".pdf", dpi=600)
        else:
          plt.savefig("./figures/best result_"+str(idx)+"_" +str(options.numBlock)+"_fold_"+"FCdenseNet_V2_4-class"+".pdf", dpi=600)
  import pandas as pd
  d = {'Sensitivy': selSen, 'Specificity': selSpec, 'Dice': selDic, 'selGM': selGM, 'selIOU': selIOU, 'Pixel Accuracy': selPixs}
  df = pd.DataFrame(data=d)
  print("Dice:")
  print(selDic)
  print("IoU:")
  print(selIOU)
  df.round(4)
  if options.model:
    df.to_csv("./results/"+str(options.numBlock)+"_densetNet multi class 1.csv",index=False)
  else:
    df.to_csv("./results/"+str(options.numBlock)+"_densetNet V2 multi class 1.csv",index=False)      
  d2 = {'Sensitivy': selSen2, 'Specificity': selSpec2, 'Dice': selDic2, 'selGM': selGM2, 'selIOU': selIOU2, 'Pixel Accuracy': selPixs2}
  df2 = pd.DataFrame(data=d2)
  df2.round(4)
  if options.model:
    df2.to_csv("./results/"+str(options.numBlock)+"_densetNet multi class 2.csv",index=False)
  else:
    df2.to_csv("./results/"+str(options.numBlock)+"_densetNet V2 multi class 2.csv",index=False)
  d3 = {'Sensitivy': selSen3, 'Specificity': selSpec3, 'Dice': selDic3, 'selGM': selGM3, 'selIOU': selIOU3, 'Pixel Accuracy': selPixs3}
  df3 = pd.DataFrame(data=d3)
  df3.round(4)
  if options.model:
    df3.to_csv("./results/"+str(options.numBlock)+"_densetNet multi class 3.csv",index=False)
  else:
    df3.to_csv("./results/"+str(options.numBlock)+"_densetNet V2 multi class 3.csv",index=False)
