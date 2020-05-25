######################### Import des libraries #######################

from sklearn.model_selection import train_test_split
from pyrsgis import raster
from torch.utils.data import Dataset, DataLoader, sampler
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import time

######################### Fonctions auxiliaires #########################

def apply_model_to_image(model, img, size):
    new_image = np.zeros((img.shape[1], img.shape[2]))
    with torch.no_grad():
        for i in range(int(img.shape[1] * img.shape[2] // size**2)):
            input = img[:11,((i * size) % img.shape[1]):(((i+1) * size) % img.shape[1]), (int(i * size / img.shape[1])*size):((int(i * size / img.shape[1]) + 1)*size)]
            if input.shape[1]*input.shape[2] != 0:
                output = torch.reshape(model(torch.reshape(torch.tensor([input]), (1, size, input.shape[2]*input.shape[1] // size, 11))), (input.shape[1], input.shape[2],3)).numpy()
                new_image[((i * size) % img.shape[1]):(((i+1) * size) % img.shape[1]), (int(i * size / img.shape[1])*size):((int(i * size / img.shape[1]) + 1)*size)] = np.argmax(output,2)*122
    return Image.fromarray(np.uint8(new_image))

def convert_image_msi(img):
    return Image.fromarray(np.uint8(img[10]*255))

def convert_image_rgb(img):
    new_image = np.zeros((img.shape[1], img.shape[2],3))
    for i in range(int(img.shape[1])):
        for j in range(int(img.shape[2])):
            new_image[i,j,0] = int(img[7,i,j]*255)
            new_image[i,j,1] = int(img[9,i,j]*255)
            new_image[i,j,2] = int(img[3,i,j]*255)
    return Image.fromarray(np.uint8(new_image), 'RGB')

######################### Modèle UNET #########################

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
      return getattr(nn, activation_type)()
    else:
      return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
  layers = []
  layers.append(ConvBatchNorm(in_channels, out_channels, activation))

  for _ in range(nb_Conv-1):
      layers.append(ConvBatchNorm(out_channels, out_channels, activation))
  return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
  """(convolution => [BN] => ReLU)"""

  def __init__(self, in_channels, out_channels, activation='ReLU'):
    super(ConvBatchNorm, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, padding=1)
    self.norm = nn.BatchNorm2d(out_channels)
    self.dropout = nn.Dropout(p=0.5)
    self.activation = get_activation(activation)

  def forward(self, x):
    out = self.conv(x)
    out = self.norm(out)
    out = self.dropout(out)
    return self.activation(out)

class DownBlock(nn.Module):
  """Downscaling with maxpool convolution"""

  def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
    super(DownBlock, self).__init__()
    self.maxpool = nn.MaxPool2d(2)
    self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

  def forward(self, x):
    out = self.maxpool(x)
    return self.nConvs(out)

class UpBlock(nn.Module):
  """Upscaling then conv"""

  def __init__(self, in_channels, out_channels, nb_Conv, in_padding=0, out_padding=1, activation='ReLU'):
    super(UpBlock, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2,\
                                 padding=in_padding, output_padding=out_padding)
    self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

  def forward(self, x, skip_x):
    out = self.up(x)
    x = torch.cat([out, skip_x], dim=1)
    return self.nConvs(x)

class UNet(nn.Module):
  def __init__(self, n_channels, n_classes, batch):
    '''
    n_channels : number of channels of the input.
                    By default 4, because we have 4 modalities
    n_labels : number of channels of the ouput.
                  By default 4 (3 labels + 1 for the background)
    '''
    super(UNet, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.inc = ConvBatchNorm(n_channels, 64)
    self.down1 = DownBlock(64, 128, nb_Conv=2)
    self.down2 = DownBlock(128, 256, nb_Conv=2)
    self.down3 = DownBlock(256, 512, nb_Conv=2)
    self.up1 = UpBlock(512+256, 256, nb_Conv=2, out_padding=0)
    self.up2 = UpBlock(256+128, 128, nb_Conv=2, out_padding = (0,1))
    self.up3 = UpBlock(128+64, 64, nb_Conv=2, out_padding=(0,1))
    self.outd = nn.Conv2d(64, n_channels, kernel_size=3, stride=1, padding=1)
    self.outc = nn.Linear(11, n_classes)
    self.last_activation = get_activation('Sigmoid')
    self.batch = batch

  def forward(self, x):
    x1 = self.inc(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x = self.up1(x4, x3)
    x = self.up2(x, x2)
    x = self.up3(x, x1)
    x = self.outd(x)
    logits = self.last_activation(self.outc(x))
    return logits

def predict(file_path):
    ds1, img =  raster.read(file_path, bands='all')
    size = 64
    batch_size = 1
    model = UNet(n_channels=64, n_classes=3, batch=batch_size)
    model.load_state_dict(torch.load("flask_server/static/unet"))
    mask = apply_model_to_image(model, img, size)
    msi = convert_image_msi(img)
    rgb = convert_image_rgb(img)

    return mask, msi, rgb
