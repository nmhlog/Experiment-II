import argparse

from enum import Enum
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from glob import glob
import numpy as np
import cv2 as cv
import os
import timm
from sklearn.metrics import roc_auc_score, f1_score,accuracy_score,cohen_kappa_score
import argparse

from tqdm.std import tqdm

parser = argparse.ArgumentParser(description='PyTorch Pathology Training')
parser.add_argument('--data',default='Patch_dataset/Training_dataset ', type=str,  metavar='DIR')
parser.add_argument('-a', '--arch', metavar='ARCH', default='inception_resnet_v2',
                    help='model architecture name' + ' (default: inception_resnet_v2)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--numlayer',default=32, type=int, metavar='N')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
class Dataset(torch.utils.data.Dataset):
    def __init__(self, path_data,transforms =None):
        super(Dataset, self).__init__()
        self.transform = transforms
        self.path_data = path_data

    def get_label(self,path):
        if path.find("malignant200x200")!=-1:
          return 1
        else :
          return 0
        
    def __getitem__(self, idx):
        img = cv.imread(self.path_data[idx],cv.COLOR_BGR2RGB)
        label = self.get_label(self.path_data[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img,int(label)


    def __len__(self):
        return len(self.path_data)

if "__init__"=="__main__":
    args = parser.parse_args()
    resume=f"{args.resume}/f{args.arch}"
    all_dataset= f"{args.data}/**/**.bmp"
    val_transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])
    test_dataset= Dataset(all_dataset,val_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    best_model = torch.load(f"{resume}/model_best.pth.tar")
    model = timm.create_model(args.arch, pretrained=False, num_classes=2)
    model.load_state_dict(best_model['state_dict'])
    model = model.cuda()

    predicted = []
    Y_true = []
    for data,label in tqdm(test_loader,total=len(test_loader)):
        with torch.no_grad():
            model.eval()
            output_ = model(data.cuda())
            predicted.append(output_.argmax().item())
            Y_true.append(label.unsqueeze_(0).item())
    roc_auc_score_test = roc_auc_score(Y_true, predicted)
    f1_score_test = f1_score(Y_true, predicted,average='micro')
    accuracy_score_test = accuracy_score(Y_true, predicted)
    cohen_kappa_score_test = cohen_kappa_score(Y_true, predicted)

    print(f"roc_auc_score_test : {roc_auc_score_test}")
    print(f"f1_score_test : {f1_score_test}")
    print(f"accuracy_score_test : {accuracy_score_test}")
    print(f"cohen_kappa_score_test : {cohen_kappa_score_test}")

    torch.save(
    {"roc_auc_score_test" : roc_auc_score_test,
    "f1_score_test" : f1_score_test,
    'accuracy_score_test' : accuracy_score_test,
    'cohen_kappa_score_test' : cohen_kappa_score_test,},resume+"hasilprediksi.pth")

