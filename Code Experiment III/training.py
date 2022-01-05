import argparse
import os
import random
import shutil
import time
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from glob import glob
import numpy as np
import cv2 as cv
import os
import timm
from sklearn.metrics import roc_auc_score, f1_score,accuracy_score,cohen_kappa_score

parser = argparse.ArgumentParser(description='PyTorch Pathology Training')

parser.add_argument('--data',default='Patch_dataset/Training_dataset', type=str,  metavar='PATH')
parser.add_argument('-a', '--arch', metavar='ARCH', default='inception_resnet_v2',
                    help='model architecture name' + ' (default: inception_resnet_v2)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--resume', default='Training', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--numlayer',default=16, type=int, metavar='N')
parser.add_argument('--pretrained', dest='pretrained', default=False,
                    help='use pre-trained model')

def count_dataset(list_dataset_path):  
  label_data=[]
  list_benign_200x200 = []
  list_malignant_200x200 = []
  for i in all_dataset :
    if i.find("malignant200x200") != -1:
      list_malignant_200x200.append(i)
      label_data.append(1)
    else:
      list_benign_200x200.append(i)
      label_data.append(0)
  print(f"benign200x200 total file : {str(len(list_benign_200x200))}")
  print(f"malignant200x200 total file : {str(len(list_malignant_200x200))}")
  return list_benign_200x200,list_malignant_200x200,np.array(label_data)

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

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def save_checkpoint(state, is_best,path,filename='checkpoint.pth.tar'):
    torch.save(state, f"{path}/{filename}")
    if is_best:
        shutil.copyfile(f"{path}/{filename}", f"{path}/model_best.pth.tar")

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # rocauc = AverageMeter('rocauc', ':6.2f')
    f1score = AverageMeter('f1score', ':6.2f')
    accuracyscore = AverageMeter('accuracyscore', ':6.2f')
    cohenkappascore = AverageMeter('cohenkappascore', ':6.2f')


    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses,  f1score,accuracyscore,cohenkappascore],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        data, target = torch.tensor(data).cuda().float(), torch.tensor(target).cuda()

        # compute output
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        y_actual = target.data.cpu().numpy()
        y_pred = torch.argmax(output,dim=1).detach().cpu().numpy()
        # rocauc.update(roc_auc_score(y_actual, y_pred.round()), data.size(0))
        f1score.update(f1_score(y_actual, y_pred.round(),average='micro'), data.size(0))
        accuracyscore.update(accuracy_score(y_actual, y_pred.round()), data.size(0))
        cohenkappascore.update(cohen_kappa_score(y_actual, y_pred.round()), data.size(0))
        losses.update(loss.item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(i)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    # rocauc = AverageMeter('rocauc', ':6.2f', Summary.AVERAGE)
    f1score = AverageMeter('f1score', ':6.2f', Summary.AVERAGE)
    accuracyscore = AverageMeter('accuracyscore', ':6.2f', Summary.AVERAGE)
    cohenkappascore = AverageMeter('cohenkappascore', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses,  f1score,accuracyscore,cohenkappascore],
        prefix='Test: ')
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):

            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target,)

            # measure accuracy and record loss
            y_actual = target.data.cpu().numpy()
            y_pred = torch.argmax(output,dim=1).detach().cpu().numpy()
            # rocauc.update(roc_auc_score(y_actual, y_pred.round()), data.size(0))
            f1score.update(f1_score(y_actual, y_pred.round(),average='micro'), data.size(0))
            accuracyscore.update(accuracy_score(y_actual, y_pred.round()), data.size(0))
            cohenkappascore.update(cohen_kappa_score(y_actual, y_pred.round()), data.size(0))
            losses.update(loss.item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i)

        progress.display_summary()

    return losses.avg

def freeze_by_layer_child(num_max_layer_freeze,model):
  count = 0
  for child in model.children():
    count+=1
    if count <num_max_layer_freeze: # classifier on child layers 6
      for param in child.parameters():
            param.requires_grad = False
  return model
  
def check_layer_requires_grad(nlayers,model):
    for i in list(model.children())[nlayers].parameters():
      print(i.requires_grad  )
      
def main_worker(training_list_data,test_list_data,resume,batch_size,numlayer,class_weights=None,name="seresnet18",pretrain=False,num_classes=2,start_epoch=0,epochs=50):
    global best_val_loss
   
    train_transforms = transforms.Compose([
          transforms.ToPILImage(),                                   
          transforms.RandomVerticalFlip(),
          transforms.RandomHorizontalFlip(),
          transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=.5),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
    val_transform = transforms.Compose([
          transforms.ToPILImage(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
    training_dataset= Dataset(training_list_data,train_transforms)
    validation_dataset= Dataset(test_list_data,val_transform)

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    model = timm.create_model(name, pretrained=pretrain, num_classes=num_classes)
    if pretrain:
      model= freeze_by_layer_child(numlayer,model)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float()).cuda()

    optimizer = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =407,last_epoch=-1)
    # optionally resume from a checkpoint
    agg_time_train= []
    agg_time_val = []
    if resume:
        if os.path.isfile(f"{resume}/checkpoint.pth.tar"):
            print("=> loading checkpoint '{}'".format(f"{resume}/checkpoint.pth.tar"))
            checkpoint = torch.load(f"{resume}/checkpoint.pth.tar")
            agg_time_train = checkpoint['agg_time_train']
            agg_time_val = checkpoint['agg_time_val']
            start_epoch = checkpoint['epoch']
            # best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    cudnn.benchmark = True

    
    for epoch in range(start_epoch, epochs):
        
        # train for one epoch
        end = time.time()
        train(train_loader, model, criterion, optimizer, epoch)
        agg_time_train.append(time.time()-end)
        lr_scheduler.step()
        # evaluate on validation set
        end = time.time()
        val_loss = validate(val_loader, model, criterion)
        agg_time_val.append(time.time()-end)

        is_best = val_loss <best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizer' : optimizer.state_dict(),
            'lr_scheduler' : lr_scheduler.state_dict(),
            'agg_time_train' :agg_time_train,
            'agg_time_val':agg_time_val
            }, 
            is_best,
            path=resume)
        
if __name__ == '__main__':
    args = parser.parse_args()
    # print(args)
    print(f"training model {args.arch} with batch size {args.batch_size}")
    best_val_loss = 10
    all_dataset= glob(f"{args.data}/**/***.bmp")
    list_benign_200x200,list_malignant_200x200,_ = count_dataset(all_dataset)
    x_80_malignant_200x200, x_20_malignant_200x200 =  train_test_split(list_malignant_200x200, test_size =.20, shuffle  = True)
    x_80_benign_200x200, x_20_benign_200x200 =  train_test_split(list_benign_200x200, test_size =.20, shuffle  = True)
    training_list_data = x_80_malignant_200x200+x_80_benign_200x200
    test_list_data = x_20_malignant_200x200+x_20_benign_200x200

    np.random.seed(304)
    random.seed(304)
    torch.manual_seed(304)
    random.shuffle(training_list_data)
    random.shuffle(test_list_data)
    _,_,y_train = count_dataset(training_list_data)
    class_weights = class_weight.compute_class_weight('balanced',
                                                    classes = np.unique(np.unique(np.array(y_train))),
                                                    y= np.array(y_train))

    cudnn.deterministic = True
    resume=f"{args.resume}/f{args.arch}"

    os.makedirs(resume,exist_ok=True)

    main_worker(training_list_data=training_list_data,
                test_list_data=test_list_data,
                resume=resume,
                class_weights=class_weights,
                num_classes=2,
                name=args.arch,
                batch_size=args.batch_size,
                numlayer=args.numlayer,
                pretrain=args.pretrained,
                start_epoch=args.start_epoch,
                epochs=args.epochs)
  
