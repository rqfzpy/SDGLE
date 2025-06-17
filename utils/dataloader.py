import os
from colorama import Fore, Style
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

csv_file = '/mnt/data/meddata/APTOS/train.csv'
img_dir = '/mnt/data/meddata/APTOS/train_images'

def datainfo(args):
    if args.dataset == 'CIFAR10':
        # print(Fore.YELLOW+'*'*80)
        # print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_mean, img_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        img_size = 32        
        
    elif args.dataset == 'CIFAR100':
        # print(Fore.YELLOW+'*'*80)
        # print('*'*80 + Style.RESET_ALL)
        n_classes = 100
        img_mean, img_std = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762) 
        img_size = 32        
        
    elif args.dataset == 'SVHN':
        # print(Fore.YELLOW+'*'*80)
        # print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_mean, img_std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970) 
        img_size = 32
        
    elif args.dataset == 'T-IMNET':
        # print(Fore.YELLOW+'*'*80)
        # print('*'*80 + Style.RESET_ALL)
        n_classes = 200
        img_mean, img_std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        img_size = 64
    elif args.dataset == 'IMNET':
        # print(Fore.YELLOW+'*'*80)
        # print('*'*80 + Style.RESET_ALL)
        n_classes = 1000
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 224
    elif args.dataset == 'FL102':
        # print(Fore.YELLOW+'*'*80)
        # print('*'*80 + Style.RESET_ALL)
        n_classes = 102
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size = 256
    elif args.dataset == 'APTOS':
        # print(Fore.YELLOW+'*'*80)
        # print('*'*80 + Style.RESET_ALL)
        n_classes = 5
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size =256
    elif args.dataset == 'IDRID':
        # print(Fore.YELLOW+'*'*80)
        # print('*'*80 + Style.RESET_ALL)
        n_classes = 3
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size =256
    elif args.dataset == 'ISIC':
        # print(Fore.YELLOW+'*'*80)
        # print('*'*80 + Style.RESET_ALL)
        n_classes = 3
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_size =256
    data_info = dict()
    data_info['n_classes'] = n_classes
    data_info['stat'] = (img_mean, img_std)
    data_info['img_size'] = img_size
    
    return data_info

def dataload(args, augmentations, normalize, data_info):
    if args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=augmentations)
        val_dataset = datasets.CIFAR10(
            root=args.data_path, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(data_info['img_size']),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'CIFAR100':

        train_dataset = datasets.CIFAR100(
            root=args.data_path, train=True, download=True, transform=augmentations)
        val_dataset = datasets.CIFAR100(
            root=args.data_path, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(data_info['img_size']),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'SVHN':

        train_dataset = datasets.SVHN(
            root=args.data_path, split='train', download=True, transform=augmentations)
        val_dataset = datasets.SVHN(
            root=args.data_path, split='test', download=True, transform=transforms.Compose([
            transforms.Resize(data_info['img_size']),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'T-IMNET':
        train_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'tiny_imagenet', 'train'), transform=augmentations)
        val_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'tiny_imagenet', 'val','images'), 
            transform=transforms.Compose([
            transforms.Resize(data_info['img_size']), transforms.ToTensor(), *normalize]))

    elif args.dataset == 'IMNET':
        train_dataset= datasets.ImageFolder(
            '/mnt/data/imagenet2012/train',
            transform=augmentations,)
        val_dataset = datasets.ImageFolder(
            '/mnt/data/imagenet2012/val',
            transform=transforms.Compose([
            transforms.Resize(data_info['img_size']), transforms.ToTensor(), *normalize]))
    elif args.dataset == 'FL102':
        train_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'fl102/prepare_pic', 'train'), transform=augmentations)
        val_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, 'fl102/prepare_pic', 'test'), 
            transform=transforms.Compose([
            transforms.Resize(data_info['img_size']), transforms.ToTensor(), *normalize]))
    elif args.dataset == 'IDRID':
        train_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, '/mnt/data/meddata/IDRID/data_cop/Images', 'train'), transform=augmentations)
        val_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, '/mnt/data/meddata/IDRID/data_cop/Images', 'test'), 
            transform=transforms.Compose([
            transforms.Resize(data_info['img_size']), transforms.ToTensor(), *normalize]))
    elif args.dataset == 'ISIC':
        train_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, '/mnt/data/meddata/ISIC', 'train'), transform=augmentations)
        val_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_path, '/mnt/data/meddata/ISIC', 'test'), 
            transform=transforms.Compose([
            transforms.Resize(data_info['img_size']), transforms.ToTensor(), *normalize]))
    elif args.dataset == 'APTOS':
        train_dataset = RetinopathyDataset(csv_file=csv_file, img_dir=img_dir, transform=augmentations, train=True)
        val_dataset = RetinopathyDataset(csv_file=csv_file, img_dir=img_dir, transform=transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]), train=False)
    return train_dataset, val_dataset


# 自定义数据集类
class CIFARCDataset(Dataset):
    def __init__(self, data_dir, corruption_type, transform=None):
        self.data = np.load(os.path.join(data_dir, corruption_type + '.npy'))
        self.labels = np.load(os.path.join(data_dir, 'labels.npy'))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

    
import numpy as np
import os
import glob2 as glob
from torch.utils.data import Dataset
from PIL import Image
import cv2
import torchvision
import torch

CLASS_LIST_FILE = 'wnids.txt'
EXTENSION = 'JPEG'

class CorruptTiny(Dataset):

    def __init__(self, root, severity,  corrupt, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.corrupt = corrupt
        self.severity = severity
        self.transform = transform
        self.target_transform = target_transform
        self.corrupt_dir = os.path.join(self.root, self.corrupt, str(self.severity))

        self.image_paths = sorted(glob.iglob(os.path.join(self.corrupt_dir, '**', '*.%s' % EXTENSION), recursive=True))

        self.labels = {}

        with open(os.path.join(self.root, CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        for text, i in self.label_text_to_number.items():
            for _, _, filenames in os.walk(os.path.join(self.corrupt_dir, text)):
                for f_name in filenames:
                   self.labels[f_name] = i

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]
        img = cv2.imread(file_path)
        img = torchvision.transforms.functional.to_pil_image(img)
        img = self.transform(img)

        return img, self.labels[os.path.basename(file_path)]

class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, train=True, val_split=0.2):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.train = train
        self.val_split = val_split
        
        if self.train:
            self.data = self.data.iloc[:int(len(self.data) * (1 - self.val_split))]
        else:
            self.data = self.data.iloc[int(len(self.data) * (1 - self.val_split)):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0] + '.png')
        image = Image.open(img_name)
        label = torch.tensor(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label
