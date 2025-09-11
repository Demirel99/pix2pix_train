# data/base_dataset.py
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

class BaseDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise NotImplementedError


def get_transform(opt, grayscale=False, method=Image.BICUBIC):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    
    osize = [opt.load_size, opt.load_size]
    transform_list.append(transforms.Resize(osize, method))
    
    if opt.isTrain:
        transform_list.append(transforms.RandomCrop(opt.crop_size))
        transform_list.append(transforms.RandomHorizontalFlip())
    
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)