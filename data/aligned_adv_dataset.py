# data/aligned_adv_dataset.py
import os
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random

class AlignedAdvDataset(BaseDataset):
    """
    This dataset class loads paired images from two directories.
    It is designed for the adversarial -> clean image task and assumes an ImageNet-style folder structure.
    It requires that the file structures (subdirectories and filenames) of the two directories are IDENTICAL.

    This version uses a robust pairing method and ensures that random augmentations (crop, flip)
    are applied identically to both images in a pair to maintain alignment.

    It will be hosted at '/data/aligned_adv_dataset.py'.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options."""
        parser.add_argument('--dataroot_A', required=True, help='path to images in domain A (adversarial images)')
        parser.add_argument('--dataroot_B', required=True, help='path to images in domain B (clean images)')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.abspath(opt.dataroot_A)
        self.dir_B = os.path.abspath(opt.dataroot_B)

        print(f"Scanning for images in source domain A: {self.dir_A}")
        dataset_A = datasets.ImageFolder(self.dir_A)
        self.A_paths = sorted([path for path, _ in dataset_A.imgs])

        if not self.A_paths:
            raise(RuntimeError(f"Found 0 images in: {self.dir_A}. Please check the path and data format."))

        print(f"Deriving corresponding paths for target domain B: {self.dir_B}")
        self.B_paths = [os.path.join(self.dir_B, os.path.relpath(p, self.dir_A)) for p in self.A_paths]

        if not os.path.exists(self.B_paths[0]):
            raise FileNotFoundError(
                f"\nCRITICAL ERROR: Dataset mismatch detected!\n"
                f"Path in domain A: '{self.A_paths[0]}'\n"
                f"Derived path in B does not exist: '{self.B_paths[0]}'\n\n"
                f"Please ensure directory structures and filenames are IDENTICAL."
            )

        self.A_size = len(self.A_paths)
        print(f"Successfully paired {self.A_size} images.")

        # We will not use the generic get_transform directly.
        # Instead, we will construct the logic in __getitem__ to ensure sync.

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        This version implements synchronized random augmentations.
        """
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.A_size]
        
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # --- 1. Apply synchronized transformations ---
        
        # Resize
        A_img = A_img.resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC)
        B_img = B_img.resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC)

        # Random crop (get parameters once, apply to both)
        if self.opt.isTrain:
            i, j, h, w = transforms.RandomCrop.get_params(
                A_img, output_size=(self.opt.crop_size, self.opt.crop_size))
            A_img = F.crop(A_img, i, j, h, w)
            B_img = F.crop(B_img, i, j, h, w)

            # Random horizontal flip (make one decision, apply to both)
            if random.random() > 0.5:
                A_img = F.hflip(A_img)
                B_img = F.hflip(B_img)
        
        # --- 2. Convert to tensor and normalize ---
        
        # ToTensor
        A = F.to_tensor(A_img)
        B = F.to_tensor(B_img)
        
        # Normalize
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        A = normalize(A)
        B = normalize(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size