# data/aligned_adv_dataset.py
import os
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torchvision.datasets as datasets # <-- Import datasets

class AlignedAdvDataset(BaseDataset):
    """
    This dataset class loads paired images from two directories using torchvision.datasets.ImageFolder.
    It is designed for the adversarial -> clean image task and assumes an ImageNet-style folder structure.
    It requires that the file structures (subdirectories and filenames) of the two directories are identical.

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
        self.dir_A = opt.dataroot_A  # path for domain A (adversarial)
        self.dir_B = opt.dataroot_B  # path for domain B (clean)

        print(f"Loading images from domain A: {self.dir_A}")
        print(f"Loading images from domain B: {self.dir_B}")

        # --- Use the robust ImageFolder to find all images ---
        dataset_A = datasets.ImageFolder(self.dir_A)
        dataset_B = datasets.ImageFolder(self.dir_B)

        # Extract the file paths from the ImageFolder object
        # The .imgs attribute is a list of (path, class_index) tuples. We only need the path.
        self.A_paths = [path for path, _ in dataset_A.imgs]
        self.B_paths = [path for path, _ in dataset_B.imgs]

        # --- Critical Sanity Check ---
        # Ensure both datasets have the same number of images. Since ImageFolder sorts
        # subdirectories and files alphabetically, the lists should be perfectly aligned.
        assert len(self.A_paths) == len(self.B_paths), \
            f"Mismatch in dataset size. Dataset A has {len(self.A_paths)} images, " \
            f"but Dataset B has {len(self.B_paths)} images. Please check the directories."

        self.A_size = len(self.A_paths)
        if self.A_size == 0:
            raise(RuntimeError(f"Found 0 images in: {self.dir_A}. Please check the path and data format."))

        self.transform = get_transform(self.opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths, B_paths
            A (tensor) - - an image in the input domain (adversarial)
            B (tensor) - - its corresponding image in the target domain (clean)
            A_paths (str) - - image paths
            B_paths (str) - - image paths
        """
        # Get the aligned paths from our pre-compiled lists
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.A_size]
        
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # Apply the same transform to both images for alignment
        A = self.transform(A_img)
        B = self.transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size