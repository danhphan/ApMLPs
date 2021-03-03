import torch
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClassificationDataset:
    def __init__(self, image_paths, targets, resize=None, augmentations=None):
        """
        :param image_paths: list of path to images
        :param targets: numpy array
        :param resize: tuple, e.g. (256, 256) to resize image if not None
        :param augmentations: augmentations
        """
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        """
        For a given "item" index, return needed features for training
        """
        image = Image.open(self.image_paths[item])
        image = image.convert("RGB")
        targets = self.targets[item]
        # Resize if needed
        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)
        image = np.array(image)
        # Augmentation
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        # Pytoch expects CHW instead HWC
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # Return tensors of image and targets
        return {"image": torch.tensor(image, dtype=torch.float), "targets": torch.tensor(targets, dtype=torch.long),}
        
