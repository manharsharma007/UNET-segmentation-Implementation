from PIL import Image
import os
import torch
import numpy as np
from torch.utils.data import Dataset

class CarvanaData(Dataset):

	def __init__(self, image_dir, mask_dir, transforms = None):
		super(CarvanaData, self).__init__()

		self.image_dir = image_dir
		self.mask_dir = mask_dir
		self.transforms = transforms

		self.images = os.listdir(self.image_dir)


	def __len__(self):
		return len(self.images)


	def __getitem__(self, index):

		path = os.path.join(self.image_dir, self.images[index])
		mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))

		image = np.array(Image.open(path).convert("RGB"))
		mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
		mask[mask == 255.0] = 1.0

		for transform in self.transforms:

			augment = transform(image = image, mask = mask)

			image = augment["image"]
			mask = augment["mask"]

		return image, mask