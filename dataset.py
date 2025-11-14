import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import torch.nn.functional as F


MONAI_AVAILABLE = False  # Disable MONAI 


class ElasticTransform:
    """Elastic deformation """
    def __init__(self, alpha=50, sigma=5):
        self.alpha = alpha
        self.sigma = sigma
        self._has_scipy = False
        try:
            import scipy.ndimage
            import scipy.interpolate
            self._has_scipy = True
        except ImportError:
            pass
    
    def __call__(self, img):
        if not self._has_scipy:
            return img  
        
        try:
            from scipy.ndimage import gaussian_filter
            from scipy.interpolate import griddata
            
            img_array = np.array(img)
            shape = img_array.shape[:2]
            
            dx = np.random.randn(*shape) * self.alpha
            dy = np.random.randn(*shape) * self.alpha
            
            # Smooth the displacement
            dx = gaussian_filter(dx, self.sigma)
            dy = gaussian_filter(dy, self.sigma)
            
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            x_new = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
            y_new = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)
            
            
            points = np.column_stack([y.flatten(), x.flatten()])
            new_points = np.column_stack([y_new.flatten(), x_new.flatten()])
            
            if len(img_array.shape) == 3:
                transformed = np.zeros_like(img_array)
                for c in range(img_array.shape[2]):
                    values = img_array[:, :, c].flatten()
                    transformed[:, :, c] = griddata(
                        points, values, new_points, method='linear', fill_value=0
                    ).reshape(shape)
            else:
                values = img_array.flatten()
                transformed = griddata(
                    points, values, new_points, method='linear', fill_value=0
                ).reshape(shape)
            
            return Image.fromarray(transformed.astype(np.uint8))
        except:
            return img  


class SegmentationDataset(Dataset):
    def __init__(self, df, images_dir, masks_dir, img_size=512, augment=False, aug_config=None):
        self.df = df if isinstance(df, pd.DataFrame) else pd.read_csv(df)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.augment = augment
        self.aug_config = aug_config or {}
        
        # Base transforms for grayscale images
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1] range 
        ])
        
     
        self.img_size = img_size
        
    
    def _apply_augmentation(self, img, mask):
        """Apply ultrasound-specific augmentations"""
        if not self.augment:
            return img, mask
        
        #  horizontal flp
        if random.random() < self.aug_config.get('horizontal_flip', 0.5):
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        
        #  vertical 
        if random.random() < self.aug_config.get('vertical_flip', 0.5):
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        
        #  rotation 
        if 'rotation_degrees' in self.aug_config:
            angle = random.uniform(-self.aug_config['rotation_degrees'], 
                                  self.aug_config['rotation_degrees'])
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)
        
        # Brightness 
        if 'brightness_range' in self.aug_config:
            brightness_factor = random.uniform(*self.aug_config['brightness_range'])
            img = TF.adjust_brightness(img, brightness_factor)
        
        # Contrast 
        if 'contrast_range' in self.aug_config:
            contrast_factor = random.uniform(*self.aug_config['contrast_range'])
            img = TF.adjust_contrast(img, contrast_factor)
        
        # Elastic deformation
        if self.aug_config.get('elastic_alpha', 0) > 0 and random.random() < 0.3:
            elastic = ElasticTransform(
                alpha=self.aug_config.get('elastic_alpha', 50),
                sigma=self.aug_config.get('elastic_sigma', 5)
            )
            img = elastic(img)
            mask = elastic(mask)
        
        return img, mask
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['ImageId']
        mask_name = row['MaskId']
        
        # load data
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        img_array = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
        mask_array = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
        
        # Convert mask to binary (0 or 1)
        mask_binary = (mask_array > 0).astype(np.float32)
        
        # normalize 
        img_array = img_array / 255.0
        
        # perform augs
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
        mask_pil = Image.fromarray((mask_binary * 255).astype(np.uint8), mode='L')
        img_pil, mask_pil = self._apply_augmentation(img_pil, mask_pil)
        
      
        img_array = np.array(img_pil, dtype=np.float32) / 255.0
        mask_binary = (np.array(mask_pil, dtype=np.float32) > 127.5).astype(np.float32)
        
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # [1, H, W]
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # [1, H, W]
        
        # Resize 
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),  # [1, 1, H, W]
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # [1, H, W]
        
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0),  # [1, 1, H, W]
            size=(self.img_size, self.img_size),
            mode='nearest'
        ).squeeze(0)  # [1, H, W]
        
        # Normalize 
        img_tensor = (img_tensor - 0.5) / 0.5
        
        # binarize mask 
        mask_tensor = (mask_tensor > 0.5).float()
        
      
        mask_tensor = mask_tensor.squeeze(0)  # [H, W]
        
        
        return img_tensor, mask_tensor
