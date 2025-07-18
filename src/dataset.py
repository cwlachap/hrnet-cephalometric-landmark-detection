import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
from typing import Tuple, List, Dict, Any
import yaml
import glob
from pathlib import Path


class CephDataset(Dataset):
    """
    Dataset for cephalometric landmark detection.
    Handles BMP images and TXT landmark files with automatic train/val/test splitting.
    """
    
    def __init__(self, 
                 data_root: str,
                 mode: str = 'train',
                 config: Dict[str, Any] = None,
                 transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_root: Path to ISBI dataset directory (contains RawImage/ and AnnotationsByMD/)
            mode: 'train', 'val', or 'test'
            config: Configuration dictionary
            transform: Augmentation transforms
        """
        self.data_root = Path(data_root)
        self.mode = mode
        self.config = config or {}
        self.transform = transform
        
        # ISBI dataset structure
        self.images_dir1 = self.data_root / "RawImage" / "RawImage" / "TrainingData"
        self.images_dir2 = self.data_root / "RawImage" / "RawImage" / "Test1Data"
        self.images_dir3 = self.data_root / "RawImage" / "RawImage" / "Test2Data"
        self.landmarks_dir = self.data_root / "AnnotationsByMD" / "400_junior"
        self.num_landmarks = self.config.get('NUM_LANDMARKS', 19)
        self.split_ratios = self.config.get('SPLIT_RATIOS', [0.7, 0.15, 0.15])
        self.seed = self.config.get('SEED', 42)
        
        # Image processing parameters
        self.input_size = self.config.get('IMAGE_SIZE', [768, 768])
        self.original_size = self.config.get('ORIGINAL_SIZE', [1935, 2400])
        self.normalize = self.config.get('NORMALIZE', True)
        self.mean = np.array(self.config.get('MEAN', [0.485, 0.456, 0.406]))
        self.std = np.array(self.config.get('STD', [0.229, 0.224, 0.225]))
        
        # Load and split data
        self.file_list = self._load_file_list()
        self._validate_data()
        
        print(f"Loaded {len(self.file_list)} files for {mode} mode")
    
    def _load_file_list(self) -> List[str]:
        """Load and split file list based on mode."""
        # Get all image files from all three directories
        image_files = (
            sorted(glob.glob(str(self.images_dir1 / "*.bmp"))) + 
            sorted(glob.glob(str(self.images_dir2 / "*.bmp"))) + 
            sorted(glob.glob(str(self.images_dir3 / "*.bmp")))
        )
        
        if not image_files:
            raise ValueError(f"No BMP files found in {self.images_dir1}, {self.images_dir2}, or {self.images_dir3}")
        
        # Extract base names (without extension)
        base_names = [Path(f).stem for f in image_files]
        
        # Filter to only include files where we can successfully load exactly 19 landmarks
        valid_base_names = []
        for base_name in base_names:
            landmark_path = self.landmarks_dir / f"{base_name}.txt"
            if landmark_path.exists():
                try:
                    # Use the same logic as _load_landmarks to ensure consistency
                    landmarks = []
                    
                    with open(landmark_path, 'r') as f:
                        lines = f.readlines()
                        
                    # Only process the first 19 lines (landmarks), ignore orthodontic classification
                    for i, line in enumerate(lines[:self.num_landmarks]):
                        line = line.strip()
                        if line:
                            # Handle both comma-separated and space-separated formats
                            if ',' in line:
                                parts = line.split(',')
                            else:
                                parts = line.split()
                            
                            if len(parts) >= 2:
                                x, y = float(parts[0]), float(parts[1])
                                landmarks.append([x, y])
                    
                    # Only include if we can load exactly 19 landmarks
                    if len(landmarks) == self.num_landmarks:
                        valid_base_names.append(base_name)
                        
                except Exception:
                    # Skip files that can't be read or parsed
                    continue
        
        if not valid_base_names:
            raise ValueError(f"No valid image-landmark pairs found with {self.num_landmarks} readable landmarks")
        
        print(f"Filtered to {len(valid_base_names)} files with {self.num_landmarks} valid landmarks from {len(base_names)} total files")
        base_names = valid_base_names
        
        # Split data
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # First split: train + val, test
        train_val, test = train_test_split(
            base_names, 
            test_size=self.split_ratios[2], 
            random_state=self.seed
        )
        
        # Second split: train, val
        val_ratio = self.split_ratios[1] / (self.split_ratios[0] + self.split_ratios[1])
        train, val = train_test_split(
            train_val, 
            test_size=val_ratio, 
            random_state=self.seed
        )
        
        # Return appropriate split
        if self.mode == 'train':
            return train
        elif self.mode == 'val':
            return val
        elif self.mode == 'test':
            return test
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
    
    def _validate_data(self):
        """Validate that all required files exist."""
        missing_images = []
        missing_landmarks = []
        
        for base_name in self.file_list:
            # Check if image exists in any of the three directories
            img_path1 = self.images_dir1 / f"{base_name}.bmp"
            img_path2 = self.images_dir2 / f"{base_name}.bmp"
            img_path3 = self.images_dir3 / f"{base_name}.bmp"
            landmark_path = self.landmarks_dir / f"{base_name}.txt"
            
            if not (img_path1.exists() or img_path2.exists() or img_path3.exists()):
                missing_images.append(f"{base_name}.bmp (not found in TrainingData, Test1Data, or Test2Data)")
            if not landmark_path.exists():
                missing_landmarks.append(str(landmark_path))
        
        if missing_images:
            raise FileNotFoundError(f"Missing image files: {missing_images}")
        if missing_landmarks:
            raise FileNotFoundError(f"Missing landmark files: {missing_landmarks}")
    
    def _find_image_path(self, base_name: str) -> Path:
        """Find the correct path for an image file (could be in TrainingData, Test1Data, or Test2Data)."""
        path1 = self.images_dir1 / f"{base_name}.bmp"
        path2 = self.images_dir2 / f"{base_name}.bmp"
        path3 = self.images_dir3 / f"{base_name}.bmp"
        
        if path1.exists():
            return path1
        elif path2.exists():
            return path2
        elif path3.exists():
            return path3
        else:
            raise FileNotFoundError(f"Image file not found: {base_name}.bmp in TrainingData, Test1Data, or Test2Data")
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _load_landmarks(self, landmark_path: str) -> np.ndarray:
        """Load landmarks from TXT file (only first 19 lines - ignore orthodontic classification)."""
        landmarks = []
        
        with open(landmark_path, 'r') as f:
            lines = f.readlines()
            
        # Only process the first 19 lines (landmarks), ignore orthodontic classification
        for i, line in enumerate(lines[:self.num_landmarks]):
            line = line.strip()
            if line:
                # Handle both comma-separated and space-separated formats
                if ',' in line:
                    parts = line.split(',')
                else:
                    parts = line.split()
                
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    landmarks.append([x, y])
        
        landmarks = np.array(landmarks, dtype=np.float32)
        
        if len(landmarks) != self.num_landmarks:
            raise ValueError(f"Expected {self.num_landmarks} landmarks, got {len(landmarks)} in {landmark_path}. This file should have been filtered out during data loading.")
        
        return landmarks
    
    def _resize_with_aspect_ratio(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Resize image with aspect ratio preservation (isotropic resize).
        Apply letterbox padding to reach target size.
        """
        h, w = image.shape[:2]
        target_h, target_w = self.input_size
        
        # Calculate scale factor (minimum to fit both dimensions)
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Calculate padding
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        pad_right = target_w - new_w - pad_left
        pad_bottom = target_h - new_h - pad_top
        
        # Apply padding
        padded_image = cv2.copyMakeBorder(
            resized_image, 
            pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )
        
        # Transform landmarks
        transformed_landmarks = landmarks.copy()
        transformed_landmarks[:, 0] = landmarks[:, 0] * scale + pad_left
        transformed_landmarks[:, 1] = landmarks[:, 1] * scale + pad_top
        
        # Store transformation parameters for inverse transform
        transform_params = {
            'scale': scale,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'original_size': (h, w)
        }
        
        return padded_image, transformed_landmarks, transform_params
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0,1] and apply ImageNet normalization."""
        # Convert to [0,1]
        image = image.astype(np.float32) / 255.0
        
        if self.normalize:
            # Apply ImageNet normalization
            image = (image - self.mean) / self.std
        
        return image
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_name = self.file_list[idx]
        
        # Load image and landmarks
        image_path = self._find_image_path(base_name)
        landmark_path = self.landmarks_dir / f"{base_name}.txt"
        
        image = self._load_image(str(image_path))
        landmarks = self._load_landmarks(str(landmark_path))
        
        # Resize and pad
        image, landmarks, transform_params = self._resize_with_aspect_ratio(image, landmarks)
        
        # Apply augmentations if in training mode with safety checks
        if self.mode == 'train' and self.transform:
            # Try augmentation up to 5 times to ensure we keep all landmarks
            original_landmarks = landmarks.copy()
            for attempt in range(5):
                try:
                    augmented = self.transform(image=image, keypoints=landmarks)
                    aug_image = augmented['image']
                    aug_landmarks = np.array(augmented['keypoints'], dtype=np.float32)
                    
                    # Check if we have the correct number of landmarks
                    if len(aug_landmarks) == self.num_landmarks:
                        image = aug_image
                        landmarks = aug_landmarks
                        break
                    else:
                        # If landmarks were dropped, try again
                        if attempt == 4:  # Last attempt
                            # Fall back to no augmentation
                            print(f"Warning: Augmentation dropped landmarks for {base_name}, using original image")
                            landmarks = original_landmarks
                except Exception as e:
                    # If augmentation fails, fall back to no augmentation
                    print(f"Warning: Augmentation failed for {base_name}: {e}")
                    landmarks = original_landmarks
                    break
            else:
                # If all attempts failed, use original
                landmarks = original_landmarks
        
        # Normalize image
        image = self._normalize_image(image)
        
        # Convert to PyTorch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC -> CHW, ensure float32
        landmarks = torch.from_numpy(landmarks).float()  # Ensure float32
        
        return {
            'image': image,
            'landmarks': landmarks,
            'transform_params': transform_params,
            'filename': base_name
        }


def inverse_transform_landmarks(landmarks: np.ndarray, transform_params: Dict[str, float]) -> np.ndarray:
    """
    Transform landmarks back to original image coordinates.
    
    Args:
        landmarks: Landmarks in processed image coordinates
        transform_params: Parameters from the forward transform
    
    Returns:
        Landmarks in original image coordinates
    """
    transformed = landmarks.copy()
    
    # Remove padding
    transformed[:, 0] -= transform_params['pad_left']
    transformed[:, 1] -= transform_params['pad_top']
    
    # Undo scaling
    transformed[:, 0] /= transform_params['scale']
    transformed[:, 1] /= transform_params['scale']
    
    return transformed


def custom_collate_fn(batch):
    """Custom collate function with better error handling for landmark shape mismatches."""
    import torch
    from torch.utils.data._utils.collate import default_collate
    
    # Check landmark shapes before collating
    landmark_shapes = [sample['landmarks'].shape for sample in batch]
    if not all(shape == landmark_shapes[0] for shape in landmark_shapes):
        # Provide detailed error information
        for i, sample in enumerate(batch):
            print(f"Sample {i}: {sample['filename']} - landmarks shape: {sample['landmarks'].shape}")
        raise ValueError(f"Landmark shape mismatch in batch! Expected all samples to have shape {landmark_shapes[0]}, but got different shapes: {set(landmark_shapes)}")
    
    # Check image shapes
    image_shapes = [sample['image'].shape for sample in batch]
    if not all(shape == image_shapes[0] for shape in image_shapes):
        raise ValueError(f"Image shape mismatch in batch! Expected all samples to have shape {image_shapes[0]}, but got different shapes: {set(image_shapes)}")
    
    # Use default collate if all shapes are consistent
    return default_collate(batch)


def create_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from augment import get_train_transforms, get_val_transforms
    
    # Create datasets
    train_dataset = CephDataset(
        data_root=config['DATA']['DATA_ROOT'],
        mode='train',
        config=config.get('DATA', {}),
        transform=get_train_transforms(config.get('AUGMENTATION', {}))
    )
    
    val_dataset = CephDataset(
        data_root=config['DATA']['DATA_ROOT'],
        mode='val',
        config=config.get('DATA', {}),
        transform=get_val_transforms()
    )
    
    test_dataset = CephDataset(
        data_root=config['DATA']['DATA_ROOT'],
        mode='test',
        config=config.get('DATA', {}),
        transform=get_val_transforms()
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['TRAIN']['BATCH_SIZE'],
        shuffle=True,
        num_workers=0,  # Disable multiprocessing to avoid Windows compatibility issues
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['EVAL']['BATCH_SIZE'],
        shuffle=False,
        num_workers=0,  # Disable multiprocessing to avoid Windows compatibility issues
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['EVAL']['BATCH_SIZE'],
        shuffle=False,
        num_workers=0,  # Disable multiprocessing to avoid Windows compatibility issues
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader, test_loader


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    # Test the dataset
    config = load_config('../configs/hrnet_w32_768x768.yaml')
    
    dataset = CephDataset(
        data_root='../data',
        mode='train',
        config=config['DATA']
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Landmarks shape: {sample['landmarks'].shape}")
    print(f"Transform params: {sample['transform_params']}") 