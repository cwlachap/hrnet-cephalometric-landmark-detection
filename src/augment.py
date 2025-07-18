import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, Any, List


def get_train_transforms(config: Dict[str, Any]) -> A.Compose:
    """
    Get training augmentation transforms using Albumentations.
    
    Args:
        config: Augmentation configuration dictionary
    
    Returns:
        Albumentations composition for training
    """
    transforms = []
    
    # Get augmentation parameters
    rotation_limit = config.get('ROTATION', 12)
    scale_limit = config.get('SCALE', [0.9, 1.1])
    use_elastic = config.get('ELASTIC', True)
    use_clahe = config.get('CLAHE', True)
    noise_std = config.get('GAUSSIAN_NOISE', 0.02)
    cutout_prob = config.get('CUTOUT', 0.05)
    aug_prob = config.get('PROB', 0.8)
    
    # Convert scale to albumentations format
    scale_low = scale_limit[0] - 1.0  # [0.9, 1.1] -> [-0.1, 0.1]
    scale_high = scale_limit[1] - 1.0
    
    # Geometric transformations (made more conservative to prevent landmark loss)
    geometric_transforms = [
        A.Rotate(
            limit=min(rotation_limit, 8),  # Limit to 8 degrees max to prevent landmarks going out of bounds
            interpolation=1,  # cv2.INTER_LINEAR
            border_mode=0,    # cv2.BORDER_CONSTANT
            p=0.6  # Reduced probability
        ),
        A.Affine(
            scale=(max(scale_low, -0.05), min(scale_high, 0.05)),  # Clamp scale to [-0.05, 0.05] (95%-105%)
            translate_percent=0.02,  # Reduced translation from 0.05 to 0.02
            rotate=0,  # Handled by separate Rotate
            shear=0,
            interpolation=1,
            p=0.5  # Reduced probability
        )
    ]
    
    # Add elastic transform if enabled (more conservative)
    if use_elastic:
        geometric_transforms.append(
            A.ElasticTransform(
                alpha=0.5,  # Reduced from 1.0
                sigma=30,   # Reduced from 50
                p=0.3       # Reduced probability
            )
        )
    
    # Add geometric transforms
    transforms.extend(geometric_transforms)
    
    # Color/intensity transformations
    color_transforms = []
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if use_clahe:
        color_transforms.append(
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=0.5
            )
        )
    
    # Brightness/contrast adjustments
    color_transforms.extend([
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.5
        ),
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.3
        )
    ])
    
    # Add color transforms
    transforms.extend(color_transforms)
    
    # Noise and dropout
    noise_transforms = []
    
    # Gaussian noise
    if noise_std > 0:
        noise_transforms.append(
            A.GaussNoise(
                var_limit=(0, noise_std * 255),
                p=0.3
            )
        )
    
    # Cutout/CoarseDropout
    if cutout_prob > 0:
        noise_transforms.append(
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=cutout_prob
            )
        )
    
    # Add noise transforms
    transforms.extend(noise_transforms)
    
    # Compose all transforms
    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False,
            angle_in_degrees=True
        ),
        p=aug_prob
    )


def get_val_transforms(config: Dict[str, Any] = None) -> A.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        config: Configuration dictionary (unused but kept for compatibility)
    
    Returns:
        Albumentations composition for validation/test
    """
    return A.Compose(
        [],  # No augmentations for validation/test
        keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False,
            angle_in_degrees=True
        )
    )


def get_tta_transforms() -> List[A.Compose]:
    """
    Get Test Time Augmentation (TTA) transforms.
    
    Returns:
        List of transform compositions for TTA
    """
    tta_transforms = []
    
    # Original (no augmentation)
    tta_transforms.append(get_val_transforms())
    
    # Horizontal flip
    tta_transforms.append(A.Compose([
        A.HorizontalFlip(p=1.0)
    ], keypoint_params=A.KeypointParams(
        format='xy',
        remove_invisible=False,
        angle_in_degrees=True
    )))
    
    # Small rotation
    tta_transforms.append(A.Compose([
        A.Rotate(limit=5, p=1.0)
    ], keypoint_params=A.KeypointParams(
        format='xy',
        remove_invisible=False,
        angle_in_degrees=True
    )))
    
    # Slight scale
    tta_transforms.append(A.Compose([
        A.Affine(scale=(0.95, 1.05), p=1.0)
    ], keypoint_params=A.KeypointParams(
        format='xy',
        remove_invisible=False,
        angle_in_degrees=True
    )))
    
    return tta_transforms


def apply_tta(image: np.ndarray, landmarks: np.ndarray, 
              tta_transforms: List[A.Compose]) -> List[Dict[str, Any]]:
    """
    Apply Test Time Augmentation to image and landmarks.
    
    Args:
        image: Input image
        landmarks: Input landmarks
        tta_transforms: List of TTA transforms
    
    Returns:
        List of augmented samples
    """
    tta_samples = []
    
    for i, transform in enumerate(tta_transforms):
        augmented = transform(image=image, keypoints=landmarks)
        
        tta_samples.append({
            'image': augmented['image'],
            'landmarks': np.array(augmented['keypoints']),
            'transform_id': i
        })
    
    return tta_samples


def apply_multiple_augmentations(image: np.ndarray, landmarks: np.ndarray,
                                transform: A.Compose, num_samples: int = 5) -> List[Dict[str, Any]]:
    """
    Apply augmentation multiple times to generate multiple samples.
    
    Args:
        image: Input image
        landmarks: Input landmarks
        transform: Augmentation transform
        num_samples: Number of augmented samples to generate
    
    Returns:
        List of augmented samples
    """
    samples = []
    
    for i in range(num_samples):
        # Apply augmentation
        augmented = transform(image=image, keypoints=landmarks)
        
        samples.append({
            'image': augmented['image'],
            'landmarks': np.array(augmented['keypoints']),
            'sample_id': i
        })
    
    return samples


def test_augmentation_pipeline():
    """Test the augmentation pipeline with dummy data."""
    # Create dummy image and landmarks
    image = np.random.randint(0, 255, (768, 768, 3), dtype=np.uint8)
    landmarks = np.random.rand(19, 2) * 768  # 19 random landmarks
    
    # Test training transforms
    train_transform = get_train_transforms({
        'ROTATION': 12,
        'SCALE': [0.9, 1.1],
        'ELASTIC': True,
        'CLAHE': True,
        'GAUSSIAN_NOISE': 0.02,
        'CUTOUT': 0.05,
        'PROB': 0.8
    })
    
    # Test validation transforms
    val_transform = get_val_transforms()
    
    # Apply transforms
    print("Testing training transforms...")
    for i in range(3):
        try:
            augmented = train_transform(image=image, keypoints=landmarks)
            print(f"Sample {i+1}: Success")
            print(f"  Image shape: {augmented['image'].shape}")
            print(f"  Landmarks shape: {np.array(augmented['keypoints']).shape}")
        except Exception as e:
            print(f"Sample {i+1}: Error - {e}")
    
    print("\nTesting validation transforms...")
    try:
        augmented = val_transform(image=image, keypoints=landmarks)
        print("Validation: Success")
        print(f"  Image shape: {augmented['image'].shape}")
        print(f"  Landmarks shape: {np.array(augmented['keypoints']).shape}")
    except Exception as e:
        print(f"Validation: Error - {e}")
    
    # Test TTA
    print("\nTesting TTA transforms...")
    tta_transforms = get_tta_transforms()
    tta_samples = apply_tta(image, landmarks, tta_transforms)
    print(f"TTA samples generated: {len(tta_samples)}")
    
    print("\nAugmentation pipeline test completed!")


if __name__ == "__main__":
    test_augmentation_pipeline() 