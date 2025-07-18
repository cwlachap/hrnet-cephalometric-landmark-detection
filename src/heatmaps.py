import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any
import math


def generate_gaussian_heatmap(landmark: np.ndarray, 
                            heatmap_size: Tuple[int, int], 
                            sigma: float = 2.0) -> np.ndarray:
    """
    Generate a Gaussian heatmap for a single landmark.
    
    Args:
        landmark: (x, y) coordinates of the landmark
        heatmap_size: (height, width) of the output heatmap
        sigma: Standard deviation of the Gaussian kernel
    
    Returns:
        Gaussian heatmap as numpy array
    """
    h, w = heatmap_size
    x, y = landmark
    
    # Create coordinate grids
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    
    # Calculate Gaussian
    gaussian = np.exp(-((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * sigma ** 2))
    
    # Normalize to [0, 1]
    gaussian = gaussian / np.max(gaussian) if np.max(gaussian) > 0 else gaussian
    
    return gaussian.astype(np.float32)


def generate_target_heatmaps(landmarks: np.ndarray, 
                           heatmap_size: Tuple[int, int], 
                           sigma: float = 2.0) -> np.ndarray:
    """
    Generate target heatmaps for all landmarks.
    
    Args:
        landmarks: Array of shape (N, 2) containing landmark coordinates
        heatmap_size: (height, width) of the output heatmaps
        sigma: Standard deviation of the Gaussian kernel
    
    Returns:
        Heatmaps array of shape (N, height, width)
    """
    num_landmarks = landmarks.shape[0]
    h, w = heatmap_size
    
    heatmaps = np.zeros((num_landmarks, h, w), dtype=np.float32)
    
    for i, landmark in enumerate(landmarks):
        # Scale landmark coordinates to heatmap size
        # Assuming input landmarks are in the same scale as the image (768x768)
        # and heatmap is 192x192 (768/4)
        scaled_landmark = landmark * np.array([w/768, h/768])
        
        # Generate Gaussian heatmap
        heatmap = generate_gaussian_heatmap(scaled_landmark, heatmap_size, sigma)
        heatmaps[i] = heatmap
    
    return heatmaps


def batch_generate_target_heatmaps(batch_landmarks: torch.Tensor, 
                                 heatmap_size: Tuple[int, int], 
                                 sigma: float = 2.0) -> torch.Tensor:
    """
    Generate target heatmaps for a batch of landmarks.
    
    Args:
        batch_landmarks: Tensor of shape (batch_size, num_landmarks, 2)
        heatmap_size: (height, width) of the output heatmaps
        sigma: Standard deviation of the Gaussian kernel
    
    Returns:
        Heatmaps tensor of shape (batch_size, num_landmarks, height, width)
    """
    batch_size, num_landmarks, _ = batch_landmarks.shape
    h, w = heatmap_size
    
    # Convert to numpy for processing
    landmarks_np = batch_landmarks.cpu().numpy()
    
    # Generate heatmaps for each item in the batch
    batch_heatmaps = []
    for i in range(batch_size):
        heatmaps = generate_target_heatmaps(landmarks_np[i], heatmap_size, sigma)
        batch_heatmaps.append(heatmaps)
    
    # Stack and convert back to tensor
    batch_heatmaps = np.stack(batch_heatmaps, axis=0)
    return torch.from_numpy(batch_heatmaps).float()


def soft_argmax_2d(heatmaps: torch.Tensor, 
                   temperature: float = 1.0,
                   normalize: bool = True) -> torch.Tensor:
    """
    Apply soft-argmax to extract coordinates from heatmaps.
    
    Args:
        heatmaps: Tensor of shape (batch_size, num_joints, height, width)
        temperature: Temperature parameter for softmax
        normalize: Whether to normalize coordinates to [0, 1]
    
    Returns:
        Coordinates tensor of shape (batch_size, num_joints, 2)
    """
    batch_size, num_joints, height, width = heatmaps.shape
    
    # Apply temperature scaling
    heatmaps = heatmaps / temperature
    
    # Reshape for softmax
    heatmaps_reshaped = heatmaps.view(batch_size, num_joints, -1)
    
    # Apply softmax
    softmax_maps = F.softmax(heatmaps_reshaped, dim=2)
    
    # Reshape back
    softmax_maps = softmax_maps.view(batch_size, num_joints, height, width)
    
    # Create coordinate grids
    device = heatmaps.device
    x_coords = torch.arange(width, dtype=torch.float32, device=device)
    y_coords = torch.arange(height, dtype=torch.float32, device=device)
    
    # Create meshgrid
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Expand grids to match heatmap dimensions
    x_grid = x_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, num_joints, -1, -1)
    y_grid = y_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, num_joints, -1, -1)
    
    # Calculate expected coordinates
    x_coords = torch.sum(softmax_maps * x_grid, dim=(2, 3))
    y_coords = torch.sum(softmax_maps * y_grid, dim=(2, 3))
    
    # Stack coordinates
    coordinates = torch.stack([x_coords, y_coords], dim=2)
    
    if normalize:
        # Normalize to [0, 1]
        coordinates[:, :, 0] /= (width - 1)
        coordinates[:, :, 1] /= (height - 1)
    
    return coordinates


def hard_argmax_2d(heatmaps: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Apply hard-argmax to extract coordinates from heatmaps.
    
    Args:
        heatmaps: Tensor of shape (batch_size, num_joints, height, width)
        normalize: Whether to normalize coordinates to [0, 1]
    
    Returns:
        Coordinates tensor of shape (batch_size, num_joints, 2)
    """
    batch_size, num_joints, height, width = heatmaps.shape
    
    # Reshape for argmax
    heatmaps_reshaped = heatmaps.view(batch_size, num_joints, -1)
    
    # Find maximum indices
    max_indices = torch.argmax(heatmaps_reshaped, dim=2)
    
    # Convert to 2D coordinates
    y_coords = max_indices // width
    x_coords = max_indices % width
    
    # Convert to float
    x_coords = x_coords.float()
    y_coords = y_coords.float()
    
    # Stack coordinates
    coordinates = torch.stack([x_coords, y_coords], dim=2)
    
    if normalize:
        # Normalize to [0, 1]
        coordinates[:, :, 0] /= (width - 1)
        coordinates[:, :, 1] /= (height - 1)
    
    return coordinates


def heatmap_to_coordinates(heatmaps: torch.Tensor, 
                         method: str = 'soft_argmax',
                         temperature: float = 1.0,
                         normalize: bool = True) -> torch.Tensor:
    """
    Convert heatmaps to coordinates using specified method.
    
    Args:
        heatmaps: Tensor of shape (batch_size, num_joints, height, width)
        method: 'soft_argmax' or 'hard_argmax'
        temperature: Temperature parameter for soft-argmax
        normalize: Whether to normalize coordinates
    
    Returns:
        Coordinates tensor of shape (batch_size, num_joints, 2)
    """
    if method == 'soft_argmax':
        return soft_argmax_2d(heatmaps, temperature, normalize)
    elif method == 'hard_argmax':
        return hard_argmax_2d(heatmaps, normalize)
    else:
        raise ValueError(f"Unknown method: {method}")


def scale_coordinates(coordinates: torch.Tensor, 
                     from_size: Tuple[int, int], 
                     to_size: Tuple[int, int]) -> torch.Tensor:
    """
    Scale coordinates from one size to another.
    
    Args:
        coordinates: Tensor of shape (batch_size, num_joints, 2)
        from_size: (height, width) of the source coordinate system
        to_size: (height, width) of the target coordinate system
    
    Returns:
        Scaled coordinates tensor
    """
    from_h, from_w = from_size
    to_h, to_w = to_size
    
    scale_x = to_w / from_w
    scale_y = to_h / from_h
    
    scaled_coords = coordinates.clone()
    scaled_coords[:, :, 0] *= scale_x
    scaled_coords[:, :, 1] *= scale_y
    
    return scaled_coords


def visualize_heatmap(heatmap: np.ndarray, 
                     landmark: np.ndarray = None,
                     title: str = "Heatmap") -> np.ndarray:
    """
    Visualize a heatmap with optional landmark overlay.
    
    Args:
        heatmap: 2D numpy array representing the heatmap
        landmark: Optional (x, y) coordinates to overlay
        title: Title for the visualization
    
    Returns:
        Visualization image as numpy array
    """
    # Normalize heatmap to [0, 255]
    heatmap_vis = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    
    # Add landmark if provided
    if landmark is not None:
        x, y = int(landmark[0]), int(landmark[1])
        cv2.circle(heatmap_colored, (x, y), 3, (255, 255, 255), -1)
        cv2.circle(heatmap_colored, (x, y), 3, (0, 0, 0), 1)
    
    return heatmap_colored


def visualize_batch_heatmaps(batch_heatmaps: torch.Tensor, 
                           batch_landmarks: torch.Tensor = None,
                           num_samples: int = 4) -> List[np.ndarray]:
    """
    Visualize a batch of heatmaps.
    
    Args:
        batch_heatmaps: Tensor of shape (batch_size, num_joints, height, width)
        batch_landmarks: Optional landmarks for overlay
        num_samples: Number of samples to visualize
    
    Returns:
        List of visualization images
    """
    batch_size, num_joints, height, width = batch_heatmaps.shape
    num_samples = min(num_samples, batch_size)
    
    visualizations = []
    
    for i in range(num_samples):
        # Create a grid of heatmaps for this sample
        grid_size = int(math.ceil(math.sqrt(num_joints)))
        grid_height = grid_size * height
        grid_width = grid_size * width
        
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        for j in range(num_joints):
            row = j // grid_size
            col = j % grid_size
            
            y_start = row * height
            y_end = y_start + height
            x_start = col * width
            x_end = x_start + width
            
            heatmap = batch_heatmaps[i, j].cpu().numpy()
            
            # Get landmark if provided
            landmark = None
            if batch_landmarks is not None:
                landmark = batch_landmarks[i, j].cpu().numpy()
                # Scale landmark to heatmap size
                landmark = landmark * np.array([width/768, height/768])
            
            heatmap_vis = visualize_heatmap(heatmap, landmark, f"Joint {j}")
            grid_image[y_start:y_end, x_start:x_end] = heatmap_vis
        
        visualizations.append(grid_image)
    
    return visualizations


class HeatmapGenerator(nn.Module):
    """
    PyTorch module for generating target heatmaps.
    """
    
    def __init__(self, heatmap_size: Tuple[int, int], sigma: float = 2.0):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        
        # Pre-compute coordinate grids
        h, w = heatmap_size
        y_coords = torch.arange(h, dtype=torch.float32)
        x_coords = torch.arange(w, dtype=torch.float32)
        
        # Create meshgrid
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Register as buffer (won't be updated during training)
        self.register_buffer('x_grid', x_grid)
        self.register_buffer('y_grid', y_grid)
    
    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Generate target heatmaps for landmarks.
        
        Args:
            landmarks: Tensor of shape (batch_size, num_landmarks, 2)
        
        Returns:
            Heatmaps tensor of shape (batch_size, num_landmarks, height, width)
        """
        batch_size, num_landmarks, _ = landmarks.shape
        h, w = self.heatmap_size
        
        # Scale landmarks to heatmap size
        scaled_landmarks = landmarks * torch.tensor([w/768, h/768], 
                                                   device=landmarks.device)
        
        # Expand grids and landmarks for broadcasting
        x_grid = self.x_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, num_landmarks, -1, -1)
        y_grid = self.y_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, num_landmarks, -1, -1)
        
        landmark_x = scaled_landmarks[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        landmark_y = scaled_landmarks[:, :, 1].unsqueeze(-1).unsqueeze(-1)
        
        # Calculate Gaussian heatmaps
        gaussian = torch.exp(-((x_grid - landmark_x) ** 2 + (y_grid - landmark_y) ** 2) / (2 * self.sigma ** 2))
        
        # Normalize each heatmap
        gaussian_max = torch.max(gaussian.view(batch_size, num_landmarks, -1), dim=2, keepdim=True)[0]
        gaussian_max = gaussian_max.unsqueeze(-1)
        gaussian = gaussian / (gaussian_max + 1e-8)
        
        return gaussian


def test_heatmap_generation():
    """Test the heatmap generation functions."""
    # Create dummy landmarks
    landmarks = np.array([[100, 150], [200, 250], [300, 350]], dtype=np.float32)
    heatmap_size = (192, 192)
    
    print("Testing heatmap generation...")
    
    # Test single heatmap generation
    heatmap = generate_gaussian_heatmap(landmarks[0], heatmap_size, sigma=2.0)
    print(f"Single heatmap shape: {heatmap.shape}")
    print(f"Heatmap max value: {np.max(heatmap)}")
    
    # Test batch generation
    batch_landmarks = torch.tensor(landmarks).unsqueeze(0)  # Add batch dimension
    batch_heatmaps = batch_generate_target_heatmaps(batch_landmarks, heatmap_size)
    print(f"Batch heatmaps shape: {batch_heatmaps.shape}")
    
    # Test soft-argmax
    predicted_coords = soft_argmax_2d(batch_heatmaps, normalize=False)
    print(f"Predicted coordinates shape: {predicted_coords.shape}")
    print(f"Original landmarks: {landmarks}")
    print(f"Predicted coordinates: {predicted_coords[0].numpy()}")
    
    # Test PyTorch module
    heatmap_gen = HeatmapGenerator(heatmap_size, sigma=2.0)
    torch_heatmaps = heatmap_gen(batch_landmarks)
    print(f"PyTorch heatmaps shape: {torch_heatmaps.shape}")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_heatmap_generation() 