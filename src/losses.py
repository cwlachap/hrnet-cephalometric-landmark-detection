import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Any, List
from heatmaps import soft_argmax_2d, scale_coordinates


class MSELoss(nn.Module):
    """
    Mean Squared Error loss for heatmaps.
    """
    
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred_heatmaps: torch.Tensor, target_heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Calculate MSE loss between predicted and target heatmaps.
        
        Args:
            pred_heatmaps: Predicted heatmaps (batch_size, num_joints, height, width)
            target_heatmaps: Target heatmaps (batch_size, num_joints, height, width)
        
        Returns:
            MSE loss tensor
        """
        loss = F.mse_loss(pred_heatmaps, target_heatmaps, reduction=self.reduction)
        return loss


class WingLoss(nn.Module):
    """
    Wing loss for coordinate regression.
    More robust than L1/L2 loss for landmark detection.
    """
    
    def __init__(self, omega=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        
        # Pre-calculate constants
        self.C = self.omega - self.omega * np.log(1 + self.omega / self.epsilon)
    
    def forward(self, pred_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
        """
        Calculate Wing loss between predicted and target coordinates.
        
        Args:
            pred_coords: Predicted coordinates (batch_size, num_joints, 2)
            target_coords: Target coordinates (batch_size, num_joints, 2)
        
        Returns:
            Wing loss tensor
        """
        # Calculate absolute difference
        diff = torch.abs(pred_coords - target_coords)
        
        # Calculate Wing loss
        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )
        
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal loss for heatmap regression.
    Helps focus on hard examples.
    """
    
    def __init__(self, alpha=2.0, beta=4.0, gamma=0.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, pred_heatmaps: torch.Tensor, target_heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal loss for heatmaps.
        
        Args:
            pred_heatmaps: Predicted heatmaps (batch_size, num_joints, height, width)
            target_heatmaps: Target heatmaps (batch_size, num_joints, height, width)
        
        Returns:
            Focal loss tensor
        """
        pos_mask = target_heatmaps.eq(1).float()
        neg_mask = target_heatmaps.lt(1).float()
        
        neg_weights = torch.pow(1 - target_heatmaps, self.beta)
        
        pos_loss = torch.log(pred_heatmaps) * torch.pow(1 - pred_heatmaps, self.alpha) * pos_mask
        neg_loss = torch.log(1 - pred_heatmaps) * torch.pow(pred_heatmaps, self.alpha) * neg_weights * neg_mask
        
        num_pos = pos_mask.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        if num_pos == 0:
            return -neg_loss
        else:
            return -(pos_loss + neg_loss) / num_pos


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing loss that combines the benefits of Wing loss and smooth L1 loss.
    """
    
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
    
    def forward(self, pred_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
        """
        Calculate Adaptive Wing loss.
        
        Args:
            pred_coords: Predicted coordinates (batch_size, num_joints, 2)
            target_coords: Target coordinates (batch_size, num_joints, 2)
        
        Returns:
            Adaptive Wing loss tensor
        """
        diff = torch.abs(pred_coords - target_coords)
        
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target_coords))) * \
            (self.alpha - target_coords) * torch.pow(self.theta / self.epsilon, self.alpha - target_coords - 1) * \
            (1 / self.epsilon)
        
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - target_coords))
        
        loss = torch.where(
            diff < self.theta,
            self.omega * torch.log(1 + torch.pow(diff / self.epsilon, self.alpha - target_coords)),
            A * diff - C
        )
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function that includes both heatmap loss and coordinate loss.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(CombinedLoss, self).__init__()
        
        # Parse configuration
        self.heatmap_loss_type = config.get('HEATMAP_LOSS', 'mse')
        self.coord_loss_type = config.get('COORD_LOSS', 'wing')
        self.heatmap_weight = config.get('HEATMAP_WEIGHT', 1.0)
        self.coord_weight = config.get('COORD_WEIGHT', 0.5)
        
        # Wing loss parameters
        self.wing_omega = config.get('WING_OMEGA', 10.0)
        self.wing_epsilon = config.get('WING_EPSILON', 2.0)
        
        # Initialize loss functions
        if self.heatmap_loss_type == 'mse':
            self.heatmap_loss = MSELoss()
        elif self.heatmap_loss_type == 'focal':
            self.heatmap_loss = FocalLoss()
        else:
            raise ValueError(f"Unknown heatmap loss type: {self.heatmap_loss_type}")
        
        if self.coord_loss_type == 'wing':
            self.coord_loss = WingLoss(self.wing_omega, self.wing_epsilon)
        elif self.coord_loss_type == 'adaptive_wing':
            self.coord_loss = AdaptiveWingLoss()
        elif self.coord_loss_type == 'mse':
            self.coord_loss = nn.MSELoss()
        elif self.coord_loss_type == 'l1':
            self.coord_loss = nn.L1Loss()
        else:
            raise ValueError(f"Unknown coordinate loss type: {self.coord_loss_type}")
        
        # Store heatmap size for coordinate extraction
        self.heatmap_size = (192, 192)  # Default from config
        
    def forward(self, pred_heatmaps: torch.Tensor, target_heatmaps: torch.Tensor, 
                target_coords: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate combined loss.
        
        Args:
            pred_heatmaps: Predicted heatmaps (batch_size, num_joints, height, width)
            target_heatmaps: Target heatmaps (batch_size, num_joints, height, width)
            target_coords: Target coordinates (batch_size, num_joints, 2)
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Calculate heatmap loss
        heatmap_loss = self.heatmap_loss(pred_heatmaps, target_heatmaps)
        
        # Extract coordinates from predicted heatmaps using soft-argmax
        pred_coords = soft_argmax_2d(pred_heatmaps, normalize=False)
        
        # Scale coordinates to match target coordinate space (768x768)
        pred_coords = scale_coordinates(pred_coords, self.heatmap_size, (768, 768))
        
        # Calculate coordinate loss
        coord_loss = self.coord_loss(pred_coords, target_coords)
        
        # Calculate total loss
        total_loss = self.heatmap_weight * heatmap_loss + self.coord_weight * coord_loss
        
        # Return loss components for logging
        loss_dict = {
            'total_loss': total_loss,
            'heatmap_loss': heatmap_loss,
            'coord_loss': coord_loss,
            'heatmap_weight': self.heatmap_weight,
            'coord_weight': self.coord_weight
        }
        
        return total_loss, loss_dict


class JointMSELoss(nn.Module):
    """
    Joint-specific MSE loss that can weight different joints differently.
    """
    
    def __init__(self, joint_weights: torch.Tensor = None):
        super(JointMSELoss, self).__init__()
        self.joint_weights = joint_weights
    
    def forward(self, pred_heatmaps: torch.Tensor, target_heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Calculate joint-specific MSE loss.
        
        Args:
            pred_heatmaps: Predicted heatmaps (batch_size, num_joints, height, width)
            target_heatmaps: Target heatmaps (batch_size, num_joints, height, width)
        
        Returns:
            Weighted MSE loss tensor
        """
        # Calculate MSE loss for each joint
        mse_loss = F.mse_loss(pred_heatmaps, target_heatmaps, reduction='none')
        
        # Average over spatial dimensions
        mse_loss = mse_loss.mean(dim=(2, 3))  # (batch_size, num_joints)
        
        # Apply joint weights if provided
        if self.joint_weights is not None:
            if self.joint_weights.device != mse_loss.device:
                self.joint_weights = self.joint_weights.to(mse_loss.device)
            mse_loss = mse_loss * self.joint_weights.unsqueeze(0)
        
        return mse_loss.mean()


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG features.
    Can be useful for better feature representation.
    """
    
    def __init__(self, feature_layers=[0, 5, 10, 19, 28]):
        super(PerceptualLoss, self).__init__()
        
        # Load pre-trained VGG19
        import torchvision.models as models
        vgg = models.vgg19(pretrained=True)
        
        # Extract features layers
        self.feature_layers = feature_layers
        self.features = nn.ModuleList()
        
        for i, layer in enumerate(vgg.features):
            self.features.add_module(str(i), layer)
            if i in feature_layers:
                break
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, pred_heatmaps: torch.Tensor, target_heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Calculate perceptual loss using VGG features.
        
        Args:
            pred_heatmaps: Predicted heatmaps (batch_size, num_joints, height, width)
            target_heatmaps: Target heatmaps (batch_size, num_joints, height, width)
        
        Returns:
            Perceptual loss tensor
        """
        # Convert single channel to RGB
        pred_rgb = pred_heatmaps.repeat(1, 3, 1, 1) if pred_heatmaps.shape[1] == 1 else pred_heatmaps
        target_rgb = target_heatmaps.repeat(1, 3, 1, 1) if target_heatmaps.shape[1] == 1 else target_heatmaps
        
        # Extract features
        pred_features = []
        target_features = []
        
        x_pred = pred_rgb
        x_target = target_rgb
        
        for i, layer in enumerate(self.features):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            if i in self.feature_layers:
                pred_features.append(x_pred)
                target_features.append(x_target)
        
        # Calculate loss
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += F.mse_loss(pred_feat, target_feat)
        
        return loss


def get_loss_function(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create loss function based on configuration.
    
    Args:
        config: Loss configuration dictionary
    
    Returns:
        Loss function module
    """
    return CombinedLoss(config)


def calculate_landmark_weights(landmark_difficulties: List[float]) -> torch.Tensor:
    """
    Calculate weights for landmarks based on their detection difficulty.
    
    Args:
        landmark_difficulties: List of difficulty scores for each landmark
    
    Returns:
        Tensor of landmark weights
    """
    difficulties = torch.tensor(landmark_difficulties, dtype=torch.float32)
    
    # Normalize difficulties to create weights
    # Higher difficulty = higher weight
    weights = difficulties / difficulties.mean()
    
    return weights


def test_loss_functions():
    """Test the loss functions."""
    # Create dummy data
    batch_size = 2
    num_joints = 19
    height, width = 192, 192
    
    # Create dummy tensors
    pred_heatmaps = torch.randn(batch_size, num_joints, height, width)
    target_heatmaps = torch.randn(batch_size, num_joints, height, width)
    target_coords = torch.randn(batch_size, num_joints, 2) * 768  # Scale to image size
    
    # Test MSE loss
    mse_loss = MSELoss()
    mse_result = mse_loss(pred_heatmaps, target_heatmaps)
    print(f"MSE Loss: {mse_result.item():.4f}")
    
    # Test Wing loss
    wing_loss = WingLoss()
    pred_coords = torch.randn(batch_size, num_joints, 2) * 768
    wing_result = wing_loss(pred_coords, target_coords)
    print(f"Wing Loss: {wing_result.item():.4f}")
    
    # Test Combined loss
    config = {
        'HEATMAP_LOSS': 'mse',
        'COORD_LOSS': 'wing',
        'HEATMAP_WEIGHT': 1.0,
        'COORD_WEIGHT': 0.5,
        'WING_OMEGA': 10.0,
        'WING_EPSILON': 2.0
    }
    
    combined_loss = CombinedLoss(config)
    total_loss, loss_dict = combined_loss(pred_heatmaps, target_heatmaps, target_coords)
    
    print(f"Combined Loss: {total_loss.item():.4f}")
    print(f"Loss Components: {loss_dict}")
    
    # Test joint-specific loss
    joint_weights = torch.ones(num_joints)
    joint_weights[0] = 2.0  # Give higher weight to first joint
    joint_mse_loss = JointMSELoss(joint_weights)
    joint_result = joint_mse_loss(pred_heatmaps, target_heatmaps)
    print(f"Joint MSE Loss: {joint_result.item():.4f}")
    
    print("All loss function tests passed!")


if __name__ == "__main__":
    test_loss_functions() 