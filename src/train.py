import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import create_data_loaders, load_config
from model_hrnet import get_hrnet_w32, load_pretrained_hrnet
from losses import get_loss_function
from heatmaps import HeatmapGenerator
from augment import get_train_transforms, get_val_transforms


class Trainer:
    """
    Trainer class for HRNet cephalometric landmark detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        self.model_dir = Path(config['PATHS']['MODEL_DIR'])
        self.log_dir = Path(config['PATHS']['LOG_DIR'])
        self.model_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self._init_data_loaders()
        self._init_model()
        self._init_loss_function()
        self._init_optimizer()
        self._init_scheduler()
        self._init_logging()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_mre = float('inf')
        self.patience_counter = 0
        self.scaler = GradScaler() if config['TRAIN']['AMP'] else None
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
    
    def _init_data_loaders(self):
        """Initialize data loaders."""
        print("Initializing data loaders...")
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(self.config)
        
        # Update config with actual input and heatmap sizes
        self.config['INPUT']['HEATMAP_SIZE'] = [192, 192]  # 768/4
        
    def _init_model(self):
        """Initialize model."""
        print("Initializing model...")
        self.model = get_hrnet_w32(self.config['MODEL'])
        
        # Load pretrained weights if specified
        if self.config['MODEL'].get('PRETRAINED_PATH'):
            self.model = load_pretrained_hrnet(self.model, self.config['MODEL']['PRETRAINED_PATH'])
        
        self.model = self.model.to(self.device)
        
        # Initialize heatmap generator
        self.heatmap_generator = HeatmapGenerator(
            tuple(self.config['INPUT']['HEATMAP_SIZE']),
            self.config['INPUT']['SIGMA']
        ).to(self.device)
    
    def _init_loss_function(self):
        """Initialize loss function."""
        print("Initializing loss function...")
        self.criterion = get_loss_function(self.config['LOSS'])
        
    def _init_optimizer(self):
        """Initialize optimizer."""
        print("Initializing optimizer...")
        if self.config['TRAIN']['OPTIMIZER'] == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['TRAIN']['LR'],
                weight_decay=self.config['TRAIN']['WEIGHT_DECAY']
            )
        elif self.config['TRAIN']['OPTIMIZER'] == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['TRAIN']['LR'],
                weight_decay=self.config['TRAIN']['WEIGHT_DECAY']
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['TRAIN']['OPTIMIZER']}")
    
    def _init_scheduler(self):
        """Initialize learning rate scheduler."""
        print("Initializing scheduler...")
        if self.config['TRAIN']['SCHEDULER'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['TRAIN']['EPOCHS'],
                eta_min=self.config['TRAIN']['LR'] * 0.01
            )
        elif self.config['TRAIN']['SCHEDULER'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['TRAIN']['EPOCHS'] // 3,
                gamma=0.1
            )
        else:
            self.scheduler = None
    
    def _init_logging(self):
        """Initialize logging."""
        print("Initializing logging...")
        if self.config['LOGGING']['USE_TENSORBOARD']:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        running_heatmap_loss = 0.0
        running_coord_loss = 0.0
        num_batches = 0
        
        # Initialize progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1}/{self.config["TRAIN"]["EPOCHS"]}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            landmarks = batch['landmarks'].to(self.device)
            
            # Generate target heatmaps
            with torch.no_grad():
                target_heatmaps = self.heatmap_generator(landmarks)
            
            # Forward pass with AMP
            if self.scaler is not None:
                with autocast():
                    pred_heatmaps = self.model(images)
                    loss, loss_dict = self.criterion(pred_heatmaps, target_heatmaps, landmarks)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config['TRAIN']['GRAD_ACCUMULATION']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config['TRAIN']['GRAD_ACCUMULATION'] == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                pred_heatmaps = self.model(images)
                loss, loss_dict = self.criterion(pred_heatmaps, target_heatmaps, landmarks)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config['TRAIN']['GRAD_ACCUMULATION']
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config['TRAIN']['GRAD_ACCUMULATION'] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update running losses
            running_loss += loss.item() * self.config['TRAIN']['GRAD_ACCUMULATION']
            running_heatmap_loss += loss_dict['heatmap_loss'].item()
            running_coord_loss += loss_dict['coord_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/num_batches:.4f}',
                'HM': f'{running_heatmap_loss/num_batches:.4f}',
                'Coord': f'{running_coord_loss/num_batches:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to TensorBoard
            if self.writer and batch_idx % self.config['LOGGING']['LOG_INTERVAL'] == 0:
                step = self.epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item() * self.config['TRAIN']['GRAD_ACCUMULATION'], step)
                self.writer.add_scalar('Train/HeatmapLoss', loss_dict['heatmap_loss'].item(), step)
                self.writer.add_scalar('Train/CoordLoss', loss_dict['coord_loss'].item(), step)
                self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], step)
        
        # Handle any remaining gradients
        if num_batches % self.config['TRAIN']['GRAD_ACCUMULATION'] != 0:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        return {
            'loss': running_loss / num_batches,
            'heatmap_loss': running_heatmap_loss / num_batches,
            'coord_loss': running_coord_loss / num_batches
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        running_loss = 0.0
        running_heatmap_loss = 0.0
        running_coord_loss = 0.0
        running_mre = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                
                # Generate target heatmaps
                target_heatmaps = self.heatmap_generator(landmarks)
                
                # Forward pass
                pred_heatmaps = self.model(images)
                loss, loss_dict = self.criterion(pred_heatmaps, target_heatmaps, landmarks)
                
                # Calculate MRE (Mean Radial Error) - CORRECTED VERSION
                from heatmaps import soft_argmax_2d, scale_coordinates
                pred_coords = soft_argmax_2d(pred_heatmaps, normalize=False)
                pred_coords = scale_coordinates(pred_coords, (192, 192), (768, 768))
                
                # Convert coordinates to original image space (1935×2400) for accurate mm calculation
                original_height, original_width = 1935, 2400  # Original cephalometric image size
                current_height, current_width = 768, 768      # Current coordinate system
                
                # Scale coordinates back to original image space
                pred_coords_orig = pred_coords.clone()
                pred_coords_orig[:, :, 0] *= original_width / current_width    # Scale x coordinates
                pred_coords_orig[:, :, 1] *= original_height / current_height  # Scale y coordinates
                
                landmarks_orig = landmarks.clone()
                landmarks_orig[:, :, 0] *= original_width / current_width      # Scale x coordinates  
                landmarks_orig[:, :, 1] *= original_height / current_height    # Scale y coordinates
                
                # Calculate MRE in original image space with correct pixel spacing
                pixel_spacing = 0.1  # 0.1mm per pixel in ORIGINAL image (typical for cephalometric X-rays)
                mre = torch.mean(torch.sqrt(torch.sum((pred_coords_orig - landmarks_orig) ** 2, dim=2))) * pixel_spacing
                
                # Update running losses
                running_loss += loss.item()
                running_heatmap_loss += loss_dict['heatmap_loss'].item()
                running_coord_loss += loss_dict['coord_loss'].item()
                running_mre += mre.item()
                num_batches += 1
        
        return {
            'loss': running_loss / num_batches,
            'heatmap_loss': running_heatmap_loss / num_batches,
            'coord_loss': running_coord_loss / num_batches,
            'mre': running_mre / num_batches
        }
    
    def save_checkpoint(self, is_best: bool = False, filename: str = None):
        """Save model checkpoint."""
        if filename is None:
            filename = f'epoch_{self.epoch+1}.pth'
        
        checkpoint = {
            'epoch': self.epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_mre': self.best_val_mre,
            'config': self.config
        }
        
        checkpoint_path = self.model_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.model_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_mre = checkpoint['best_val_mre']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.epoch, self.config['TRAIN']['EPOCHS']):
            self.epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            print(f"Epoch {epoch+1}/{self.config['TRAIN']['EPOCHS']}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val MRE: {val_metrics['mre']:.4f} mm")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Epoch/Train_Loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('Epoch/Val_Loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Epoch/Val_MRE', val_metrics['mre'], epoch)
                self.writer.add_scalar('Epoch/Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            is_best = val_metrics['mre'] < self.best_val_mre
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_val_mre = val_metrics['mre']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoints
            if (epoch + 1) % self.config['LOGGING']['SAVE_INTERVAL'] == 0:
                self.save_checkpoint(is_best=is_best)
            
            if is_best:
                self.save_checkpoint(is_best=True)
            
            # Early stopping
            if self.patience_counter >= self.config['TRAIN']['EARLY_STOPPING']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Final checkpoint
        self.save_checkpoint(filename='final_model.pth')
        
        # Training summary
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best validation MRE: {self.best_val_mre:.4f} mm")
        
        if self.writer:
            self.writer.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train HRNet for cephalometric landmark detection')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.batch_size is not None:
        config['TRAIN']['BATCH_SIZE'] = args.batch_size
    if args.epochs is not None:
        config['TRAIN']['EPOCHS'] = args.epochs
    if args.lr is not None:
        config['TRAIN']['LR'] = args.lr
    
    # Debug mode
    if args.debug:
        config['LOGGING']['LOG_INTERVAL'] = 1
        config['LOGGING']['SAVE_INTERVAL'] = 1
        config['TRAIN']['EPOCHS'] = 5
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint(filename='interrupted_model.pth')
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        trainer.save_checkpoint(filename='failed_model.pth')
        raise


if __name__ == "__main__":
    main() 