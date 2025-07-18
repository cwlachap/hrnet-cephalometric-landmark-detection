#!/usr/bin/env python3
"""
Script to check the current best model without interrupting training.
"""

import torch
import os
from pathlib import Path

def check_best_model():
    """Check the current best model information."""
    model_dir = Path('models')
    best_model_path = model_dir / 'best_model.pth'
    
    if not best_model_path.exists():
        print("No best_model.pth found")
        return
    
    print(f"📁 Loading best model from: {best_model_path}")
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(best_model_path, map_location='cpu')
        
        # Extract information
        epoch = checkpoint.get('epoch', 'Unknown')
        best_val_loss = checkpoint.get('best_val_loss', 'Unknown')
        best_val_mre = checkpoint.get('best_val_mre', 'Unknown')
        
        print(f"📊 Best Model Information:")
        print(f"  🎯 Epoch: {epoch}")
        print(f"  📉 Best Validation Loss: {best_val_loss:.6f}" if isinstance(best_val_loss, (int, float)) else f"  📉 Best Validation Loss: {best_val_loss}")
        print(f"  📏 Best Validation MRE: {best_val_mre:.4f} mm" if isinstance(best_val_mre, (int, float)) else f"  📏 Best Validation MRE: {best_val_mre} mm")
        
        # Check model state dict keys to verify it's complete
        model_keys = len(checkpoint.get('model_state_dict', {}))
        print(f"  🔧 Model parameters: {model_keys} keys")
        
        # Check file size and modification time
        file_size = best_model_path.stat().st_size / (1024 * 1024)  # MB
        mod_time = best_model_path.stat().st_mtime
        
        print(f"  💾 File size: {file_size:.1f} MB")
        import datetime
        print(f"  🕐 Last modified: {datetime.datetime.fromtimestamp(mod_time)}")
        
        # Check what other recent epochs we have
        print(f"\n📂 Recent epoch checkpoints:")
        epoch_files = sorted([f for f in model_dir.glob('epoch_*.pth')], 
                           key=lambda x: int(x.stem.split('_')[1]), reverse=True)
        
        for epoch_file in epoch_files[:5]:  # Show last 5
            epoch_num = int(epoch_file.stem.split('_')[1])
            file_size = epoch_file.stat().st_size / (1024 * 1024)
            print(f"  📄 {epoch_file.name} ({file_size:.1f} MB)")
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_best_model() 