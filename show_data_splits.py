#!/usr/bin/env python3
"""
Show the exact train/validation/test split used in the dataset.
"""

import sys
import os
sys.path.append('src')

from src.dataset import CephDataset, load_config
import numpy as np

def show_data_splits():
    """Show the exact data splits used."""
    config = load_config('configs/hrnet_w32_768x768.yaml')
    
    print("🔍 DATA SPLIT ANALYSIS")
    print("=" * 50)
    
    # Create all three datasets
    train_dataset = CephDataset(
        data_root=config['DATA']['DATA_ROOT'],
        mode='train',
        config=config.get('DATA', {}),
        transform=None
    )
    
    val_dataset = CephDataset(
        data_root=config['DATA']['DATA_ROOT'],
        mode='val',
        config=config.get('DATA', {}),
        transform=None
    )
    
    test_dataset = CephDataset(
        data_root=config['DATA']['DATA_ROOT'],
        mode='test',
        config=config.get('DATA', {}),
        transform=None
    )
    
    print(f"\n📊 Split Summary:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} samples")
    
    # Show actual file names
    print(f"\n📁 TRAINING SET ({len(train_dataset)} samples):")
    train_files = sorted([int(f) for f in train_dataset.file_list])
    print(f"  Files: {train_files}")
    
    print(f"\n📁 VALIDATION SET ({len(val_dataset)} samples):")
    val_files = sorted([int(f) for f in val_dataset.file_list])
    print(f"  Files: {val_files}")
    
    print(f"\n📁 TEST SET ({len(test_dataset)} samples):")
    test_files = sorted([int(f) for f in test_dataset.file_list])
    print(f"  Files: {test_files}")
    
    # Verify no overlap
    train_set = set(train_dataset.file_list)
    val_set = set(val_dataset.file_list)
    test_set = set(test_dataset.file_list)
    
    print(f"\n🔍 VERIFICATION:")
    print(f"  No overlap between train/val: {len(train_set & val_set) == 0}")
    print(f"  No overlap between train/test: {len(train_set & test_set) == 0}")
    print(f"  No overlap between val/test: {len(val_set & test_set) == 0}")
    
    # Show some statistics
    all_files = train_files + val_files + test_files
    print(f"\n📈 STATISTICS:")
    print(f"  Min file number: {min(all_files)}")
    print(f"  Max file number: {max(all_files)}")
    print(f"  File range: {max(all_files) - min(all_files)}")
    
    # Show test set in ranges for easier reading
    print(f"\n🎯 TEST SET BREAKDOWN:")
    print(f"  Test files (sorted): {test_files}")
    
    # Group consecutive numbers
    ranges = []
    start = test_files[0]
    end = test_files[0]
    
    for i in range(1, len(test_files)):
        if test_files[i] == end + 1:
            end = test_files[i]
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = test_files[i]
            end = test_files[i]
    
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    
    print(f"  Ranges: {', '.join(ranges)}")
    
    # Save to file
    with open('data_splits.txt', 'w') as f:
        f.write("HRNet Cephalometric Dataset Splits\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training ({len(train_dataset)} samples): {train_files}\n\n")
        f.write(f"Validation ({len(val_dataset)} samples): {val_files}\n\n")
        f.write(f"Test ({len(test_dataset)} samples): {test_files}\n\n")
        f.write(f"Test ranges: {', '.join(ranges)}\n")
    
    print(f"\n📁 Split information saved to: data_splits.txt")

if __name__ == "__main__":
    show_data_splits() 