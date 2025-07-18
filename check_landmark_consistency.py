#!/usr/bin/env python3
"""
Check landmark files for consistency in number of landmarks.
"""

import os
from pathlib import Path

def check_landmark_consistency():
    """Check which landmark files have inconsistent numbers of landmarks."""
    landmarks_dir = Path(r"C:\Users\lacha\Downloads\ISBI Lateral Cephs\AnnotationsByMD\400_junior")
    
    print("🔍 LANDMARK CONSISTENCY CHECK")
    print("=" * 50)
    
    landmark_files = list(landmarks_dir.glob("*.txt"))
    print(f"📊 Total landmark files: {len(landmark_files)}")
    
    # Count landmarks in each file
    landmark_counts = {}
    
    for landmark_file in landmark_files:
        try:
            with open(landmark_file, 'r') as f:
                lines = f.readlines()
                
            # Count non-empty lines
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            count = len(non_empty_lines)
            
            if count not in landmark_counts:
                landmark_counts[count] = []
            landmark_counts[count].append(landmark_file.stem)
            
        except Exception as e:
            print(f"❌ Error reading {landmark_file.name}: {e}")
    
    # Report results
    print(f"\n📈 Landmark count distribution:")
    for count in sorted(landmark_counts.keys()):
        files = landmark_counts[count]
        print(f"  {count} landmarks: {len(files)} files")
        
        if count != 19:  # Expected number
            print(f"    ❌ INCONSISTENT! Expected 19 landmarks")
            if len(files) <= 10:
                for file in sorted(files):
                    print(f"      - {file}.txt")
            else:
                for file in sorted(files[:10]):
                    print(f"      - {file}.txt")
                print(f"      ... and {len(files) - 10} more")
    
    # Show sample content from different counts
    print(f"\n🔍 Sample content from different landmark counts:")
    for count in sorted(landmark_counts.keys()):
        if count != 19:  # Show problematic files
            sample_file = landmark_counts[count][0]
            sample_path = landmarks_dir / f"{sample_file}.txt"
            
            print(f"\n📄 Sample from {sample_file}.txt ({count} landmarks):")
            try:
                with open(sample_path, 'r') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines[:5]):  # Show first 5 lines
                    print(f"  {i+1}: {line.strip()}")
                
                if len(lines) > 5:
                    print(f"  ... and {len(lines) - 5} more lines")
                    
            except Exception as e:
                print(f"  ❌ Error reading file: {e}")
    
    # Check if we have images for all consistent landmark files
    print(f"\n🔍 Checking image availability for consistent landmark files:")
    data_root = Path(r"C:\Users\lacha\Downloads\ISBI Lateral Cephs")
    
    consistent_files = landmark_counts.get(19, [])
    print(f"📊 Files with 19 landmarks: {len(consistent_files)}")
    
    # Check image availability
    images_dir1 = data_root / "RawImage" / "RawImage" / "TrainingData"
    images_dir2 = data_root / "RawImage" / "RawImage" / "Test1Data"
    images_dir3 = data_root / "RawImage" / "RawImage" / "Test2Data"
    
    with_images = 0
    without_images = []
    
    for file_stem in consistent_files:
        img_path1 = images_dir1 / f"{file_stem}.bmp"
        img_path2 = images_dir2 / f"{file_stem}.bmp"
        img_path3 = images_dir3 / f"{file_stem}.bmp"
        
        if img_path1.exists() or img_path2.exists() or img_path3.exists():
            with_images += 1
        else:
            without_images.append(file_stem)
    
    print(f"✅ Consistent landmark files with images: {with_images}")
    print(f"❌ Consistent landmark files without images: {len(without_images)}")
    
    if without_images and len(without_images) <= 10:
        print(f"  Missing images for:")
        for file in sorted(without_images):
            print(f"    - {file}")
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"  - Use only the {with_images} files with 19 landmarks AND matching images")
    print(f"  - This ensures consistent dataset structure")
    print(f"  - Expected split: Train={int(with_images*0.7)}, Val={int(with_images*0.15)}, Test={with_images-int(with_images*0.7)-int(with_images*0.15)}")

if __name__ == "__main__":
    check_landmark_consistency() 