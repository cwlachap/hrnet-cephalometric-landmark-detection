#!/usr/bin/env python3
"""
Simple check for landmark consistency.
"""

import os
from pathlib import Path

def simple_landmark_check():
    """Simple check for landmark consistency."""
    landmarks_dir = Path(r"C:\Users\lacha\Downloads\ISBI Lateral Cephs\AnnotationsByMD\400_junior")
    
    print("Checking landmark files...")
    
    landmark_files = list(landmarks_dir.glob("*.txt"))
    print(f"Total files: {len(landmark_files)}")
    
    # Count landmarks in each file
    counts = {}
    problem_files = []
    
    for i, landmark_file in enumerate(landmark_files):
        if i < 10:  # Check first 10 files
            try:
                with open(landmark_file, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    count = len(lines)
                    
                print(f"{landmark_file.name}: {count} landmarks")
                
                if count != 19:
                    problem_files.append((landmark_file.name, count))
                    
                if count not in counts:
                    counts[count] = 0
                counts[count] += 1
                
            except Exception as e:
                print(f"Error reading {landmark_file.name}: {e}")
    
    print(f"\nSummary (first 10 files):")
    for count, num_files in counts.items():
        print(f"  {count} landmarks: {num_files} files")
    
    if problem_files:
        print(f"\nProblem files (not 19 landmarks):")
        for filename, count in problem_files:
            print(f"  - {filename}: {count} landmarks")

if __name__ == "__main__":
    simple_landmark_check() 