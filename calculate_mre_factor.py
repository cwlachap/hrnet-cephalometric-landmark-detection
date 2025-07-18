#!/usr/bin/env python3
"""
Calculate the exact MRE scaling factor to convert from reported values to actual mm.
"""

def calculate_mre_factor():
    """Calculate the exact MRE scaling factor."""
    
    # Original image dimensions (from config)
    original_width = 2400
    original_height = 1935
    
    # Target dimensions (from config)
    target_width = 768
    target_height = 768
    
    print(f"📐 Image Transformation Analysis:")
    print(f"  Original size: {original_width} × {original_height}")
    print(f"  Target size: {target_width} × {target_height}")
    
    # Calculate scale factor (aspect ratio preservation)
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    
    print(f"\n🔍 Scale factors:")
    print(f"  X direction: {target_width}/{original_width} = {scale_x:.6f}")
    print(f"  Y direction: {target_height}/{original_height} = {scale_y:.6f}")
    
    # The actual scale used is the minimum (aspect ratio preservation)
    scale = min(scale_x, scale_y)
    print(f"  Used scale: min({scale_x:.6f}, {scale_y:.6f}) = {scale:.6f}")
    
    # Calculate new dimensions after scaling
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    print(f"\n📏 After scaling:")
    print(f"  New width: {original_width} × {scale:.6f} = {new_width}")
    print(f"  New height: {original_height} × {scale:.6f} = {new_height}")
    
    # Calculate padding
    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2
    
    print(f"\n🖼️ Padding:")
    print(f"  Left padding: ({target_width} - {new_width}) // 2 = {pad_left}")
    print(f"  Top padding: ({target_height} - {new_height}) // 2 = {pad_top}")
    
    # The MRE scaling factor is the inverse of the scale factor
    mre_factor = 1 / scale
    
    print(f"\n🎯 MRE Scaling Factor:")
    print(f"  Factor: 1 / {scale:.6f} = {mre_factor:.6f}")
    print(f"  Rounded: {mre_factor:.4f}")
    
    # Test with example values
    print(f"\n💡 Example conversions:")
    test_values = [0.9143, 0.9220, 1.0000, 1.5000, 2.0000]
    
    for reported_mre in test_values:
        actual_mre = reported_mre * mre_factor
        print(f"  Reported: {reported_mre:.4f} mm → Actual: {actual_mre:.4f} mm")
    
    return mre_factor

if __name__ == "__main__":
    factor = calculate_mre_factor()
    print(f"\n📊 SUMMARY:")
    print(f"  To convert reported MRE to actual mm: multiply by {factor:.4f}")
    print(f"  Your latest MRE of 0.9143 mm = {0.9143 * factor:.4f} mm actual") 