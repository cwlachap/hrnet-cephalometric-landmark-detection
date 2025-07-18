#!/usr/bin/env python3
"""
Streamlit demo app for HRNet cephalometric landmark detection.
"""

import sys
import os
sys.path.append('src')

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from src.dataset import CephDataset, load_config
from src.model_hrnet import get_hrnet_w32
from src.heatmaps import soft_argmax_2d, scale_coordinates
from src.cephalometric_analysis import CephalometricAnalyzer

# Configure page
st.set_page_config(
    page_title="HRNet Cephalometric Landmark Detection Demo",
    page_icon="🦷",
    layout="wide"
)

@st.cache_resource
def load_model_and_data():
    """Load model and test dataset (cached for performance)."""
    config_path = 'configs/hrnet_w32_768x768.yaml'
    checkpoint_path = 'models/best_model.pth'
    
    # Load config
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = get_hrnet_w32(config['MODEL'])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test dataset
    test_dataset = CephDataset(
        data_root=config['DATA']['DATA_ROOT'],
        mode='test',
        config=config.get('DATA', {}),
        transform=None
    )
    
    return model, test_dataset, device, config

def calculate_mre(pred_coords, true_coords, transform_params):
    """Calculate MRE with proper coordinate transformation."""
    scale = transform_params['scale']
    pad_left = transform_params['pad_left']
    pad_top = transform_params['pad_top']
    
    # Convert to original coordinate system
    pred_orig = pred_coords.copy()
    pred_orig[:, 0] = (pred_coords[:, 0] - pad_left) / scale
    pred_orig[:, 1] = (pred_coords[:, 1] - pad_top) / scale
    
    true_orig = true_coords.copy()
    true_orig[:, 0] = (true_coords[:, 0] - pad_left) / scale
    true_orig[:, 1] = (true_coords[:, 1] - pad_top) / scale
    
    # Calculate distances
    distances = np.sqrt(np.sum((pred_orig - true_orig) ** 2, axis=1))
    
    # Apply pixel spacing (0.1mm per pixel)
    pixel_spacing = 0.1
    return np.mean(distances) * pixel_spacing, distances * pixel_spacing

def predict_landmarks(model, image_tensor, device):
    """Predict landmarks for a single image."""
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        pred_heatmaps = model(image_tensor)
        
        # Extract coordinates
        pred_coords = soft_argmax_2d(pred_heatmaps, normalize=False)
        pred_coords = scale_coordinates(pred_coords, (192, 192), (768, 768))
        pred_coords = pred_coords.squeeze(0).cpu().numpy()
        
    return pred_coords

def visualize_landmarks(image, pred_coords, true_coords, title="Landmark Visualization"):
    """Create visualization with overlaid landmarks."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 15))
    
    # Display image
    ax.imshow(image)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Define landmark colors and labels
    landmark_names = [
        "Sella", "Nasion", "Orbitale", "Porion", "Subspinale", "Supramentale",
        "Pogonion", "Menton", "Gnathion", "Gonion", "Lower Incisor Tip",
        "Lower Incisor Root", "Upper Incisor Tip", "Upper Incisor Root",
        "Upper Lip", "Lower Lip", "Subnasale", "Soft Tissue Pogonion", "Articulare"
    ]
    
    # Plot true landmarks (ground truth) in green
    for i, (x, y) in enumerate(true_coords):
        ax.scatter(x, y, c='lime', s=100, marker='o', alpha=0.8, edgecolor='darkgreen', linewidth=2)
        ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=10, color='darkgreen', fontweight='bold')
    
    # Plot predicted landmarks in red
    for i, (x, y) in enumerate(pred_coords):
        ax.scatter(x, y, c='red', s=80, marker='x', alpha=0.8, linewidth=3)
    
    # Add legend
    ax.scatter([], [], c='lime', s=100, marker='o', alpha=0.8, edgecolor='darkgreen', 
              linewidth=2, label='Ground Truth')
    ax.scatter([], [], c='red', s=80, marker='x', alpha=0.8, linewidth=3, label='Predicted')
    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    return fig

def main():
    st.title("🦷 HRNet Cephalometric Landmark Detection Demo")
    st.markdown("---")
    
    # Load model and data
    with st.spinner("Loading model and test data..."):
        model, test_dataset, device, config = load_model_and_data()
    
    st.success(f"✅ Model loaded! Using device: {device}")
    st.info(f"📊 Test dataset contains {len(test_dataset)} samples")
    
    # Create sidebar for controls
    st.sidebar.header("🎛️ Controls")
    
    # Sample selection
    sample_names = [test_dataset.file_list[i] for i in range(len(test_dataset))]
    selected_sample = st.sidebar.selectbox(
        "Select Test Sample:",
        options=range(len(sample_names)),
        format_func=lambda x: f"Sample {sample_names[x]}",
        index=0
    )
    
    # Display options
    st.sidebar.header("📋 Display Options")
    show_coordinates = st.sidebar.checkbox("Show Coordinate Values", value=False)
    show_distances = st.sidebar.checkbox("Show Per-Landmark Distances", value=False)
    show_cephalometric = st.sidebar.checkbox("Show Cephalometric Analysis", value=True)
    
    # Load selected sample
    sample = test_dataset[selected_sample]
    filename = sample['filename']
    true_landmarks = sample['landmarks'].numpy()
    transform_params = sample['transform_params']
    
    # Predict landmarks
    with st.spinner("Running inference..."):
        pred_coords = predict_landmarks(model, sample['image'], device)
    
    # Calculate metrics
    mre, distances = calculate_mre(pred_coords, true_landmarks, transform_params)
    
    # Main display area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"🖼️ Sample: {filename}")
        
        # Convert tensor to displayable image
        image_display = sample['image'].permute(1, 2, 0).numpy()
        
        # Denormalize image for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_display = image_display * std + mean
        image_display = np.clip(image_display, 0, 1)
        
        # Create visualization
        fig = visualize_landmarks(image_display, pred_coords, true_landmarks, 
                                f"Sample {filename} - MRE: {mre:.3f}mm")
        st.pyplot(fig)
    
    with col2:
        st.header("📊 Results")
        
        # Display metrics
        st.metric("Mean Radial Error", f"{mre:.3f} mm")
        
        # Performance assessment
        if mre < 2.0:
            st.success("✅ EXCELLENT Performance")
        elif mre < 2.5:
            st.info("✅ GOOD Performance")
        elif mre < 3.0:
            st.warning("⚠️ ACCEPTABLE Performance")
        else:
            st.error("❌ NEEDS IMPROVEMENT")
        
        # SDR calculation
        sdr_2mm = np.sum(distances <= 2.0) / len(distances) * 100
        sdr_2_5mm = np.sum(distances <= 2.5) / len(distances) * 100
        sdr_3mm = np.sum(distances <= 3.0) / len(distances) * 100
        
        st.subheader("🎯 Success Detection Rate")
        st.write(f"SDR@2.0mm: {sdr_2mm:.1f}%")
        st.write(f"SDR@2.5mm: {sdr_2_5mm:.1f}%")
        st.write(f"SDR@3.0mm: {sdr_3mm:.1f}%")
        
        # Cephalometric Analysis
        if show_cephalometric:
            st.subheader("📐 Cephalometric Analysis")
            
            # Use predicted landmarks for analysis
            analyzer = CephalometricAnalyzer(pred_coords)
            measurements = analyzer.calculate_all_measurements()
            
            # Create two columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Skeletal Measurements**")
                
                # SNA, SNB, ANB
                sna = measurements['SNA']
                snb = measurements['SNB']
                anb = measurements['ANB']
                
                status_sna, _ = analyzer.interpret_measurement('SNA', sna)
                status_snb, _ = analyzer.interpret_measurement('SNB', snb)
                status_anb, _ = analyzer.interpret_measurement('ANB', anb)
                
                sna_color = "🟢" if status_sna == "Normal" else "🟡" if status_sna == "Low" else "🔴"
                snb_color = "🟢" if status_snb == "Normal" else "🟡" if status_snb == "Low" else "🔴"
                anb_color = "🟢" if status_anb == "Normal" else "🟡" if status_anb == "Low" else "🔴"
                
                st.write(f"{sna_color} **SNA**: {sna:.1f}° ({status_sna})")
                st.write(f"{snb_color} **SNB**: {snb:.1f}° ({status_snb})")
                st.write(f"{anb_color} **ANB**: {anb:.1f}° ({status_anb})")
                
                # Vertical measurements
                st.markdown("**Vertical Measurements**")
                
                fma = measurements['FMA']
                sn_mp = measurements['SN-MP']
                pfh_afh = measurements['PFH/AFH']
                
                status_fma, _ = analyzer.interpret_measurement('FMA', fma)
                status_sn_mp, _ = analyzer.interpret_measurement('SN-MP', sn_mp)
                status_pfh_afh, _ = analyzer.interpret_measurement('PFH/AFH', pfh_afh)
                
                fma_color = "🟢" if status_fma == "Normal" else "🟡" if status_fma == "Low" else "🔴"
                sn_mp_color = "🟢" if status_sn_mp == "Normal" else "🟡" if status_sn_mp == "Low" else "🔴"
                pfh_afh_color = "🟢" if status_pfh_afh == "Normal" else "🟡" if status_pfh_afh == "Low" else "🔴"
                
                st.write(f"{fma_color} **FMA**: {fma:.1f}° ({status_fma})")
                st.write(f"{sn_mp_color} **SN-MP**: {sn_mp:.1f}° ({status_sn_mp})")
                st.write(f"{pfh_afh_color} **PFH/AFH**: {pfh_afh:.2f} ({status_pfh_afh})")
            
            with col2:
                st.markdown("**Dental Measurements**")
                
                # Dental measurements
                u1_sn = measurements['U1-SN']
                impa = measurements['IMPA']
                interincisal = measurements['Interincisal']
                wits = measurements['Wits']
                
                status_u1_sn, _ = analyzer.interpret_measurement('U1-SN', u1_sn)
                status_impa, _ = analyzer.interpret_measurement('IMPA', impa)
                status_interincisal, _ = analyzer.interpret_measurement('Interincisal', interincisal)
                status_wits, _ = analyzer.interpret_measurement('Wits', wits)
                
                u1_sn_color = "🟢" if status_u1_sn == "Normal" else "🟡" if status_u1_sn == "Low" else "🔴"
                impa_color = "🟢" if status_impa == "Normal" else "🟡" if status_impa == "Low" else "🔴"
                interincisal_color = "🟢" if status_interincisal == "Normal" else "🟡" if status_interincisal == "Low" else "🔴"
                wits_color = "🟢" if status_wits == "Normal" else "🟡" if status_wits == "Low" else "🔴"
                
                st.write(f"{u1_sn_color} **U1-SN**: {u1_sn:.1f}° ({status_u1_sn})")
                st.write(f"{impa_color} **IMPA**: {impa:.1f}° ({status_impa})")
                st.write(f"{interincisal_color} **Interincisal**: {interincisal:.1f}° ({status_interincisal})")
                st.write(f"{wits_color} **Wits**: {wits:.1f}mm ({status_wits})")
                
                # Summary interpretation
                st.markdown("**Clinical Summary**")
                normal_count = 0
                for measurement_name, measurement_value in measurements.items():
                    status, _ = analyzer.interpret_measurement(measurement_name, measurement_value)
                    if status == "Normal":
                        normal_count += 1
                
                total_measurements = len(measurements)
                percentage_normal = (normal_count / total_measurements) * 100
                
                if percentage_normal >= 80:
                    st.success(f"✅ **Excellent**: {normal_count}/{total_measurements} measurements normal ({percentage_normal:.1f}%)")
                elif percentage_normal >= 60:
                    st.warning(f"⚠️ **Good**: {normal_count}/{total_measurements} measurements normal ({percentage_normal:.1f}%)")
                else:
                    st.error(f"❌ **Needs Attention**: {normal_count}/{total_measurements} measurements normal ({percentage_normal:.1f}%)")
            
            # Add normal ranges reference
            with st.expander("📊 Normal Ranges Reference"):
                st.markdown("""
                **Skeletal Measurements:**
                - SNA: 80-84°
                - SNB: 78-82°
                - ANB: 0-4°
                - FMA: 20-30°
                - SN-MP: 28-38°
                - PFH/AFH: 0.60-0.80
                
                **Dental Measurements:**
                - U1-SN: 100-110°
                - IMPA: 88-95°
                - Interincisal: 120-140°
                - Wits: -1 to 3mm
                """)
        
        # Model info
        st.subheader("🔧 Model Information")
        st.write(f"**Architecture**: HRNet-W32")
        st.write(f"**Parameters**: 29.3M")
        st.write(f"**Best Epoch**: 383")
        st.write(f"**Input Size**: 768×768")
        st.write(f"**Landmarks**: 19 anatomical points")
        
        # Transform info
        st.subheader("📐 Transform Parameters")
        st.write(f"**Scale**: {transform_params['scale']:.6f}")
        st.write(f"**Padding**: Left {transform_params['pad_left']}, Top {transform_params['pad_top']}")
        st.write(f"**Original Size**: {transform_params['original_size']}")
        
        # Optional detailed displays
        if show_coordinates:
            st.subheader("📍 Landmark Coordinates")
            
            landmark_names = [
                "Sella", "Nasion", "Orbitale", "Porion", "Subspinale", "Supramentale",
                "Pogonion", "Menton", "Gnathion", "Gonion", "Lower Incisor Tip",
                "Lower Incisor Root", "Upper Incisor Tip", "Upper Incisor Root",
                "Upper Lip", "Lower Lip", "Subnasale", "Soft Tissue Pogonion", "Articulare"
            ]
            
            for i, name in enumerate(landmark_names):
                pred_x, pred_y = pred_coords[i]
                true_x, true_y = true_landmarks[i]
                st.write(f"**{i+1}. {name}**")
                st.write(f"  Pred: ({pred_x:.1f}, {pred_y:.1f})")
                st.write(f"  True: ({true_x:.1f}, {true_y:.1f})")
        
        if show_distances:
            st.subheader("📏 Per-Landmark Distances")
            landmark_names = [
                "Sella", "Nasion", "Orbitale", "Porion", "Subspinale", "Supramentale",
                "Pogonion", "Menton", "Gnathion", "Gonion", "Lower Incisor Tip",
                "Lower Incisor Root", "Upper Incisor Tip", "Upper Incisor Root",
                "Upper Lip", "Lower Lip", "Subnasale", "Soft Tissue Pogonion", "Articulare"
            ]
            
            for i, (name, dist) in enumerate(zip(landmark_names, distances)):
                color = "🟢" if dist <= 2.0 else "🟡" if dist <= 2.5 else "🔴"
                st.write(f"{color} **{i+1}. {name}**: {dist:.3f}mm")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this demo:**
    - Model trained on ISBI Lateral Cephalograms dataset
    - 19 anatomical landmarks detected
    - Green circles (○) show ground truth landmarks
    - Red X marks show model predictions
    - Numbers indicate landmark IDs
    """)

if __name__ == "__main__":
    main() 