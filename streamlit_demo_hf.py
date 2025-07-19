#!/usr/bin/env python3
"""
Streamlit demo app for HRNet cephalometric landmark detection.
Uses Hugging Face Hub for model hosting.
"""

import sys
import os
import tempfile
sys.path.append('src')

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from huggingface_hub import hf_hub_download

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

# Configuration - Your Hugging Face repository
HUGGINGFACE_REPO = "cwlachap/hrnet-cephalometric-landmark-detection"

@st.cache_resource
def download_model_from_hf():
    """Download model from Hugging Face Hub (cached for performance)."""
    try:
        with st.spinner("🤗 Downloading model from Hugging Face Hub..."):
            # Download model file
            model_path = hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename="best_model.pth",
                cache_dir=tempfile.gettempdir()
            )
            
            # Try to download config file
            try:
                config_path = hf_hub_download(
                    repo_id=HUGGINGFACE_REPO,
                    filename="config.yaml",
                    cache_dir=tempfile.gettempdir()
                )
            except:
                # If config not available on HF, use local config
                config_path = 'configs/hrnet_w32_768x768.yaml'
                
            return model_path, config_path
            
    except Exception as e:
        st.error(f"❌ Failed to download model from Hugging Face: {str(e)}")
        st.info("💡 Make sure the repository name is correct and the model is public")
        st.stop()

@st.cache_resource
def load_model_and_data():
    """Load model and test dataset (cached for performance)."""
    # Download model from Hugging Face
    model_path, config_path = download_model_from_hf()
    
    # Load config
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    with st.spinner("🧠 Loading HRNet model..."):
        model = get_hrnet_w32(config['MODEL'])
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
    
    # Try to load test dataset if available locally
    try:
        test_dataset = CephDataset(
            data_root=config['DATA']['DATA_ROOT'],
            mode='test',
            config=config.get('DATA', {}),
            transform=None
        )
        st.success(f"✅ Loaded {len(test_dataset)} test samples from local dataset")
    except Exception as e:
        st.warning("⚠️ Local test dataset not available. Upload functionality will work instead.")
        test_dataset = None
    
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
        image_tensor = image_tensor.to(device)
        pred_heatmaps = model(image_tensor)
        
        # Extract coordinates
        pred_coords = soft_argmax_2d(pred_heatmaps, normalize=False)
        pred_coords = scale_coordinates(pred_coords, (192, 192), (768, 768))
        pred_coords = pred_coords.squeeze(0).cpu().numpy()
        
    return pred_coords

def visualize_landmarks(image, pred_coords, true_coords=None, title="Landmark Detection Results"):
    """Create visualization with overlaid landmarks."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 15))
    
    # Display image
    ax.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Define landmark colors and labels
    landmark_names = [
        "Sella", "Nasion", "Orbitale", "Porion", "Subspinale", "Supramentale",
        "Pogonion", "Menton", "Gnathion", "Gonion", "Lower Incisor Tip",
        "Upper Incisor Tip", "Upper Lip", "Lower Lip", "Subnasale", 
        "Soft Tissue Pogonion", "Posterior Nasal Spine", "Anterior Nasal Spine", "Articulare"
    ]
    
    # Plot true landmarks (ground truth) in green if available
    if true_coords is not None:
        for i, (x, y) in enumerate(true_coords):
            ax.scatter(x, y, c='lime', s=100, marker='o', alpha=0.8, edgecolor='darkgreen', linewidth=2)
            ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=10, color='darkgreen', fontweight='bold')
    
    # Plot predicted landmarks in red
    for i, (x, y) in enumerate(pred_coords):
        ax.scatter(x, y, c='red', s=80, marker='x', alpha=0.8, linewidth=3)
        if true_coords is None:  # Only show numbers if no ground truth
            ax.annotate(f'{i+1}', (x, y), xytext=(5, -15), textcoords='offset points', 
                       fontsize=10, color='red', fontweight='bold')
    
    # Add legend
    legend_elements = []
    if true_coords is not None:
        legend_elements.append(plt.scatter([], [], c='lime', s=100, marker='o', alpha=0.8, 
                                         edgecolor='darkgreen', linewidth=2, label='Ground Truth'))
    legend_elements.append(plt.scatter([], [], c='red', s=80, marker='x', alpha=0.8, 
                                     linewidth=3, label='Predicted'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    return fig

def process_uploaded_image(uploaded_file, model, device, config):
    """Process uploaded image using EXACT same preprocessing as training dataset."""
    # Load image
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    image_np = np.array(image)
    
    # Apply EXACT same preprocessing as training dataset
    # 1. Resize with aspect ratio preservation (letterbox padding)
    h, w = image_np.shape[:2]
    target_h, target_w = 768, 768  # Model input size
    
    # Calculate scale factor (minimum to fit both dimensions)
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image with cubic interpolation
    resized_image = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Calculate padding
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    pad_right = target_w - new_w - pad_left
    pad_bottom = target_h - new_h - pad_top
    
    # Apply letterbox padding
    padded_image = cv2.copyMakeBorder(
        resized_image, 
        pad_top, pad_bottom, pad_left, pad_right, 
        cv2.BORDER_CONSTANT, 
        value=[0, 0, 0]
    )
    
    # 2. Apply EXACT ImageNet normalization as in training
    # Convert to [0,1]
    image_normalized = padded_image.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization (same as training)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_normalized - mean) / std
    
    # 3. Convert to PyTorch tensor with CHW format
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float().unsqueeze(0)
    
    # Store transform params for coordinate conversion
    transform_params = {
        'scale': scale,
        'pad_left': pad_left,
        'pad_top': pad_top,
        'original_size': (h, w)
    }
    
    # Predict landmarks
    pred_coords = predict_landmarks(model, image_tensor, device)
    
    # Convert coordinates back to original image space
    pred_coords_orig = pred_coords.copy()
    pred_coords_orig[:, 0] = (pred_coords[:, 0] - pad_left) / scale
    pred_coords_orig[:, 1] = (pred_coords[:, 1] - pad_top) / scale
    
    return image_np, pred_coords_orig, padded_image

def main():
    st.title("🦷 HRNet Cephalometric Landmark Detection Demo")
    st.markdown("---")
    
    # Info about the model
    st.info(f"🤗 **Model Source**: {HUGGINGFACE_REPO}")
    
    # Load model and data
    with st.spinner("Loading model and test data..."):
        model, test_dataset, device, config = load_model_and_data()
    
    st.success(f"✅ Model loaded! Using device: {device}")
    
    # Create two tabs for different input methods
    if test_dataset is not None:
        tab1, tab2 = st.tabs(["📊 Test Dataset", "📤 Upload Image"])
    else:
        tab1, tab2 = st.tabs(["📤 Upload Image", "ℹ️ About"])
        # Swap the tabs if no test dataset
        tab2, tab1 = tab1, tab2
    
    # Tab 1: Test Dataset (if available)
    if test_dataset is not None:
        with tab1:
            st.header("🧪 Test on Pre-loaded Dataset")
            st.info(f"📊 Test dataset contains {len(test_dataset)} samples")
            
            # Create sidebar for controls
            st.sidebar.header("🎛️ Dataset Controls")
            
            # Sample selection
            sample_names = [test_dataset.file_list[i] for i in range(len(test_dataset))]
            selected_sample = st.sidebar.selectbox(
                "Select Test Sample:",
                options=range(len(sample_names)),
                format_func=lambda x: f"Sample {sample_names[x]}",
                index=0
            )
            
            # Analysis options
            show_analysis = st.sidebar.checkbox("Show Cephalometric Analysis", value=True)
            show_coordinates = st.sidebar.checkbox("Show Landmark Coordinates", value=False)
            show_distances = st.sidebar.checkbox("Show Per-Landmark Distances", value=False)
            
            # Get sample data
            sample_data = test_dataset[selected_sample]
            image = sample_data['image']
            true_landmarks = sample_data['landmarks']
            transform_params = sample_data['transform_params']
            
            # Convert image tensor to numpy array for display
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:
                    image_np = image.permute(1, 2, 0).numpy()
                else:
                    image_np = image.numpy()
            else:
                image_np = image
            
            # Predict landmarks
            with st.spinner("🔮 Predicting landmarks..."):
                pred_coords = predict_landmarks(model, image, device)
            
            # Calculate MRE
            mre, distances = calculate_mre(pred_coords, true_landmarks, transform_params)
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🎯 Landmark Detection Results")
                fig = visualize_landmarks(
                    image_np, 
                    pred_coords, 
                    true_landmarks, 
                    f"Sample {sample_names[selected_sample]} - MRE: {mre:.3f}mm"
                )
                st.pyplot(fig)
            
            with col2:
                st.subheader("📊 Performance Metrics")
                st.metric("Mean Radial Error (MRE)", f"{mre:.3f} mm")
                st.metric("Good Detections (≤2mm)", f"{np.sum(distances <= 2.0)}/19")
                st.metric("Excellent Detections (≤1.5mm)", f"{np.sum(distances <= 1.5)}/19")
                
                # Performance color coding
                if mre <= 1.5:
                    st.success("🎉 Excellent Performance")
                elif mre <= 2.0:
                    st.warning("👍 Good Performance")
                else:
                    st.error("⚠️ Needs Improvement")
                
                # Cephalometric Analysis
                if show_analysis:
                    st.subheader("🔬 Cephalometric Analysis")
                    analyzer = CephalometricAnalyzer(pred_coords)
                    measurements = analyzer.calculate_all_measurements()
                    normal_ranges = analyzer.get_normal_ranges()
                    
                    for measurement, value in measurements.items():
                        status, interpretation = analyzer.interpret_measurement(measurement, value)
                        
                        if status == "Normal":
                            st.success(f"**{measurement}**: {value:.1f}° ✅")
                        elif status == "High":
                            st.warning(f"**{measurement}**: {value:.1f}° ⬆️")
                        else:
                            st.info(f"**{measurement}**: {value:.1f}° ⬇️")
                
                # Show coordinates if requested
                if show_coordinates:
                    st.subheader("📍 Landmark Coordinates")
                    landmark_names = [
                        "Sella", "Nasion", "Orbitale", "Porion", "Subspinale", "Supramentale",
                        "Pogonion", "Menton", "Gnathion", "Gonion", "Lower Incisor Tip",
                        "Upper Incisor Tip", "Upper Lip", "Lower Lip", "Subnasale", 
                        "Soft Tissue Pogonion", "Posterior Nasal Spine", "Anterior Nasal Spine", "Articulare"
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
                        "Upper Incisor Tip", "Upper Lip", "Lower Lip", "Subnasale", 
                        "Soft Tissue Pogonion", "Posterior Nasal Spine", "Anterior Nasal Spine", "Articulare"
                    ]
                    
                    for i, (name, dist) in enumerate(zip(landmark_names, distances)):
                        color = "🟢" if dist <= 2.0 else "🟡" if dist <= 2.5 else "🔴"
                        st.write(f"{color} **{i+1}. {name}**: {dist:.3f}mm")
    
    # Tab 2: Upload Image
    with tab2:
        st.header("📤 Upload Your Own Cephalometric X-ray")
        
        uploaded_file = st.file_uploader(
            "Choose a cephalometric X-ray image...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a lateral cephalometric radiograph for landmark detection"
        )
        
        if uploaded_file is not None:
            with st.spinner("🔮 Processing uploaded image..."):
                try:
                    original_image, pred_coords, processed_image = process_uploaded_image(
                        uploaded_file, model, device, config
                    )
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("🎯 Landmark Detection Results")
                        fig = visualize_landmarks(
                            original_image,
                            pred_coords,
                            title=f"Uploaded Image: {uploaded_file.name}"
                        )
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("🔬 Cephalometric Analysis")
                        analyzer = CephalometricAnalyzer(pred_coords)
                        measurements = analyzer.calculate_all_measurements()
                        
                        for measurement, value in measurements.items():
                            status, interpretation = analyzer.interpret_measurement(measurement, value)
                            
                            if status == "Normal":
                                st.success(f"**{measurement}**: {value:.1f}° ✅")
                            elif status == "High":
                                st.warning(f"**{measurement}**: {value:.1f}° ⬆️")
                            else:
                                st.info(f"**{measurement}**: {value:.1f}° ⬇️")
                        
                        # Download results
                        st.subheader("💾 Download Results")
                        
                        # Prepare coordinates for download
                        coords_text = "Landmark,X,Y\n"
                        landmark_names = [
                            "Sella", "Nasion", "Orbitale", "Porion", "Subspinale", "Supramentale",
                            "Pogonion", "Menton", "Gnathion", "Gonion", "Lower Incisor Tip",
                            "Upper Incisor Tip", "Upper Lip", "Lower Lip", "Subnasale", 
                            "Soft Tissue Pogonion", "Posterior Nasal Spine", "Anterior Nasal Spine", "Articulare"
                        ]
                        
                        for i, (name, coord) in enumerate(zip(landmark_names, pred_coords)):
                            coords_text += f"{name},{coord[0]:.2f},{coord[1]:.2f}\n"
                        
                        st.download_button(
                            label="📁 Download Coordinates (CSV)",
                            data=coords_text,
                            file_name=f"landmarks_{uploaded_file.name}.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
                    st.info("💡 Make sure the uploaded image is a valid cephalometric X-ray")
        
        else:
            st.info("👆 Upload a cephalometric X-ray image to get started!")
            st.markdown("""
            ### 📝 Instructions:
            1. Upload a lateral cephalometric radiograph
            2. The model will automatically detect 19 anatomical landmarks
            3. View the results with cephalometric analysis
            4. Download the landmark coordinates as CSV
            
            ### 🎯 Supported Formats:
            - PNG, JPG, JPEG, BMP, TIFF
            - Lateral cephalometric radiographs
            - Any resolution (will be resized automatically)
            """)
    
    # About information
    if test_dataset is None:
        with tab1:  # This is actually the "About" tab when no dataset
            st.header("ℹ️ About This Model")
            st.markdown(f"""
            ### 🤗 Model Information
            - **Source**: [{HUGGINGFACE_REPO}](https://huggingface.co/{HUGGINGFACE_REPO})
            - **Architecture**: HRNet-W32 (High-Resolution Network)
            - **Task**: 19-point cephalometric landmark detection
            - **Dataset**: ISBI Lateral Cephalograms
            
            ### 🎯 Performance
            - **Mean Radial Error**: ~1.2-1.6mm
            - **Detection Rate**: ~80-85% within 2mm
            - **Inference Time**: <1 second per image
            
            ### 🏥 Applications
            - Orthodontic treatment planning
            - Cephalometric analysis automation
            - Research and education
            - Clinical decision support
            
            ### ⚠️ Disclaimer
            This tool is for research and educational purposes only. 
            Clinical decisions should always involve qualified healthcare professionals.
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    **🤗 Powered by Hugging Face Hub** | **🧠 HRNet Architecture** | **🦷 Cephalometric Analysis**
    
    Model hosted at: [{HUGGINGFACE_REPO}](https://huggingface.co/{HUGGINGFACE_REPO})
    
    - Green circles (○) show ground truth landmarks (when available)
    - Red X marks show model predictions
    - Numbers indicate landmark IDs (1-19)
    """)

if __name__ == "__main__":
    main() 