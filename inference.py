"""
Inference Script for USCNet

This script performs inference on new CT images using a trained USCNet model.
"""

import os
import argparse
import json
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from src.models.networks.nets import DoubleFlow
from src.dataloader.load_data import MyDataset
from utils import load_pretrain


def load_model(checkpoint_path, device='cuda'):
    """
    Load the trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        model: Loaded model
    """
    # Initialize model
    model = DoubleFlow(
        in_channels=1,
        out_channels=1,
        img_size=(48, 48, 48),
        feature_size=16,
        hidden_size=768,
        num_heads=12,
        mlp_dim=3072,
        pos_embed='conv'
    )
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path, target_size=(48, 48, 48)):
    """
    Preprocess a CT image for inference.
    
    Args:
        image_path: Path to the CT image
        target_size: Target size for resizing
    
    Returns:
        preprocessed: Preprocessed image tensor
        original_image: Original SimpleITK image (for saving results)
    """
    # Read image
    image = sitk.ReadImage(image_path)
    original_image = image
    
    # Get array
    img_array = sitk.GetArrayFromImage(image)
    
    # Normalize (same as training)
    min_val = -400
    max_val = 2000
    img_array[img_array < min_val] = min_val
    img_array[img_array > max_val] = max_val
    img_array = (img_array - min_val) / (max_val - min_val)
    
    # Resize
    from scipy.ndimage import zoom
    factors = [t / s for t, s in zip(target_size, img_array.shape)]
    img_array = zoom(img_array, factors, order=1)
    
    # Convert to tensor [1, 1, D, H, W]
    img_tensor = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0)
    
    return img_tensor, original_image


def inference(model, image_path, device='cuda'):
    """
    Perform inference on a single image.
    
    Args:
        model: Loaded model
        image_path: Path to input image
        device: Device to run inference on
    
    Returns:
        prediction: Classification prediction
        segmentation: Segmentation mask
    """
    # Preprocess
    img_tensor, original_image = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # Create dummy clinical data (if model expects it)
    clinical = torch.zeros(1, 15).to(device)
    
    # Inference
    with torch.no_grad():
        seg_output, cls_output = model(img_tensor, clinical)
    
    # Process outputs
    seg_mask = torch.sigmoid(seg_output).cpu().numpy()[0, 0]
    cls_prob = torch.sigmoid(cls_output).cpu().numpy()[0, 0]
    cls_pred = 1 if cls_prob > 0.5 else 0
    
    return cls_pred, cls_prob, seg_mask, original_image


def save_results(seg_mask, original_image, output_path, threshold=0.5):
    """
    Save segmentation results.
    
    Args:
        seg_mask: Predicted segmentation mask
        original_image: Original SimpleITK image (for reference metadata)
        output_path: Path to save the result
        threshold: Threshold for binary segmentation
    """
    # Binarize mask
    binary_mask = (seg_mask > threshold).astype(np.uint8)
    
    # Convert back to original size
    original_size = sitk.GetArrayFromImage(original_image).shape
    from scipy.ndimage import zoom
    factors = [o / s for o, s in zip(original_size, binary_mask.shape)]
    binary_mask = zoom(binary_mask, factors, order=0)
    
    # Create SimpleITK image
    mask_image = sitk.GetImageFromArray(binary_mask)
    mask_image.SetOrigin(original_image.GetOrigin())
    mask_image.SetSpacing(original_image.GetSpacing())
    mask_image.SetDirection(original_image.GetDirection())
    
    # Save
    sitk.WriteImage(mask_image, output_path)
    print(f"Segmentation saved to {output_path}")


def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    if model is None:
        return
    
    # Process single image or directory
    if os.path.isfile(args.input):
        # Single image
        print(f"Processing: {args.input}")
        cls_pred, cls_prob, seg_mask, original_image = inference(model, args.input, device)
        
        print(f"Classification: {cls_pred} (probability: {cls_prob:.4f})")
        
        if args.output:
            save_results(seg_mask, original_image, args.output, args.threshold)
    
    elif os.path.isdir(args.input):
        # Directory of images
        image_files = [f for f in os.listdir(args.input) if f.endswith('.nii.gz')]
        
        results = []
        for img_file in tqdm(image_files, desc="Processing"):
            img_path = os.path.join(args.input, img_file)
            cls_pred, cls_prob, seg_mask, original_image = inference(model, img_path, device)
            
            results.append({
                'file': img_file,
                'prediction': int(cls_pred),
                'probability': float(cls_prob)
            })
            
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                output_path = os.path.join(args.output_dir, f"seg_{img_file}")
                save_results(seg_mask, original_image, output_path, args.threshold)
        
        # Save results summary
        results_file = os.path.join(args.output_dir or '.', 'inference_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_file}")
    
    else:
        print(f"Input not found: {args.input}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='USCNet Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save segmentation output (for single image)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save outputs (for batch processing)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary segmentation')
    
    args = parser.parse_args()
    main(args)
