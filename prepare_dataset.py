"""
Dataset Preparation Script for Kidney Stone Composition Analysis

This script helps prepare your dataset for training and evaluation.
It creates the required JSON info file from your data directory.
"""

import os
import json
import argparse
from collections import defaultdict
import SimpleITK as sitk
import numpy as np


def calculate_volume(mask_path):
    """Calculate the volume of a kidney stone from its mask."""
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)
    spacing = mask.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    volume = np.sum(mask_array > 0) * voxel_volume
    return volume


def prepare_dataset_json(data_dir, img_dir, mask_dir, output_file, clinical_file=None):
    """
    Prepare the dataset info JSON file.
    
    Args:
        data_dir: Root directory containing the data
        img_dir: Directory name containing images
        mask_dir: Directory name containing masks
        output_file: Output JSON file path
        clinical_file: Optional clinical data Excel file
    """
    img_path = os.path.join(data_dir, img_dir)
    mask_path = os.path.join(data_dir, mask_dir)
    
    # Get all image files
    img_files = [f for f in os.listdir(img_path) if f.endswith('.nii.gz')]
    
    infos = []
    for img_file in img_files:
        # Extract sample ID (assuming format: {sid}.nii.gz)
        sid = img_file.replace('.nii.gz', '')
        mask_file = f"{sid}.nii.gz"
        mask_full_path = os.path.join(mask_path, mask_file)
        
        if not os.path.exists(mask_full_path):
            print(f"Warning: Mask not found for {sid}, skipping...")
            continue
        
        # Calculate volume
        try:
            volume = calculate_volume(mask_full_path)
        except Exception as e:
            print(f"Error calculating volume for {sid}: {e}")
            volume = 0
        
        # Default label (should be updated with actual labels)
        # You should modify this part to read actual labels from your annotation file
        info = {
            'sid': sid,
            'pid': sid,  # Patient ID, modify if different from sample ID
            'label': 0,  # Placeholder, update with actual labels
            'volume': float(volume),
            'img_path': os.path.join(img_dir, img_file),
            'mask_path': os.path.join(mask_dir, mask_file)
        }
        infos.append(info)
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(infos, f, indent=4, ensure_ascii=False)
    
    print(f"Dataset info saved to {output_file}")
    print(f"Total samples: {len(infos)}")
    
    # Print statistics
    volumes = [info['volume'] for info in infos]
    print(f"Volume statistics:")
    print(f"  Mean: {np.mean(volumes):.2f} mm³")
    print(f"  Median: {np.median(volumes):.2f} mm³")
    print(f"  Min: {np.min(volumes):.2f} mm³")
    print(f"  Max: {np.max(volumes):.2f} mm³")


def split_dataset(info_file, train_ratio=0.8, seed=1900):
    """
    Split dataset into training and testing sets with stratification.
    
    Args:
        info_file: Path to the info JSON file
        train_ratio: Ratio of training set
        seed: Random seed for reproducibility
    """
    import random
    
    with open(info_file, 'r', encoding='utf-8') as f:
        infos = json.load(f)
    
    # Group by label for stratified split
    class_data = defaultdict(list)
    for info in infos:
        class_data[info['label']].append(info)
    
    train_infos = []
    test_infos = []
    
    for label, data in class_data.items():
        random.seed(seed)
        random.shuffle(data)
        
        train_num = int(len(data) * train_ratio)
        train_infos.extend(data[:train_num])
        test_infos.extend(data[train_num:])
    
    # Save splits
    train_file = info_file.replace('.json', '_train.json')
    test_file = info_file.replace('.json', '_test.json')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_infos, f, indent=4, ensure_ascii=False)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_infos, f, indent=4, ensure_ascii=False)
    
    print(f"Train set: {len(train_infos)} samples -> {train_file}")
    print(f"Test set: {len(test_infos)} samples -> {test_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset for USCNet')
    parser.add_argument('--data-dir', type=str, required=True, 
                        help='Root directory containing the data')
    parser.add_argument('--img-dir', type=str, default='cropped_img',
                        help='Directory name containing images')
    parser.add_argument('--mask-dir', type=str, default='cropped_mask',
                        help='Directory name containing masks')
    parser.add_argument('--output', type=str, default='clinical_infos.json',
                        help='Output JSON file name')
    parser.add_argument('--split', action='store_true',
                        help='Whether to split into train/test sets')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Training set ratio')
    
    args = parser.parse_args()
    
    # Prepare dataset info
    output_path = os.path.join(args.data_dir, args.output)
    prepare_dataset_json(
        args.data_dir,
        args.img_dir,
        args.mask_dir,
        output_path
    )
    
    # Split dataset if requested
    if args.split:
        split_dataset(output_path, args.train_ratio)
