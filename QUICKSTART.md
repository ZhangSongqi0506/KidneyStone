# Quick Start Guide

This guide will help you get started with USCNet for kidney stone composition analysis.

## 1. Installation

```bash
# Clone or download the repository
cd KidneyStone_PublicCode

# Install dependencies
pip install -r requirements.txt
```

## 2. Data Preparation

Your data should be organized as follows:

```
your_data/
├── cropped_img/          # CT images in .nii.gz format
│   ├── sample001.nii.gz
│   ├── sample002.nii.gz
│   └── ...
└── cropped_mask/         # Segmentation masks in .nii.gz format
    ├── sample001.nii.gz
    ├── sample002.nii.gz
    └── ...
```

## 3. Prepare Dataset Info

Run the preparation script to generate the dataset info file:

```bash
python prepare_dataset.py \
    --data-dir /path/to/your_data \
    --img-dir cropped_img \
    --mask-dir cropped_mask \
    --output clinical_infos.json \
    --split
```

This will create:
- `clinical_infos.json`: All dataset information
- `clinical_infos_train.json`: Training set split
- `clinical_infos_test.json`: Testing set split

**Note**: You need to manually update the labels in `clinical_infos.json` with your actual class labels (0 or 1).

## 4. Update Configuration

Edit `configs/dataset.json` to point to your data:

```json
{
  "data_dir": "/path/to/your_data",
  "infos_name": "clinical_infos.json",
  "img_dir": "cropped_img",
  "mask_dir": "cropped_mask",
  "filter_volume": 0.006,
  "clinical_dir": null
}
```

## 5. Training

### Train SC_Net (Segmentation + Classification)

```bash
python train.py \
    --config-file configs/config.yaml \
    --task [0,1] \
    --input-size "48,48,48" \
    --num-classes 2 \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.0001 \
    --device cuda
```

### Train with Clinical Data (TMSS)

```bash
python TMSS.py \
    --config-file configs/config.yaml \
    --input-size "48,48,48" \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.0001 \
    --clinical
```

## 6. Testing

```bash
python test.py \
    --config_file configs/config.yaml \
    --task [0,1] \
    --pretrain_sc path/to/checkpoint.pth \
    --input_path /path/to/test_data \
    --batch-size 8
```

## 7. Inference on New Data

For single image:

```bash
python inference.py \
    --checkpoint path/to/best_checkpoint.pth \
    --input /path/to/image.nii.gz \
    --output /path/to/segmentation.nii.gz
```

For a directory of images:

```bash
python inference.py \
    --checkpoint path/to/best_checkpoint.pth \
    --input /path/to/image_directory/ \
    --output-dir /path/to/output_directory/
```

## 8. Evaluation

Generate ROC curves and calculate AUC:

```bash
python auc.py \
    --model_name USCNet \
    --checkpoint path/to/checkpoint.pth \
    --test_data /path/to/test_data \
    --output_dir ./results
```

## Common Issues

### Out of Memory

If you encounter CUDA out of memory errors:
- Reduce batch size: `--batch-size 4` or `--batch-size 2`
- Reduce input size: `--input-size "32,32,32"`
- Use gradient accumulation (modify train.py)

### Data Loading Errors

Make sure:
- All images and masks are in `.nii.gz` format
- Images and masks have the same dimensions
- The `sid` in your JSON matches the filename

### Model Not Found

If you get import errors:
```bash
export PYTHONPATH="${PYTHONPATH}:."
```

## Tips

1. **Input Size**: The default input size is 48×48×48. You can adjust this based on your GPU memory and image resolution.

2. **Clinical Data**: If you have clinical data (EHR), make sure to:
   - Add the Excel file path to `configs/dataset.json`
   - Use `--clinical` flag during training
   - Format: Each row should have `pid` as the first column followed by clinical features

3. **Data Augmentation**: The current implementation includes basic preprocessing. For more augmentation, modify `src/dataloader/load_data.py`.

4. **Multi-GPU Training**: The code supports DataParallel. Just run on a machine with multiple GPUs.

## Citation

If you use this code in your research, please cite our paper.

## Support

For issues and questions, please open an issue on the repository or contact the authors.
