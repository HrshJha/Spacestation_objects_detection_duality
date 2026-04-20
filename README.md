<<<<<<< HEAD
# Multiple Object Detection using YOLOv8 and Synthetic Data

This repository trains and evaluates a YOLOv8 object detection model using synthetic data and validates it on real-world images. The primary execution path is the Colab notebook `syntheticDataWorks_multiclass.ipynb`.

## What this project does

- Trains YOLOv8 on synthetic training images under `Output/train`
- Validates using `Output/val`
- Tests the model on real-world images under `testImages`
- Saves annotated predictions to `Output/predictions`
- Generates a summary report using `report/report.py`

## How to run

1. Open `syntheticDataWorks_multiclass.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Mount Google Drive when prompted
3. Enable GPU: `Edit > Notebook settings > Hardware accelerator > GPU`
4. Run all notebook cells from top to bottom

## Important notebook behavior

The notebook is designed to:

- mount the signed-in user’s Google Drive
- locate the shared source dataset folder under `MyDrive`
- copy the source project into the user's own Drive folder at `MyDrive/syntheticDataWorks_multiclass`
- then proceed with training, prediction, and report generation

### Current path setting

If your Drive folder contains the shared project under `MyDrive/Multiple_object_detection`, the notebook should use:

```python
SOURCE_PROJECT_ROOT = DRIVE_ROOT / 'Multiple_object_detection'
```

If the source folder path changes, update that line before running.

## What each script does

### `Output/train.py`
- Loads the YOLOv8 model from `Output/yolov8s.pt`
- Uses dataset configuration from `Output/yolo_params.yaml`
- Trains the model and saves results to `Output/runs/detect/train`

### `Output/predict.py`
- Loads the best model from the latest training run
- Predicts on images defined by the `test` field in `Output/yolo_params.yaml`
- Saves annotated prediction images to `Output/predictions/images`
- Saves prediction labels to `Output/predictions/labels`

### `report/report.py`
- Reads training and validation metrics from `Output/runs/detect`
- Copies example prediction images
- Generates `report/generatedreport.md`

## Outputs

- `Output/runs/detect/train` — training results and metrics
- `Output/predictions/images` — annotated real-world predictions
- `Output/predictions/labels` — predicted label files
- `report/generatedreport.md` — report summary

## Technical Assets

- **Runs**: Training and validation runs stored in `Output/runs/detect/`
- **Training Logs**: Metrics and loss curves in CSV and text formats within run directories
- **README Files**: This README.md and generated report in `report/`
- **Codes**: 
  - `syntheticDataWorks_multiclass.ipynb` — Main Colab notebook
  - `Output/train.py` — Training script
  - `Output/predict.py` — Prediction script
  - `Output/visualize.py` — Visualization utilities
  - `report/report.py` — Report generation script
- **Configuration**: `Output/yolo_params.yaml` — Dataset and model parameters
- **Datasets**: Synthetic training/validation data and real-world test images with labels

## Project structure

```
├── Output/
│   ├── classes.txt
│   ├── predict.py
│   ├── train.py
│   ├── visualize.py
│   ├── yolo_params.yaml
│   ├── train/
│   ├── val/
│   └── runs/
├── report/
│   └── report.py
├── testImages/
│   ├── images/
│   └── labels/
└── syntheticDataWorks_multiclass.ipynb
```

## Notes

- The notebook assumes the shared folder is accessible from the signed-in Google account.
- If the path is wrong, update `SOURCE_PROJECT_ROOT` in the notebook.
- The notebook no longer depends on a GitHub clone and uses local Drive storage instead.
=======
# Spacestation_objects_detection_duality
>>>>>>> c5d95b702e299344935c7b5df130dd307140790a
