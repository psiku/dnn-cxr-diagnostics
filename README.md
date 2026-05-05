# dnn-cxr-diagnostics

Chest X-ray diagnostics project for a bachelor thesis. The training pipeline lives in [cxr-training-pipeline](cxr-training-pipeline) and is configured entirely through Kedro YAML files.
This training pipeline is the first part of my thesis project on building an application for medical triage from chest X-rays.


## Project Structure

```text
dnn-cxr-diagnostics/
├── README.md
├── cxr-training-pipeline/
│   ├── conf/                   # configurations
│   │   ├── base/
│   │   └── local/
│   ├── data/                   # data folder
│   │   ├── 01_raw/
│   │   │   └── images/         # folder with images
│   │   ├── 02_intermediate/
│   │   │   ├── dataset_splits/ # datasets splits used for training
│   │   │   └── precomputed/
│   │   │       └── 224/        # precomputed images to size of {folder_name} (in this example the images will be 224x224)
│   │   ├── 03_primary/
│   │   ├── 04_feature/
│   │   ├── 05_model_input/
│   │   ├── 06_models/          # trained models
│   │   ├── 07_model_output/    # model outputs
│   │   └── 08_reporting/       # additional reporting files
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── pyproject.toml
├── streamlit_app/              # POC app in streamlit
```


## Training flow

1. Clone the repository:

```bash
git clone https://github.com/psiku/dnn-cxr-diagnostics.git
```

2. Download the NIH ChestX-ray dataset and place the raw images in `data/01_raw/images/image.jpg` and CSV files in `data/01_raw`.
3. Install the Python dependencies:

```bash
pip install -r requirements.txt
```

4. Update the configuration files in `cxr-training-pipeline/conf/`. See **"What you must configure before training"** below.
5. Change into the Kedro project directory:

```bash
cd cxr-training-pipeline
```

6. Run the training pipeline:

```bash
kedro run
```

7. Start the MLflow UI to monitor experiments:

```bash
kedro mlflow ui
```

## Folder structure



## What you must configure before training

The important files are:

1. cxr-training-pipeline/conf/base/globals.yml -> global parameters (mostly for resizing and naming mlflow runs)
2. cxr-training-pipeline/conf/base/parameters_data_science.yml -> training configuration
3. cxr-training-pipeline/conf/base/parameters_data_processing.yml -> data processing configuration
4. cxr-training-pipeline/conf/base/catalog.yml -> dataframes and other local data sources
5. cxr-training-pipeline/conf/local/mlflow.yml -> kedro initial mlflow setup

## Training Configuration

### globals.yml

- `resize`: image size used for preprocessing and training, typically `224`
- `precomputed_root`: base folder for precomputed NumPy files
- `mlflow_experiment_name`: MLflow experiment name
- `run_name`: unique run name for checkpoints and outputs

### parameters_data_science.yml

- `training.split.val_size`: validation split ratio : 0.25 (default)
- `training.split.random_state`: split seed: 42 (default)
- `training.transforms`: augmentation settings
- `training.datamodule.kwargs.batch_size`: batch size
- `training.datamodule.kwargs.num_workers`: data loader workers
- `training.classifier.name`: set to `chest_xray_classifier` as a default classifier
- `training.classifier.kwargs.backbone_name`: backbone to train
    - "resnet18"
    - "resnet50"
    - "resnet101"
    - "resnet152"
    - "densenet121"
    - "densenet169"
    - "densenet201"
    - "densenet161"
    - "efficientnet_b0"
    - "efficientnet_b1"
    - "efficientnet_b2"
    - "efficientnet_b3"
    - "efficientnet_b4"
    - "efficientnet_b5"
    - "efficientnet_b6"
    - "efficientnet_b7"
- `training.classifier.kwargs.pretrained`: whether to load pretrained weights
- `training.classifier.kwargs.grayscale`: do you want to use images in greyscale: true (default)
- `training.classifier.kwargs.backbone_trainable_layers`: layer-name substrings to keep trainable; the exact names depend on the backbone architecture
- `training.classifier.kwargs.pooling`: `lse`, `avg`, or `max`
- `training.classifier.kwargs.lse_r`: LSE pooling temperature
- `training.classifier.kwargs.dropout`: dropout after pooling
- `training.lit_module.kwargs.lr`: learning rate
- `training.lit_module.kwargs.threshold`: initial prediction threshold
- `training.trainer.max_epochs`: number of training epochs
- `training.trainer.precision`: `16-mixed` or `32`
- `training.trainer.patience`: early stopping patience

### parameters_data_processing.yml

- `data_processing.pathology_list`: the 14 target labels; if you add or remove labels, you must update the dataset and downstream preprocessing
- `data_processing_transforms.resize`: must match `globals.resize`
- `data_processing_transforms.grayscale`: transform images to greyscale: `true` (default)
- `data_processing_precompute.images_dir`: raw image folder


## Example:  Classifier config in data_science.yml

```yaml
training:
  classifier:
    name: "chest_xray_classifier"
    kwargs:
      num_classes: 14
      backbone_name: "resnet50"
      pretrained: true
      grayscale: true
      backbone_trainable_layers:
        - "features.7"
      transition_dim: 2048
      use_transition: true
      pooling: "lse"
      lse_r: 10.0
      dropout: 0.0
```

## Models specifics

Use this as a starting point when choosing `training.classifier.kwargs.transition_dim` and image size. The image sizes below are the usual pretrained-resolution defaults for each family.

| Backbone | Version | Best image size | Transition dim | Trainable layers |
| --- | --- | ---: | ---: | --- |
| ResNet | `resnet18` | 224 | 512 | `features.6`, `features.7` |
| ResNet | `resnet50` | 224 | 2048 | `features.6`, `features.7` |
| ResNet | `resnet101` | 224 | 2048 | `features.6`, `features.7` |
| ResNet | `resnet152` | 224 | 2048 | `features.6`, `features.7` |
| DenseNet | `densenet121` | 224 | 1024 | `features.denseblock4`, `features.norm5` |
| DenseNet | `densenet161` | 224 | 2208 | `features.denseblock4`, `features.norm5` |
| DenseNet | `densenet169` | 224 | 1664 | `features.denseblock4`, `features.norm5` |
| DenseNet | `densenet201` | 224 | 1920 | `features.denseblock4`, `features.norm5` |
| EfficientNet | `efficientnet_b0` | 224 | 1280 | `features.6`, `features.7` |
| EfficientNet | `efficientnet_b1` | 240 | 1280 | `features.6`, `features.7` |
| EfficientNet | `efficientnet_b2` | 260 | 1408 | `features.6`, `features.7` |
| EfficientNet | `efficientnet_b3` | 300 | 1536 | `features.6`, `features.7` |
| EfficientNet | `efficientnet_b4` | 380 | 1792 | `features.6`, `features.7` |
| EfficientNet | `efficientnet_b5` | 456 | 2048 | `features.6`, `features.7` |
| EfficientNet | `efficientnet_b6` | 528 | 2304 | `features.6`, `features.7` |
| EfficientNet | `efficientnet_b7` | 600 | 2560 | `features.6`, `features.7` |


