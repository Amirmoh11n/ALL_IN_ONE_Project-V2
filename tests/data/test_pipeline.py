"""End-to-end test for src/data/pipeline.py (DataPipeline), using a synthetic
dataset on disk instead of the real (large) Brain Tumor MRI Dataset."""

import torch

from src.data.pipeline import DataPipeline
from src.utils.config_loader import ConfigLoader


def _write_test_config(tmp_path, train_dir, test_dir):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
data:
  raw_dir: "{tmp_path}"
  train_dir_name: "{train_dir.name}"
  test_dir_name: "{test_dir.name}"
  kaggle:
    dataset_slug: "masoudnickparvar/brain-tumor-mri-dataset"
  val_split: 0.15
  seed: 42
  image_size: 64
  normalization:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  augmentation:
    random_horizontal_flip_p: 0.5
    random_rotation_degrees: 15
    color_jitter:
      brightness: 0.1
      contrast: 0.1
  dataloader:
    batch_size: 4
    num_workers: 0
    shuffle_train: true
""")
    return config_path


def test_data_pipeline_end_to_end(synthetic_training_dir, synthetic_testing_dir, tmp_path):
    config_path = _write_test_config(tmp_path, synthetic_training_dir, synthetic_testing_dir)
    config = ConfigLoader(config_path)

    train_loader, val_loader, test_loader = DataPipeline(config).prepare()

    # 80 train-pool images total, 15% held out for val -> 68 train / 12 val; 20 test
    assert len(train_loader.dataset) == 68
    assert len(val_loader.dataset) == 12
    assert len(test_loader.dataset) == 20

    images, labels = next(iter(train_loader))
    assert images.shape == (4, 3, 64, 64)
    assert labels.shape == (4,)
    assert isinstance(images, torch.Tensor)
