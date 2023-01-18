from dataclasses import dataclass, field
from typing import Tuple, Optional, List

from omegaconf import MISSING


@dataclass
class DatamoduleConfig:
    _target_: str = MISSING
    batch_size: int = field(default_factory=lambda: ExperimentConfig.batch_size)
    mode: str = MISSING


@dataclass
class PairedClevrDatamoduleConfig(DatamoduleConfig):
    _target_: str = 'paired_codebook_ae.dataset.PairedClevrDatamodule'
    path_to_data_dir: str = "${hydra:runtime.cwd}/data/"
    mode: str = 'paired_clevr'
    num_workers: int = 16


@dataclass
class PairedDspritesDatamoduleConfig(DatamoduleConfig):
    _target_: str = 'paired_codebook_ae.dataset.PairedDspritesDatamodule'
    path_to_data_dir: str = "${hydra:runtime.cwd}/data/"
    mode: str = 'paired_dsprites'


@dataclass
class DatasetConfig:
    n_features: int = MISSING
    datamodule: DatamoduleConfig = MISSING
    train_size: int = MISSING
    val_size: int = MISSING
    image_size: Tuple[int, int, int] = MISSING
    requires_fid: bool = False


@dataclass
class PairedClevrConfig(DatasetConfig):
    datamodule: DatamoduleConfig = field(default_factory=PairedClevrDatamoduleConfig)
    train_size: int = 10_000
    val_size: int = 1_000
    n_features: int = 6
    image_size: Tuple[int, int, int] = (3, 128, 128)
    requires_fid: bool = True


@dataclass
class PairedDspritesConfig(DatasetConfig):
    datamodule: DatamoduleConfig = field(default_factory=PairedDspritesDatamoduleConfig)
    train_size: int = 100_000
    val_size: int = 30_000
    n_features: int = 5
    image_size: Tuple[int, int, int] = (1, 64, 64)
    requires_fid: bool = False


@dataclass
class DecoderConfig:
    in_channels: int = 64
    hidden_channels: int = 64
    initial_size: Tuple[int, int] = (4, 4)


@dataclass
class EncoderConfig:
    hidden_channels: int = 64
    output_size: Tuple[int, int] = (8, 8)


@dataclass
class ModelConfig:
    decoder_config: DecoderConfig = field(default_factory=DecoderConfig)
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    num_slots: int = 6
    num_iterations: int = 3
    slot_size: int = 64
    latent_dim: int = 1024
    mlp_hidden_size: int = 128
    binder: str = "fourier"
    monitor: str = "Validation/Triple_loss"


@dataclass
class ExperimentConfig:
    pct_start: float = 0.2
    lr: float = 0.00025
    seed: int = 0
    batch_size: int = 64
    steps_per_epoch: int = 0
    accelerator: str = 'gpu'
    devices: List[int] = field(default_factory=lambda: [0])
    max_epochs: int = 1000
    profiler: Optional[str] = None
    gradient_clip: float = 0.0
    logging_dir: str = "${hydra:run.dir}/"
    scale: str = 'sqrt'
    log_training_graph: bool = True
    reduction: str = 'sum'


@dataclass
class CheckpointsConfig:
    save_top_k: int = 1
    every_k_epochs: int = 10
    check_val_every_n_epochs: int = 20
    ckpt_path: Optional[str] = None


@dataclass
class MetricsConfig:
    metrics_dir: str = "${hydra:run.dir}/"
    ckpt_path: str = ''
    n_samples: int = 4


@dataclass
class VSADecoderConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=PairedClevrConfig)
    checkpoint: CheckpointsConfig = field(default_factory=CheckpointsConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
