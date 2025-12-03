#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Optuna hyperparameter tuning script for Qwen2-VL fine-tuning
Optimizes warmup_steps and learning_rate based on predict_bleu-4 metric."""

import argparse
import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import optuna

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class OptunaConfig:
    """Optuna search space configuration."""

    n_trials: int = 10
    sampler_name: str = "TPESampler"
    study_name: str = "qwen2vl_hyperparameter_tuning"
    learning_rate_min: float = 1e-6
    learning_rate_max: float = 1e-3
    warmup_steps_min: int = 50
    warmup_steps_max: int = 500


@dataclass
class BaseConfig:
    """Base configuration."""

    working_directory: str = "./"
    base_output_dir: str = "./saves/optuna_tuning"
    model_base_path: str = "/home/vpp-wxs/workspace/models"
    model_name: str = "Qwen2-VL-7B-Instruct-GPTQ-Int8"
    print_detail_log: bool = True


@dataclass
class TrainingConfig:
    """Training related configuration."""

    n_epochs: int = 8
    train_dataset: str = "activitynet_qa_2000_limit_20s"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    lr_scheduler_type: str = "cosine"
    train_cutoff_len: int = 2048
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    optim: str = "adamw_torch"
    preprocessing_num_workers: int = 16
    bf16: bool = True
    finetuning_type: str = "lora"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target: str = "all"
    max_samples: int = 100000
    video_fps: float = 0.1
    # Optional parameters for non-optuna mode
    learning_rate: Optional[float] = None
    warmup_steps: Optional[int] = None


@dataclass
class EvaluationConfig:
    """Evaluation related configuration."""

    eval_dataset: str = "activitynet_qa_val_500_limit_20s"
    per_device_eval_batch_size: int = 1
    predict_with_generate: bool = True
    eval_cutoff_len: int = 1024
    max_new_tokens: int = 128
    top_p: float = 0.7
    temperature: float = 0.95
    quantization_method: str = "bitsandbytes"
    # Data related parameters
    max_samples: int = 100000
    video_fps: float = 0.1


@dataclass
class TunerConfig:
    """Complete tuning configuration."""

    optuna: Optional[OptunaConfig] = None
    base: BaseConfig = field(default_factory=BaseConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flattened dictionary (for logging output)"""
        result = {}
        for field_name, field_value in asdict(self).items():
            if isinstance(field_value, dict):
                for sub_key, sub_value in field_value.items():
                    result[f"{field_name}.{sub_key}"] = sub_value
            else:
                result[field_name] = field_value
        return result


class Qwen2VLTuner:
    def __init__(self, config: TunerConfig):

        self.config = config
        # Create base output directory
        self.base_output_dir = Path(config.base.base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        # Optuna study results storage path
        self.storage_url = f"sqlite:///{self.base_output_dir}/optuna_study.db"
        self.print_detail = config.base.print_detail_log

    def create_sampler(self):
        samplers = {
            "TPESampler": optuna.samplers.TPESampler(),
            "RandomSampler": optuna.samplers.RandomSampler(),
            "CmaEsSampler": optuna.samplers.CmaEsSampler(),
            "GPSampler": optuna.samplers.GPSampler(),
            "PartialFixedSampler": optuna.samplers.PartialFixedSampler({}, optuna.samplers.TPESampler()),
            "NSGAIISampler": optuna.samplers.NSGAIISampler(),
            "QMCSampler": optuna.samplers.QMCSampler(),
        }

        if self.config.optuna is None:
            logger.warning("No optuna config provided, using default TPESampler")
            return optuna.samplers.TPESampler()

        sampler_name = self.config.optuna.sampler_name
        if sampler_name not in samplers:
            logger.warning(f"Unknown sampler: {sampler_name}, using TPESampler instead")
            return optuna.samplers.TPESampler()

        logger.info(f"Using sampler: {sampler_name}")
        return samplers[sampler_name]

    def build_training_command(self, trial_params: Dict[str, Any], trial_output_dir: Path) -> list:
        cfg = self.config
        return [
            "llamafactory-cli",
            "train",
            "--stage",
            "sft",
            "--do_train",
            "True",
            "--model_name_or_path",
            f"{cfg.base.model_base_path}/{cfg.base.model_name}",
            "--preprocessing_num_workers",
            str(cfg.training.preprocessing_num_workers),
            "--finetuning_type",
            cfg.training.finetuning_type,
            "--template",
            "qwen2_vl",  # Fixed value
            "--flash_attn",
            "auto",  # Fixed value
            "--dataset_dir",
            "data",  # Fixed value
            "--dataset",
            cfg.training.train_dataset,  # From training config
            "--cutoff_len",
            str(cfg.training.train_cutoff_len),  # From training config
            "--learning_rate",
            str(trial_params["learning_rate"]),
            "--num_train_epochs",
            str(cfg.training.n_epochs),
            "--max_samples",
            str(cfg.training.max_samples),
            "--per_device_train_batch_size",
            str(cfg.training.per_device_train_batch_size),
            "--gradient_accumulation_steps",
            str(cfg.training.gradient_accumulation_steps),
            "--lr_scheduler_type",
            cfg.training.lr_scheduler_type,
            "--max_grad_norm",
            str(cfg.training.max_grad_norm),
            "--logging_steps",
            str(cfg.training.logging_steps),
            "--save_steps",
            str(cfg.training.save_steps),
            "--warmup_steps",
            str(trial_params["warmup_steps"]),
            "--packing",
            "False",  # Fixed value
            "--report_to",
            "none",  # Fixed value
            "--output_dir",
            str(trial_output_dir),
            "--bf16",
            str(cfg.training.bf16),
            "--plot_loss",
            "True",  # Fixed value
            "--ddp_timeout",
            "180000000",  # Fixed value
            "--optim",
            cfg.training.optim,
            "--video_fps",
            str(cfg.training.video_fps),
            "--lora_rank",
            str(cfg.training.lora_rank),
            "--lora_alpha",
            str(cfg.training.lora_alpha),
            "--lora_dropout",
            str(cfg.training.lora_dropout),
            "--lora_target",
            cfg.training.lora_target,
        ]

    def build_evaluation_command(self, checkpoint_path: str, eval_output_dir: str) -> list:
        cfg = self.config
        return [
            "llamafactory-cli",
            "train",
            "--stage",
            "sft",  # Fixed value
            "--do_predict",
            "True",  # Fixed value
            "--model_name_or_path",
            f"{cfg.base.model_base_path}/{cfg.base.model_name}",
            "--adapter_name_or_path",
            checkpoint_path,
            "--preprocessing_num_workers",
            str(cfg.training.preprocessing_num_workers),
            "--finetuning_type",
            cfg.training.finetuning_type,
            "--quantization_method",
            cfg.evaluation.quantization_method,
            "--template",
            "qwen2_vl",  # Fixed value
            "--flash_attn",
            "auto",  # Fixed value
            "--dataset_dir",
            "data",  # Fixed value
            "--eval_dataset",
            cfg.evaluation.eval_dataset,  # From evaluation config
            "--cutoff_len",
            str(cfg.evaluation.eval_cutoff_len),  # From evaluation config
            "--max_samples",
            str(cfg.evaluation.max_samples),
            "--per_device_eval_batch_size",
            str(cfg.evaluation.per_device_eval_batch_size),
            "--predict_with_generate",
            str(cfg.evaluation.predict_with_generate),
            "--max_new_tokens",
            str(cfg.evaluation.max_new_tokens),
            "--top_p",
            str(cfg.evaluation.top_p),
            "--temperature",
            str(cfg.evaluation.temperature),
            "--video_fps",
            str(cfg.evaluation.video_fps),
            "--output_dir",
            eval_output_dir,
            "--report_to",
            "none",  # Fixed value
        ]

    def run_training(self, trial_params: Dict[str, Any], trial_number: int) -> str:
        # Create trial output directory based on whether optuna config exists
        if self.config.optuna:
            trial_output_dir = self.base_output_dir / self.config.optuna.study_name / f"trial_{trial_number:03d}"
        else:
            trial_output_dir = self.base_output_dir / self.config.base.model_name / str(time.strftime("%Y%m%d_%H%M%S"))
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        training_cmd = self.build_training_command(trial_params, trial_output_dir)

        logger.info(f"Trial {trial_number}: Starting training with params: {trial_params}")
        try:
            # Run training
            logger.info(
                f"Trial {trial_number}: Executing training command in directory: {self.config.base.working_directory}"
            )
            logger.info(f"Trial {trial_number}: Training command: {' '.join(training_cmd)}")

            process = subprocess.run(
                training_cmd,
                capture_output=False if self.print_detail else True,
                text=True,
                cwd=self.config.base.working_directory,
            )
            if process.returncode != 0:
                logger.error(f"Trial {trial_number}: Training failed with return code {process.returncode}")
                raise subprocess.CalledProcessError(process.returncode, training_cmd)

            logger.info(f"Trial {trial_number}: Training completed successfully")

            # Return absolute path
            absolute_output_dir = os.path.abspath(trial_output_dir)
            logger.info(f"Trial {trial_number}: Training output directory: {absolute_output_dir}")
            return absolute_output_dir

        except Exception as e:
            logger.error(f"Trial {trial_number}: Training failed with error: {e}")
            raise

    def run_evaluation(self, checkpoint_path: str, trial_number: int) -> Dict[str, Any]:
        checkpoint_path = os.path.abspath(checkpoint_path)
        eval_output_dir = os.path.abspath(f"{checkpoint_path}/evaluation")
        os.makedirs(eval_output_dir, exist_ok=True)

        logger.info(f"Trial {trial_number}: Using checkpoint path: {checkpoint_path}")
        logger.info(f"Trial {trial_number}: Evaluation output dir: {eval_output_dir}")

        eval_cmd = self.build_evaluation_command(checkpoint_path, eval_output_dir)

        logger.info(f"Trial {trial_number}: Starting evaluation")

        try:
            # Run evaluation
            result = subprocess.run(
                eval_cmd,
                capture_output=False if self.print_detail else True,
                text=True,
                cwd=self.config.base.working_directory,
            )

            if result.returncode != 0:
                logger.error(f"Trial {trial_number}: Evaluation failed with return code {result.returncode}")
                raise subprocess.CalledProcessError(result.returncode, eval_cmd)

            # Read evaluation results
            results_file = Path(eval_output_dir) / "all_results.json"
            if not results_file.exists():
                logger.error(f"Trial {trial_number}: Results file not found: {results_file}")
                raise FileNotFoundError(f"Results file not found: {results_file}")

            with open(results_file, "r") as f:
                results = json.load(f)

            # Log all metrics
            logger.info(f"Trial {trial_number}: All evaluation metrics:")
            for metric_name, metric_value in results.items():
                logger.info(f"  {metric_name}: {metric_value}")

            return results

        except Exception as e:
            logger.error(f"Trial {trial_number}: Evaluation failed with error: {e}")
            raise

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Use configured search space
        if self.config.optuna is None:
            raise ValueError("Optuna config is None - this method should only be called when optuna config exists")

        cfg = self.config.optuna
        learning_rate = trial.suggest_float("learning_rate", cfg.learning_rate_min, cfg.learning_rate_max, log=True)
        warmup_steps = trial.suggest_int("warmup_steps", cfg.warmup_steps_min, cfg.warmup_steps_max)

        trial_params = {"learning_rate": learning_rate, "warmup_steps": warmup_steps}

        trial_number = trial.number

        try:
            logger.info(f"Trial {trial_number}: Starting with params: {trial_params}")

            # Run training
            logger.info(f"Trial {trial_number}: Starting training phase...")
            checkpoint_path = self.run_training(trial_params, trial_number)
            logger.info(f"Trial {trial_number}: Training phase completed successfully")

            # Validate training output
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Training output directory not found: {checkpoint_path}")

            # Run evaluation
            logger.info(f"Trial {trial_number}: Starting evaluation phase...")
            eval_results = self.run_evaluation(checkpoint_path, trial_number)
            logger.info(f"Trial {trial_number}: Evaluation phase completed successfully")

            # Extract BLEU-4 score for optimization
            bleu4_score = eval_results.get("predict_bleu-4", 0.0)

            # Record trial results
            logger.info(f"Trial {trial_number} completed: bleu4={bleu4_score:.4f}, params={trial_params}")

            return bleu4_score

        except Exception as e:
            logger.error(f"Trial {trial_number} failed: {e}")
            return 0.0

    def run_study(self):
        cfg = self.config

        if cfg.optuna is None:
            raise ValueError("Cannot run Optuna study - optuna config is None")

        logger.info(f"Starting Optuna study with {cfg.optuna.n_trials} trials")
        logger.info(f"Model: {cfg.base.model_name}")
        logger.info(f"Training Dataset: {cfg.training.train_dataset}")
        logger.info(f"Evaluation Dataset: {cfg.evaluation.eval_dataset}")
        logger.info(f"Epochs per trial: {cfg.training.n_epochs}")
        logger.info(f"Sampler: {cfg.optuna.sampler_name}")

        sampler = self.create_sampler()
        study = optuna.create_study(
            study_name=cfg.optuna.study_name,
            storage=self.storage_url,
            direction="maximize",  # Maximize BLEU-4 score
            load_if_exists=True,
            sampler=sampler,
        )

        # Run optimization
        study.optimize(self.objective, n_trials=cfg.optuna.n_trials)

        logger.info("Study completed!")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value (BLEU-4): {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        best_params_file = self.base_output_dir / self.config.optuna.study_name / "best_params.json"
        with open(best_params_file, "w") as f:
            json.dump(
                {
                    "best_trial_number": study.best_trial.number,
                    "best_bleu4_score": study.best_value,
                    "best_params": study.best_params,
                    "all_trials": len(study.trials),
                },
                f,
                indent=2,
            )

        logger.info(f"Best parameters saved to: {best_params_file}")

        return study


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for Qwen2-VL")

    parser.add_argument("--config_file", type=str, required=True, help="Path to JSON configuration file")
    parser.add_argument("--override", nargs="+", help="Override config values (format: key=value)")

    return parser.parse_args()


def load_config_from_file(config_file: str, overrides: Optional[list] = None) -> TunerConfig:
    """Load configuration from file and command line overrides."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r") as f:
        config_dict = json.load(f)

    # Apply command line overrides to nested configuration
    if overrides:
        for override in overrides:
            if "=" in override:
                key, value = override.split("=", 1)
                try:
                    # Try to parse as number or boolean
                    if value.lower() in ["true", "false"]:
                        parsed_value = value.lower() == "true"
                    elif value.isdigit():
                        parsed_value = int(value)
                    elif "." in value and all(part.isdigit() for part in value.split(".")):
                        parsed_value = float(value)
                    else:
                        parsed_value = value

                    # Support nested key overrides, e.g., "optuna.n_trials=5"
                    if "." in key:
                        section, param = key.split(".", 1)
                        if section in config_dict and isinstance(config_dict[section], dict):
                            config_dict[section][param] = parsed_value
                    else:
                        # Find which section the key belongs to
                        found = False
                        for section_name, section_values in config_dict.items():
                            if isinstance(section_values, dict) and key in section_values:
                                config_dict[section_name][key] = parsed_value
                                found = True
                                break
                        if not found:
                            # If not found, add to root level
                            config_dict[key] = parsed_value

                except ValueError:
                    config_dict[key] = value

    # Create TunerConfig object, now supports nested structure
    # Make optuna config optional
    optuna_config = config_dict.get("optuna")

    return TunerConfig(
        optuna=OptunaConfig(**optuna_config) if optuna_config else None,
        base=BaseConfig(**config_dict.get("base", {})),
        training=TrainingConfig(**config_dict.get("training", {})),
        evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
    )


def main():
    """Main function."""
    args = parse_arguments()

    # Load configuration from file and command line parameters
    config = load_config_from_file(args.config_file, args.override)

    # Output final configuration
    logger.info("Final configuration:")
    config_dict = config.to_dict()
    for key, value in config_dict.items():
        logger.info(f"  {key}: {value}")

    # Check what parameters are available to determine execution mode
    with open(args.config_file, "r") as f:
        raw_config = json.load(f)

    has_optuna = "optuna" in raw_config
    has_training = "training" in raw_config
    has_evaluation = "evaluation" in raw_config

    logger.info("Execution mode detection:")
    logger.info(f"  Has optuna config: {has_optuna}")
    logger.info(f"  Has training config: {has_training}")
    logger.info(f"  Has evaluation config: {has_evaluation}")

    # Create tuner
    tuner = Qwen2VLTuner(config)

    if has_optuna:
        # Run Optuna optimization
        logger.info("🔍 Running Optuna hyperparameter optimization...")
        study = tuner.run_study()

        print("\n" + "=" * 50)
        print("OPTUNA TUNING COMPLETED")
        print("=" * 50)
        print(f"Best BLEU-4 Score: {study.best_value:.4f}")
        print("Best Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print("=" * 50)

    elif has_training and has_evaluation:
        # Run training and evaluation separately
        logger.info("🚀 Running training and evaluation separately...")

        # Use parameters from training config
        if config.training.learning_rate is None or config.training.warmup_steps is None:
            logger.error("❌ Training config missing learning_rate or warmup_steps parameters!")
            raise ValueError("Training config must contain learning_rate and warmup_steps when not using Optuna")

        trial_params = {"learning_rate": config.training.learning_rate, "warmup_steps": config.training.warmup_steps}

        # Run training
        logger.info("📚 Starting training phase...")
        checkpoint_path = tuner.run_training(trial_params, trial_number=0)
        logger.info(f"✅ Training completed. Checkpoint: {checkpoint_path}")

        # Run evaluation
        logger.info("📊 Starting evaluation phase...")
        eval_results = tuner.run_evaluation(checkpoint_path, trial_number=0)
        bleu4_score = eval_results.get("predict_bleu-4", 0.0)
        logger.info(f"✅ Evaluation completed. BLEU-4 score: {bleu4_score}")

        print("\n" + "=" * 50)
        print("TRAINING AND EVALUATION COMPLETED")
        print("=" * 50)
        print(f"BLEU-4 Score: {bleu4_score:.4f}")
        print("All Evaluation Metrics:")
        for metric_name, metric_value in eval_results.items():
            print(f"  {metric_name}: {metric_value}")
        print(f"Training Parameters: {trial_params}")
        print(f"Checkpoint Path: {checkpoint_path}")
        print("=" * 50)

    elif has_training:
        # Run training only
        logger.info("🚀 Running training only...")

        # Use parameters from training config
        if config.training.learning_rate is None or config.training.warmup_steps is None:
            logger.error("❌ Training config missing learning_rate or warmup_steps parameters!")
            raise ValueError("Training config must contain learning_rate and warmup_steps when not using Optuna")

        trial_params = {"learning_rate": config.training.learning_rate, "warmup_steps": config.training.warmup_steps}

        # Run training
        logger.info("📚 Starting training phase...")
        checkpoint_path = tuner.run_training(trial_params, trial_number=0)
        logger.info(f"✅ Training completed. Checkpoint: {checkpoint_path}")

        print("\n" + "=" * 50)
        print("TRAINING COMPLETED")
        print("=" * 50)
        print(f"Training Parameters: {trial_params}")
        print(f"Checkpoint Path: {checkpoint_path}")
        print("=" * 50)

    else:
        logger.error("❌ No valid configuration found!")
        logger.error("Please provide either:")
        logger.error("  1. 'optuna' config for hyperparameter optimization")
        logger.error("  2. 'training' config for training")
        logger.error("  3. 'training' + 'evaluation' config for training and evaluation")
        raise ValueError("Invalid configuration: missing required sections")


if __name__ == "__main__":
    main()
