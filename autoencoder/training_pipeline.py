import torch
import os
import pandas as pd
import optuna
from sklearn.preprocessing import StandardScaler

from autoencoder.optimizer_settings import OptimizerSettings
from autoencoder.optuna_optim_3L import OptunaOptimization_3L
from autoencoder.config_files_3L import EncoderConfigSettings_3Layers, DecoderConfigSettings_3Layers
from autoencoder.AE_3Layers import Encoder_3L, Decoder_3L, Autoencoder

def build_optuna_optimization(
    window_size: int,
    latent_size: int,
    train_df_scaled: pd.DataFrame,
    val_df_scaled: pd.DataFrame,
    study_name: str
) -> OptunaOptimization_3L:
    """
    Instantiate and return the OptunaOptimization_3L object
    with the given data and study configuration.
    """
    op = OptunaOptimization_3L(
        window_size=window_size,
        latent_size=latent_size,
        X_train=train_df_scaled,
        X_val=val_df_scaled,
        study_name=study_name
    )
    return op

def build_pruner() -> optuna.pruners.SuccessiveHalvingPruner:
    """
    Instantiate and return the SuccessiveHalvingPruner.
    """
    min_resource = 15
    reduction_factor = 2
    min_early_stopping_rate = 1
    bootstrap_count = 0
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=min_resource,
        reduction_factor=reduction_factor,
        min_early_stopping_rate=min_early_stopping_rate,
        bootstrap_count=bootstrap_count
    )
    return pruner

def run_optuna_study(op: OptunaOptimization_3L, n_trials: int, n_jobs: int, pruner: optuna.pruners.BasePruner, storage_url: str) -> optuna.Study:
    """
    Create (or resume) the Optuna study, run the trials, and return the completed study.
    """
    study = optuna.create_study(
        direction='minimize',
        pruner=pruner,
        storage=storage_url,
        study_name=op.study_name,
        load_if_exists=True,
    )
    study.optimize(
        op.AE_3Layer_Tuner,
        n_trials=n_trials,
        n_jobs=n_jobs,
        callbacks=[op.callback],
        show_progress_bar=True
    )
    return study

def load_best_hparams(study_name: str, best_trial: int) -> dict:
    """
    Load the best hyperparameters saved during Optuna trials from a .pt file.
    """
    hparams_path = f"lightning_logs/{study_name}/best_hparams_trial_{best_trial}.pt"
    best_hparams = torch.load(hparams_path, weights_only=False)
    return best_hparams

def build_best_model(best_hparams: dict, study_name: str, best_trial: int) -> Autoencoder:
    """
    Reconstruct the best Autoencoder from saved hyperparameters and load its trained weights.
    """
    encoder_config = EncoderConfigSettings_3Layers(
        input_length=best_hparams["window_size"],
        input_channels=best_hparams["input_channels"],
        conv1_output_channels=best_hparams["conv1_channels"],
        conv1_kernel_size=best_hparams["conv1_kernel_size"],
        conv1_stride=best_hparams["conv1_stride"],
        activation_fn1=best_hparams["activation_fn1"],

        conv2_output_channels=best_hparams["conv2_channels"],
        conv2_kernel_size=best_hparams["conv2_kernel_size"],
        conv2_stride=best_hparams["conv2_stride"],
        activation_fn2=best_hparams["activation_fn2"],

        conv3_output_channels=best_hparams["conv3_channels"],
        conv3_kernel_size=best_hparams["conv3_kernel_size"],
        conv3_stride=best_hparams["conv3_stride"],
        activation_fn3=best_hparams["activation_fn3"],

        latent_size=best_hparams["latent_size"]
    )
    decoder_config = DecoderConfigSettings_3Layers(encoder_config)
    optimizer_config = OptimizerSettings(lr=best_hparams["lr"], weight_decay=best_hparams["weight_decay"])

    encoder = Encoder_3L(encoder_config)
    decoder = Decoder_3L(decoder_config)
    best_model = Autoencoder(encoder=encoder, decoder=decoder, optimizer_config=optimizer_config)

    model_weights_path = f"lightning_logs/{study_name}/best_model_weights_trial_{best_trial}.pth"
    best_model.load_state_dict(torch.load(model_weights_path, weights_only=True))

    return best_model

def save_model(best_model: Autoencoder, best_hparams: dict, scaler: StandardScaler, study_name: str, best_trial: int) -> None:
    """
    Package the model state_dict, hyperparameters, scaler, and metadata, then save to models/.
    """
    os.makedirs('models', exist_ok=True)
    train_result = {
        'state_dict': best_model.state_dict(),
        'hparams': best_hparams,
        'scaler': scaler,
        'study_name': study_name,
        'trial_number': best_trial,
    }
    model_path = f"models/{study_name}_best_model.pt"
    torch.save(train_result, model_path)
    print(f"Best model's info saved to: {model_path}")
    print("Training complete!\n")
    print(f"Best trial: {best_trial}\n")
