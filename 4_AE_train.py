import torch
torch.set_float32_matmul_precision('medium')
import os
from dotenv import load_dotenv

from utils.data_prep import load_data, scale_data
from autoencoder.training_pipeline import build_optuna_optimization, build_pruner, run_optuna_study, load_best_hparams, build_best_model, save_model

def main():
    load_dotenv()

    study_name = 'latent_space_2'
    window_size = 140
    latent_size = 2 # Set to None if optuna should optimize it

    train_df, val_df = load_data("data/processed/train.parquet", "data/processed/hyperparam.parquet")
    train_df_scaled, val_df_scaled, scaler = scale_data(train_df, val_df)

    postgres_user = os.getenv('POSTGRES_USER')
    postgres_password = os.getenv('POSTGRES_PASSWORD')
    storage_url = f"postgresql://{postgres_user}:{postgres_password}@localhost:5432/{study_name}"

    op = build_optuna_optimization(window_size, latent_size, train_df_scaled, val_df_scaled, study_name)

    pruner = build_pruner()
    
    n_trials = 100
    n_jobs = 2
    study = run_optuna_study(op, n_trials, n_jobs, pruner, storage_url)

    best_trial = study.best_trial.number
    best_hparams = load_best_hparams(study_name, best_trial)
    best_model = build_best_model(best_hparams, study_name, best_trial)

    save_model(best_model, best_hparams, scaler, study_name, best_trial)


if __name__ == '__main__':
    main()
