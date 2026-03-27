import os
import lightning.pytorch as pl
import torch
import optuna

from autoencoder.config_files_3L import EncoderConfigSettings_3Layers, DecoderConfigSettings_3Layers
from autoencoder.AE_3Layers import Encoder_3L, Decoder_3L, Autoencoder
from autoencoder.datamodule import AutoencoderDataModule
from autoencoder.callbacks import AutoencoderCallbacks
from autoencoder.optimizer_settings import OptimizerSettings

class OptunaOptimization_3L():
    def __init__(self, window_size, latent_size, X_train, X_val, study_name):
        """
        Provides optuna handling for 3L autoencoder model.
        """
        self.window_size = window_size
        self.latent_size = latent_size
        self.X_train = X_train
        self.X_val = X_val
        self.study_name = study_name
        self.best_model = None
        self.history = None
        self.best_trial = None
        self.trial_data = {}
        
        self.study_path = f"lightning_logs/{study_name}"
        os.makedirs(self.study_path, exist_ok=True)

    def AE_3Layer_Tuner(self, trial):
        """
        Defines the objective function for Optuna optimization of the 3L autoencoder.
        This function returns the metric to optimize, in this case the val_loss
        """
        # Log window_size and input_channels as user attributes for later reference
        trial.set_user_attr("window_size", self.window_size)
        trial.set_user_attr('input_channels', self.X_train.shape[-1])
        
        # Defining the data module
        window_size = self.window_size
        input_channels = self.X_train.shape[-1]
        batch_size = trial.suggest_int('batch_size', 64, 512, step = 16)
        
        data_module = AutoencoderDataModule(
            train_df=self.X_train, 
            val_df=self.X_val,
            window_size=window_size,
            step_size=window_size, # non-overlapping windows
            batch_size=batch_size
        )
        
        data_module.setup()
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        # Log latent_size if it's fixed
        if self.latent_size is not None:
            trial.set_user_attr("latent_size", self.latent_size)
        
        # Suggest autoencoder's hyperparameters search space for optuna optimization
        encoder_config = EncoderConfigSettings_3Layers(
            input_length=window_size,
            input_channels=input_channels,
            
            conv1_output_channels=trial.suggest_int("conv1_channels", 8, 64, step=8),
            conv1_kernel_size=trial.suggest_categorical("conv1_kernel_size", [3, 5, 7, 9, 11]),
            conv1_stride=trial.suggest_categorical("conv1_stride", [1, 2]),
            activation_fn1=trial.suggest_categorical("activation_fn1", ["ReLU", "LeakyReLU", "Tanh", "ELU", "SiLU", "GELU"]),

            conv2_output_channels=trial.suggest_int("conv2_channels", 8, 64, step=8),
            conv2_kernel_size=trial.suggest_categorical("conv2_kernel_size", [3, 5, 7, 9, 11]),
            conv2_stride=trial.suggest_categorical("conv2_stride", [1, 2]),
            activation_fn2=trial.suggest_categorical("activation_fn2", ["ReLU", "LeakyReLU", "Tanh", "ELU", "SiLU", "GELU"]),

            conv3_output_channels=trial.suggest_int("conv3_channels", 8, 64, step=8),
            conv3_kernel_size=trial.suggest_categorical("conv3_kernel_size", [3, 5, 7, 9, 11]),
            conv3_stride=trial.suggest_categorical("conv3_stride", [1, 2]),
            activation_fn3=trial.suggest_categorical("activation_fn3", ["ReLU", "LeakyReLU", "Tanh", "ELU", "SiLU", "GELU"]),

            latent_size=self.latent_size if self.latent_size is not None else trial.suggest_categorical("latent_size", [2, 4, 8, 12, 16, 24, 32]),
        )

        decoder_config = DecoderConfigSettings_3Layers(encoder_config)

        # Instantiate encoder and decoder
        encoder = Encoder_3L(encoder_config)
        decoder = Decoder_3L(decoder_config)

        optimizer_config = OptimizerSettings(
            lr=trial.suggest_float('lr', 1e-5, 1e-3, log=True),
            weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        )

        # Initialize the Autoencoder
        model = Autoencoder(encoder=encoder, decoder=decoder, optimizer_config=optimizer_config)

        callbacks = AutoencoderCallbacks(
            monitor='val_loss',
            min_delta=1e-4,
            patience=20,
            verbose=False,
            log_dir='lightning_logs',
            study_name=self.study_name,
        )

        callbacks_list = callbacks.get_callbacks()

        # Handle the pruning
        callbacks_list.append(optuna.integration.PyTorchLightningPruningCallback(trial, monitor='val_loss'))

        # Trainer setup
        trainer = pl.Trainer(
            max_epochs=1000,
            num_sanity_val_steps=0,
            logger=False,
            log_every_n_steps=5,
            check_val_every_n_epoch=1,
            callbacks=callbacks_list,
            enable_model_summary=False,
            enable_progress_bar=False,
            accelerator='auto',
        )

        trainer.fit(model, train_loader, val_loader)

        # Retrieveing best epoch's model's info
        checkpoint_cb = callbacks.get_checkpoint_callback()
        
        if checkpoint_cb.best_model_score is not None:
            val_loss = checkpoint_cb.best_model_score.item()
        else:
            # If no checkpoint was saved, it likely means the trial was pruned before any validation step completed.
            print(f"[Trial {trial.number}] No checkpoint score found. This trial may have been pruned before any validation step completed.")
            raise RuntimeError(f"[Trial {trial.number}] No checkpoint score found — possible config bug (I/O or aggressive pruning).")

        # Save the trial data to use in callback
        self.trial_data[trial.number] = {
            "model": model,
            "val_loss": val_loss,
            "train_losses": model.train_losses,
            "val_losses": model.val_losses,
        }

        trial.set_user_attr("checkpoint_path", checkpoint_cb.best_model_path)
        return val_loss

    def callback(self, study, trial):
        """
        Callback function to handle trial results, saving the best model and its history.
        This function is called at the end of each trial.
        """
        data = self.trial_data.get(trial.number)
        if data is None:
            if trial.state == optuna.trial.TrialState.PRUNED:
                print(f"[Trial {trial.number}] was pruned.")
            elif trial.state == optuna.trial.TrialState.FAIL:
                print(f"[Trial {trial.number}] failed.")
            else:
                print(f"[Trial {trial.number}] no data found.")
            return
        
        if study.best_trial == trial:
            self.best_trial = trial
            self.best_model = data['model']
            
            # Saving history in study directory
            loss_history_path = f"{self.study_path}/loss_history_trial_{trial.number}.pt"
            torch.save(
                {'train_losses': data['train_losses'], 'val_losses': data['val_losses']},
                loss_history_path
            )
            
            # Saving best weights
            model_weights_path = f"{self.study_path}/best_model_weights_trial_{trial.number}.pth"
            checkpoint_path = trial.user_attrs.get("checkpoint_path")
            if not checkpoint_path:
                raise RuntimeError(f"[Trial {trial.number}] No checkpoint path found — possible config bug (I/O or aggressive pruning).")
            checkpoint = torch.load(checkpoint_path)
            self.best_model.load_state_dict(checkpoint["state_dict"])
            torch.save(self.best_model.state_dict(), model_weights_path)
            
            # Saving best hyperparameters
            hparams_path = f"{self.study_path}/best_hparams_trial_{trial.number}.pt"
            torch.save({
                **trial.params,
                "window_size": trial.user_attrs["window_size"],
                "input_channels": trial.user_attrs["input_channels"],
                "latent_size": trial.user_attrs["latent_size"] if self.latent_size is not None else trial.params["latent_size"],
            }, hparams_path)
        
        # Clean up trial data
        self.trial_data.pop(trial.number, None)
