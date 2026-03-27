import os
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

class AutoencoderCallbacks:
    def __init__(self, monitor='val_loss', min_delta=0.001, patience=5, verbose=False,
                 log_dir='lightning_logs', study_name="optuna_study"):
        
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.log_dir = log_dir
        self.study_name = study_name

        self.study_path = f"{self.log_dir}/{self.study_name}"
        self.checkpoint_path = f"{self.study_path}/checkpoints"

        os.makedirs(self.study_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.early_stopping_cb = EarlyStopping(
            monitor=self.monitor,
            min_delta=self.min_delta,
            patience=self.patience,
            verbose=self.verbose,
            mode='min'
        )
        self.checkpoint_cb = ModelCheckpoint(
            monitor=self.monitor,
            save_top_k=1,
            mode='min',
            dirpath=self.checkpoint_path,
            filename=f'model' + '-{epoch}-{val_total_loss:.2f}',
            save_weights_only=True
        )

    def get_callbacks(self):
        return [self.early_stopping_cb, self.checkpoint_cb]
    
    def get_checkpoint_callback(self):
        return self.checkpoint_cb
