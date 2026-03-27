import torch.nn as nn
import torch
import lightning.pytorch as pl

from autoencoder.optimizer_settings import OptimizerSettings
from autoencoder.config_files_3L import EncoderConfigSettings_3Layers, DecoderConfigSettings_3Layers

# Encoder models
class Encoder_3L(nn.Module):
    def __init__(self, config: EncoderConfigSettings_3Layers):
        """
        Args:
            config (EncoderConfigSettings_3Layers): Configuration object for encoder parameters.
        """
        super(Encoder_3L, self).__init__()

        # Layer 1 configuration
        self.conv1 = nn.Conv1d(
            in_channels=config.conv1_input_channels,
            out_channels=config.conv1_output_channels,
            kernel_size=config.conv1_kernel_size,
            stride=config.conv1_stride,
            padding=config.conv1_padding
        )
        self.activation1 = config.conv1_activation_fn

        # Layer 2 configuration
        self.conv2 = nn.Conv1d(
            in_channels=config.conv2_input_channels,
            out_channels=config.conv2_output_channels,
            kernel_size=config.conv2_kernel_size,
            stride=config.conv2_stride,
            padding=config.conv2_padding
        )
        self.activation2 = config.conv2_activation_fn

        # Layer 3 configuration
        self.conv3 = nn.Conv1d(
            in_channels=config.conv3_input_channels,
            out_channels=config.conv3_output_channels,
            kernel_size=config.conv3_kernel_size,
            stride=config.conv3_stride,
            padding=config.conv3_padding
        )
        self.activation3 = config.conv3_activation_fn

        # Flatten layer to prepare for linear latent space mapping
        self.flatten = nn.Flatten()

        # Linear layer to map to latent space
        self.latent = nn.Linear(
            config.conv3_output_channels * config.conv3_output_length, 
            config.latent_size
        )

    def forward(self, x):
        x = self.activation1(self.conv1(x))
        #print(f"x: {x.shape}")
        x = self.activation2(self.conv2(x))
        #print(f"x: {x.shape}")
        x = self.activation3(self.conv3(x))
        #print(f"x: {x.shape}")
        x = self.flatten(x)
        #print(f"x: {x.shape}")
        z = self.latent(x)
        #print(f"z: {z.shape}")
        return z
    
# Decoder models
class Decoder_3L(nn.Module):
    def __init__(self, config: DecoderConfigSettings_3Layers):
        """
        Args:
            config (DecoderConfigSettings_3Layers): Configuration object for decoder parameters.
        """
        super(Decoder_3L, self).__init__()

        # Linear layer to map from latent space to flattened input for the deconvolution
        self.latent_inv = nn.Linear(
            config.latent_size, 
            config.deconv3_input_channels * config.deconv3_input_length
        )

        # Unflatten layer to reshape the output for the first deconvolution layer
        self.unflatten = nn.Unflatten(1, (config.deconv3_input_channels, config.deconv3_input_length))

        # Layer 3 (mirroring Layer 3 of the encoder)
        self.deconv3 = nn.ConvTranspose1d(
            in_channels=config.deconv3_input_channels,
            out_channels=config.deconv3_output_channels,
            kernel_size=config.deconv3_kernel_size,
            stride=config.deconv3_stride,
            padding=config.deconv3_padding,
            output_padding=config.deconv3_output_padding
        )
        self.activation3 = config.deconv3_activation_fn

        # Layer 2 (mirroring Layer 2 of the encoder)
        self.deconv2 = nn.ConvTranspose1d(
            in_channels=config.deconv2_input_channels,
            out_channels=config.deconv2_output_channels,
            kernel_size=config.deconv2_kernel_size,
            stride=config.deconv2_stride,
            padding=config.deconv2_padding,
            output_padding=config.deconv2_output_padding
        )
        self.activation2 = config.deconv2_activation_fn

        # Layer 1 (mirroring Layer 1 of the encoder)
        self.deconv1 = nn.ConvTranspose1d(
            in_channels=config.deconv1_input_channels,
            out_channels=config.deconv1_output_channels,
            kernel_size=config.deconv1_kernel_size,
            stride=config.deconv1_stride,
            padding=config.deconv1_padding,
            output_padding=config.deconv1_output_padding
        )
        self.activation = nn.Identity() # We don't use config.deconv1_activation_fn here (preserve original input's value range)

    def forward(self, z):
        z = self.latent_inv(z)
        #print(f"z: {z.shape}")
        z = self.unflatten(z)
        #print(f"z: {z.shape}")
        z = self.activation3(self.deconv3(z))
        #print(f"z: {z.shape}")
        z = self.activation2(self.deconv2(z))
        #print(f"z: {z.shape}")
        x_reconstructed = self.activation(self.deconv1(z))
        #print(f"x_reconstructed: {x_reconstructed.shape}")
        return x_reconstructed

class Autoencoder(pl.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, optimizer_config: OptimizerSettings):
        super(Autoencoder, self).__init__()

        # Assign the encoder and decoder directly (no re-instantiation)
        self.encoder = encoder
        self.decoder = decoder

        # Optimizer settings
        self.lr = optimizer_config.lr
        self.weight_decay = optimizer_config.weight_decay

        # Loss function
        self.criterion = nn.MSELoss() # Mean Squared Error Loss

        # Initialize loss tracking lists
        self.train_losses = []
        self.val_losses = []

    def forward(self, x, return_latent=False):
        latent = self.encoder(x) # Pass through encoder
        if return_latent:
            return latent
        reconstruction = self.decoder(latent) # Pass through decoder
        return reconstruction
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        reconstruction = self.forward(x)
        loss = self.criterion(reconstruction, x)
        self.log('train_loss', loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        reconstruction = self.forward(x)
        loss = self.criterion(reconstruction, x)
        self.log('val_loss', loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def on_train_epoch_end(self):
        # Access the logged train loss from callback metrics and store it
        avg_train_loss = self.trainer.callback_metrics["train_loss"]
        self.train_losses.append(avg_train_loss.item())

    def on_validation_epoch_end(self):
        # Access the logged val loss from callback metrics and store it
        avg_val_loss = self.trainer.callback_metrics["val_loss"]
        self.val_losses.append(avg_val_loss.item())

    def freeze_weights(self):
        self._set_requires_grad(False)

    def unfreeze_weights(self):
        self._set_requires_grad(True)

    def _set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad
