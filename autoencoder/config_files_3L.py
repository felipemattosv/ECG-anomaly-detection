from torch import nn

# Encoder Configurations
class EncoderConfigSettings_3Layers:
    """
    Configuration class for a 3-layer Encoder model.
    """
    def __init__(self, input_length=32, input_channels=1, 
                 conv1_output_channels=16, conv1_kernel_size=3, conv1_stride=2,
                 conv2_output_channels=32, conv2_kernel_size=3, conv2_stride=2,
                 conv3_output_channels=64, conv3_kernel_size=3, conv3_stride=2,
                 activation_fn1='ReLU',  activation_fn2='ReLU',  activation_fn3='ReLU',
                 latent_size=8):
        
        self.input_length = input_length
        self.input_channels = input_channels

        # Layer 1 configurations -------------------------------------------------------------------------------------------------------------------
        ## Input
        self.conv1_input_length = input_length
        self.conv1_input_channels = input_channels

        # Intermediate
        self.conv1_kernel_size = conv1_kernel_size
        self.conv1_stride = conv1_stride
        self.conv1_padding = (conv1_kernel_size - 1) // 2 # Symmetric padding
        self.conv1_activation_fn = self._get_activation(activation_fn1)

        # Output
        self.conv1_output_length = self._calculate_conv_output_length(self.conv1_input_length,
                                                                      k=self.conv1_kernel_size, s=self.conv1_stride, p=self.conv1_padding,
        )
        self.conv1_output_channels = conv1_output_channels

        # Layer 2 configurations -------------------------------------------------------------------------------------------------------------------
        ## Input
        self.conv2_input_length = self.conv1_output_length
        self.conv2_input_channels = self.conv1_output_channels

        # Intermediate
        self.conv2_kernel_size = conv2_kernel_size
        self.conv2_stride = conv2_stride
        self.conv2_padding = (conv2_kernel_size - 1) // 2 # Symmetric padding
        self.conv2_activation_fn = self._get_activation(activation_fn2)

        # Output
        self.conv2_output_length = self._calculate_conv_output_length(self.conv2_input_length,
                                                                      k=self.conv2_kernel_size, s=self.conv2_stride, p=self.conv2_padding,
        )
        self.conv2_output_channels = conv2_output_channels

        # Layer 3 configurations -------------------------------------------------------------------------------------------------------------------
        ## Input
        self.conv3_input_length = self.conv2_output_length
        self.conv3_input_channels = self.conv2_output_channels

        # Intermediate
        self.conv3_kernel_size = conv3_kernel_size
        self.conv3_stride = conv3_stride
        self.conv3_padding = (conv3_kernel_size - 1) // 2 # Symmetric padding
        self.conv3_activation_fn = self._get_activation(activation_fn3)

        # Output
        self.conv3_output_length = self._calculate_conv_output_length(self.conv3_input_length,
                                                                      k=self.conv3_kernel_size, s=self.conv3_stride, p=self.conv3_padding,
        )
        self.conv3_output_channels = conv3_output_channels

        # Latent space ------------------------------------------------------------------------------------------------------------------------------
        self.latent_size = latent_size

    def _calculate_conv_output_length(self, input_length, k, s, p):
        """Calculate the output length after 1 1D convolution."""
        return (input_length + 2 * p - k) // s + 1
    
    def _get_activation(self, activation_name):
        activations = {
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
            "Linear": nn.Identity(),
            "ELU": nn.ELU(),
            "SiLU": nn.SiLU(),
            "GELU": nn.GELU(),
        }
        return activations.get(activation_name, nn.Identity())
    
    def display_params(self):
        """Display configuration parameters."""
        print("ENCODER CONFIGURATIONS:")
        print("Layer 1:")
        print(f"Input Length: {self.input_length}")
        print(f"Input Channels: {self.input_channels}")
        print(f"Output Channels: {self.conv1_output_channels}")
        print(f"Output Length: {self.conv1_output_length}")
        print(f"Kernel Size: {self.conv1_kernel_size}")
        print(f"Stride: {self.conv1_stride}")
        print(f"Activation Function: {self.conv1_activation_fn}")
        print(f"Padding: {self.conv1_padding}")
        print("----------------------")
        print("Layer 2:")
        print(f"Input Length: {self.conv2_input_length}")
        print(f"Input Channels: {self.input_channels}")
        print(f"Output Channels: {self.conv2_output_channels}")
        print(f"Output Length: {self.conv2_output_length}")
        print(f"Kernel Size: {self.conv2_kernel_size}")
        print(f"Stride: {self.conv2_stride}")
        print(f"Activation Function: {self.conv2_activation_fn}")
        print(f"Padding: {self.conv2_padding}")
        print("----------------------")
        print("Layer 3:")
        print(f"Input Length: {self.conv3_input_length}")
        print(f"Input Channels: {self.conv3_input_channels}")
        print(f"Output Channels: {self.conv3_output_channels}")
        print(f"Output Length: {self.conv3_output_length}")
        print(f"Kernel Size: {self.conv3_kernel_size}")
        print(f"Stride: {self.conv3_stride}")
        print(f"Activation Function: {self.conv3_activation_fn}")
        print(f"Padding: {self.conv3_padding}")
        print("----------------------")
        print(f"Latent Size: {self.latent_size}")

    def display_in_out(self):
        print("ENCODER CONFIGURATIONS:")
        print("Layer 1:")
        print(f"Input Length: {self.conv1_input_length}")
        print(f"Input Channels: {self.conv1_input_channels}")
        print(f"Output Channels: {self.conv1_output_channels}")
        print(f"Output Length: {self.conv1_output_length}")
        print("----------------------")
        print("Layer 2:")
        print(f"Input length: {self.conv2_input_length}")
        print(f"Input Channels: {self.conv2_input_channels}")
        print(f"Output Channels: {self.conv2_output_channels}")
        print(f"Output Length: {self.conv2_output_length}")
        print("----------------------")
        print("Layer 3:")
        print(f"Input Length: {self.conv3_input_length}")
        print(f"Input Channels: {self.conv3_input_channels}")
        print(f"Output Channels: {self.conv3_output_channels}")
        print(f"Output Length: {self.conv3_output_length}")


# Decoder Configurations
class DecoderConfigSettings_3Layers:
    """
    Configuration class for a 3-layer Decoder model.
    Mirrors the Encoder configuration.
    """
    def __init__(self, encoder_config: EncoderConfigSettings_3Layers):
        # Mirror the encoder's latent size -----------------------------------------------------------------------------------------------
        self.latent_size = encoder_config.latent_size

        # Layer 3 (mirroring Encoder Layer 3) --------------------------------------------------------------------------------------------
        ## Input
        self.deconv3_input_length = encoder_config.conv3_output_length
        self.deconv3_input_channels = encoder_config.conv3_output_channels

        # Intermediate
        self.deconv3_kernel_size = encoder_config.conv3_kernel_size
        self.deconv3_stride = encoder_config.conv3_stride
        self.deconv3_padding = encoder_config.conv3_padding
        self.deconv3_output_padding = self._calculate_output_padding(
            desired_output_length=encoder_config.conv3_input_length,
            input_length=encoder_config.conv3_output_length,
            k=self.deconv3_kernel_size,
            s=self.deconv3_stride,
            p=self.deconv3_padding
        )
        self.deconv3_activation_fn = encoder_config.conv3_activation_fn

        # Output
        self.deconv3_output_length = encoder_config.conv3_input_length
        self.deconv3_output_channels = encoder_config.conv3_input_channels

        # Layer 2 (mirroring Encoder Layer 2) --------------------------------------------------------------------------------------------
        ## Input
        self.deconv2_input_length = encoder_config.conv2_output_length
        self.deconv2_input_channels = encoder_config.conv2_output_channels

        # Intermediate
        self.deconv2_kernel_size = encoder_config.conv2_kernel_size
        self.deconv2_stride = encoder_config.conv2_stride
        self.deconv2_padding = encoder_config.conv2_padding
        self.deconv2_output_padding = self._calculate_output_padding(
            desired_output_length=encoder_config.conv2_input_length,
            input_length=encoder_config.conv2_output_length,
            k=self.deconv2_kernel_size,
            s=self.deconv2_stride,
            p=self.deconv2_padding
        )
        self.deconv2_activation_fn = encoder_config.conv2_activation_fn

        # Output
        self.deconv2_output_length = encoder_config.conv2_input_length
        self.deconv2_output_channels = encoder_config.conv2_input_channels

        # Layer 1 (mirroring Encoder Layer 1) --------------------------------------------------------------------------------------------
        ## Input
        self.deconv1_input_length = encoder_config.conv1_output_length
        self.deconv1_input_channels = encoder_config.conv1_output_channels

        # Intermediate
        self.deconv1_kernel_size = encoder_config.conv1_kernel_size
        self.deconv1_stride = encoder_config.conv1_stride
        self.deconv1_padding = encoder_config.conv1_padding
        self.deconv1_output_padding = self._calculate_output_padding(
            desired_output_length=encoder_config.conv1_input_length,
            input_length=encoder_config.conv1_output_length,
            k=self.deconv1_kernel_size,
            s=self.deconv1_stride,
            p=self.deconv1_padding
        )
        self.deconv1_activation_fn = encoder_config.conv1_activation_fn

        # Output
        self.deconv1_output_length = encoder_config.conv1_input_length
        self.deconv1_output_channels = encoder_config.conv1_input_channels

    def _calculate_output_padding(self, desired_output_length, input_length, k, s, p):
        output_length = (input_length - 1) * s - 2 * p + k
        return max(0, desired_output_length - output_length)

    def display_params(self):
        """Display configuration parameters."""
        print("DECODER CONFIGURATIONS:")
        print(f"Latent Size: {self.latent_size}")
        print("----------------------")
        print("Layer 3 (Mirroring Encoder Layer 3):")
        print(f"Deconv Input Channels: {self.deconv3_input_channels}")
        print(f"Deconv Output Channels: {self.deconv3_output_channels}")
        print(f"Deconv Output Length: {self.deconv3_output_length}")
        print(f"Kernel Size: {self.deconv3_kernel_size}")
        print(f"Stride: {self.deconv3_stride}")
        print(f"Activation Function: {self.deconv3_activation_fn}")
        print(f"Padding: {self.deconv3_padding}")
        print(f"Output Padding: {self.deconv3_output_padding}")
        print("----------------------")
        print("Layer 2 (Mirroring Encoder Layer 2):")
        print(f"Deconv Input Channels: {self.deconv2_input_channels}")
        print(f"Deconv Output Channels: {self.deconv2_output_channels}")
        print(f"Deconv Output Length: {self.deconv2_output_length}")
        print(f"Kernel Size: {self.deconv2_kernel_size}")
        print(f"Stride: {self.deconv2_stride}")
        print(f"Activation Function: {self.deconv2_activation_fn}")
        print(f"Padding: {self.deconv2_padding}")
        print(f"Output Padding: {self.deconv2_output_padding}")
        print("----------------------")
        print("Layer 1 (Mirroring Encoder Layer 1):")
        print(f"Deconv Input Channels: {self.deconv1_input_channels}")
        print(f"Deconv Output Channels: {self.deconv1_output_channels}")
        print(f"Deconv Output Length: {self.deconv1_output_length}")
        print(f"Kernel Size: {self.deconv1_kernel_size}")
        print(f"Stride: {self.deconv1_stride}")
        print(f"Activation Function: {self.deconv1_activation_fn}")
        print(f"Padding: {self.deconv1_padding}")
        print(f"Output Padding: {self.deconv1_output_padding}")

    def display_in_out(self):
        print("DECODER CONFIGURATIONS:")
        print("Layer 1:")
        print(f"Input Length: {self.deconv1_input_length}")
        print(f"Input Channels: {self.deconv1_input_channels}")
        print(f"Output Channels: {self.deconv1_output_channels}")
        print(f"Output Length: {self.deconv1_output_length}")
        print("----------------------")
        print("Layer 2:")
        print(f"Input length: {self.deconv2_input_length}")
        print(f"Input Channels: {self.deconv2_input_channels}")
        print(f"Output Channels: {self.deconv2_output_channels}")
        print(f"Output Length: {self.deconv2_output_length}")
        print("----------------------")
        print("Layer 3:")
        print(f"Input Length: {self.deconv3_input_length}")
        print(f"Input Channels: {self.deconv3_input_channels}")
        print(f"Output Channels: {self.deconv3_output_channels}")
        print(f"Output Length: {self.deconv3_output_length}")
