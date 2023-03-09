"""Module model contains classes to construct the MLP-SRGAN."""
from functools import partial
from torch import nn
from torchvision.models import vgg19
from einops.layers.torch import Rearrange


class FeatureExtractor(nn.Module):
    """
    A feature extraction network containing layers from VGG19.

    ...

    Methods
    -------
    forward(img):
        Network forward pass.
    """
    def __init__(self):
        super().__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        '''
        Network forward pass.
        '''
        return self.vgg19_54(img)


class PreNormResidual(nn.Module):
    """
    A prenormalization residual connection for use in MLP-Mixer.

    ...

    Methods
    -------
    forward(vect):
        Network forward pass.
    """
    def __init__(self, dim, func):
        super().__init__()
        self.func = func
        self.norm = nn.LayerNorm(dim)

    def forward(self, vect):
        '''
        Network forward pass.
        '''
        return self.func(self.norm(vect)) + vect


def feed_forward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    '''
    A feed forward NN for use in MLP-Mixer, contains GELU and Dropout.

            Parameters:
                    dim (int): dimension
                    expansion_factor (int): expansion factor expand dim
                    dropout (float): dropout threshold value
                    dense (torch.nn): type of feed forward nn

            Returns:
                   Sequential combination of nn layers
    '''
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


class MLPMixer(nn.Module):
    """
    MLP-Mixer network block.

    ...

    Attributes
    ----------
    image_size : int, tuple
        size of image dimensions as an integer tuple
    channels : int
        number of channels in image
    patch_size : int
        image patch size for feed forward NN
    dim: int
        dimension
    depth: int
        MLP_Mixer block depth
    num_classes: int
        number of classes to classify in MLP-Mixer block
    expansion_factor: int
        expansion factor expand dim
    expansion_factor_token: int
        expansion factor expand dim
    dropout: float
        dropout threshold value

    Methods
    -------
    forward(vect):
        Network forward pass.
    """
    def __init__(self, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_classes = num_classes
        self.expansion_factor = expansion_factor
        self.expansion_factor_token = expansion_factor_token
        self.dropout = dropout

        image_h = image_size[0]
        image_w = image_size[1]
        assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_h // patch_size) * (image_w // patch_size)
        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

        self.model = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear((patch_size ** 2) * channels, dim),
            *[nn.Sequential(
                PreNormResidual(dim, feed_forward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, feed_forward(dim, expansion_factor_token, dropout, chan_last))
            ) for _ in range(depth)],
            nn.LayerNorm(dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size, h=8, w = 32, c = 3)
        )

    def forward(self, vect):
        '''
        Network forward pass.
        '''
        return self.model(vect)


class ResidualMLPMixer(nn.Module):
    """
    Residual MLP-Mixer network block. Contains residual connections

    ...

    Attributes
    ----------
    res_scale : float
        residual connection stength
    n_mlp_blocks : int
        number of MLP-Mixer blocks in the Residual MLP-Mixer

    Methods
    -------
    forward(vect):
        Network forward pass.
    """
    def __init__(self, res_scale=0.2, n_mlp_blocks=1):
        super().__init__()
        self.res_scale = res_scale
        self.mlp_blocks = nn.Sequential(
            *[MLPMixer(image_size=(64, 256), channels=3, patch_size=8, dim=192, depth=12, num_classes=1) for _ in range(n_mlp_blocks)]
        )

    def forward(self, vect):
        '''
        Network forward pass.
        '''
        return self.mlp_blocks(vect).mul(self.res_scale) + vect


class ResidualMLPInResidualDenseBlock(nn.Module):
    """
    Residual MLP-Mixer in Residual Dense Block network block. Contains more residual connections

    ...

    Attributes
    ----------
    res_scale : float
        residual connection stength
    n_mlp_blocks : int
        number of MLP-Mixer blocks in the Residual MLP-Mixer

    Methods
    -------
    forward(vect):
        Network forward pass.
    """
    def __init__(self, res_scale=0.2, n_mlp_blocks=3):
        super().__init__()
        self.res_scale = res_scale
        self.mlp_blocks = nn.Sequential(
            *[ResidualMLPMixer() for _ in range(n_mlp_blocks)]
        )

    def forward(self, vect):
        '''
        Network forward pass.
        '''
        return self.mlp_blocks(vect).mul(self.res_scale) + vect


class GeneratorMixer(nn.Module):
    """
    Full Generator network

    ...

    Attributes
    ----------
    in_channels : int
        number of input image channels
    out_channels : int
        number of output image channels
    filters : int
        number of CNN filters
    n_residual_blocks : int
        number of RMRDBs

    Methods
    -------
    forward(vect):
        Network forward pass.
    """
    def __init__(self, in_channels=3, out_channels=3, filters=64, n_residual_blocks=1):
        super().__init__()

        # First Layer
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, stride=1, padding=1)

        # Residual Blocks
        self.res_blocks = nn.Sequential(
            *[ResidualMLPInResidualDenseBlock() for _ in range(n_residual_blocks)]
        )

        # Second Conv Layer
        self.conv2 = nn.Conv2d(in_channels, filters, kernel_size=3, stride=1, padding=1)

        # Upsampling Layers
        scale_factor=2
        upsampling_layers = []
        for _ in range(scale_factor):
            upsampling_layers += [
                nn.Conv2d(filters, 4*filters, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2)
            ]
        self.upsampling = nn.Sequential(*upsampling_layers)

        # Final output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=5, stride=(1,2), padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=5, stride=(1,2), padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(filters, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, vect):
        '''
        Network forward pass.
        '''
        out = self.res_blocks(vect)
        out = self.conv2(out)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Module):
    """
    Discriminator network.

    ...

    Attributes
    ----------
    input_shape : int, tuple
        input image size as a tuple

    Methods
    -------
    forward(img):
        Network forward pass.
    """
    def __init__(self, input_shape):
        super().__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i==0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        '''
        Network forward pass.
        '''
        return self.model(img)
