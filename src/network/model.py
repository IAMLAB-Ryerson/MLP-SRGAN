import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math
from torchinfo import summary
from functools import partial
from einops.layers.torch import Rearrange, Reduce


class FeatureExtractor(nn.Module):
	def __init__(self):
		super(FeatureExtractor, self).__init__()
		vgg19_model = vgg19(pretrained=True)
		self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

	def forward(self, img):
		return self.vgg19_54(img)


class ResidualDenseBlock(nn.Module):
	def __init__(self, filters, res_scale=0.2):
		super(ResidualDenseBlock, self).__init__()
		self.res_scale = res_scale

		def resblock(in_features, non_linearity=True):
			layers = [nn.Conv2d(in_features, filters, kernel_size=3, stride=1, padding=1, bias=True)]
			if non_linearity:
				layers += [nn.LeakyReLU()]
			return(nn.Sequential(*layers))

		self.b1 = resblock(in_features=1 * filters)
		self.b2 = resblock(in_features=2 * filters)
		self.b3 = resblock(in_features=3 * filters)
		self.b4 = resblock(in_features=4 * filters)
		self.b5 = resblock(in_features=5 * filters, non_linearity=False)
		self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

	def forward(self, x):
		inputs = x
		for block in self.blocks:
			out = block(inputs)
			inputs = torch.cat([inputs, out], 1)
		return(out.mul(self.res_scale) + x)


class ResidualInResidualDenseBlock(nn.Module):
	def __init__(self, filters, res_scale=0.2):
		super(ResidualInResidualDenseBlock, self).__init__()
		self.res_scale = res_scale
		self.dense_blocks = nn.Sequential(
			ResidualDenseBlock(filters), 
			ResidualDenseBlock(filters),
			ResidualDenseBlock(filters)
		)

	def forward(self, x):
		return(self.dense_blocks(x).mul(self.res_scale) + x)


class PreNormResidual(nn.Module):
	def __init__(self, dim, fn):
		super().__init__()
		self.fn = fn
		self.norm = nn.LayerNorm(dim)

	def forward(self, x):
		return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		dense(dim, inner_dim),
		nn.GELU(),
		nn.Dropout(dropout),
		dense(inner_dim, dim),
		nn.Dropout(dropout)
	)


class MLPMixer(nn.Module):
	def __init__(self, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
		super(MLPMixer, self).__init__()
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
				PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
				PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
			) for _ in range(depth)],
			nn.LayerNorm(dim),
			Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size, h=8, w = 32, c = 3)
		)

	def forward(self, x):
		return(self.model(x))


class ResidualMLPMixer(nn.Module):
	def __init__(self, res_scale=0.2, n_mlp_blocks=1):
		super(ResidualMLPMixer, self).__init__()
		self.res_scale = res_scale
		self.mlp_blocks = nn.Sequential(
			*[MLPMixer(image_size=(64, 256), channels=3, patch_size=8, dim=192, depth=12, num_classes=1) for _ in range(n_mlp_blocks)]
		)

	def forward(self, x):
		return(self.mlp_blocks(x).mul(self.res_scale) + x)


class ResidualMLPInResidualDenseBlock(nn.Module):
	def __init__(self, res_scale=0.2, n_mlp_blocks=3):
		super(ResidualMLPInResidualDenseBlock, self).__init__()
		self.res_scale = res_scale
		self.mlp_blocks = nn.Sequential(
			*[ResidualMLPMixer() for _ in range(n_mlp_blocks)]
		)

	def forward(self, x):
		return(self.mlp_blocks(x).mul(self.res_scale) + x)


class GeneratorMixer(nn.Module):
	def __init__(self, in_channels=3, out_channels=3, filters=64, n_residual_blocks=1):
		super(GeneratorMixer, self).__init__()

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

	def forward(self, x):
		# Test 1
		# out1 = self.conv1(x)
		# out = self.res_blocks(x)
		# out2 = self.conv2(out)
		# out = torch.add(out1, out2)
		# out = self.upsampling(out)
		# out = self.conv3(out)
		# return(out)

		# Test 2
		out = self.res_blocks(x)
		out = self.conv2(out)
		out = self.upsampling(out)
		out = self.conv3(out)
		return(out)

		# Test 3
		# out = self.res_blocks(x)
		# out = self.conv2(out)
		# out = self.upsampling(out)
		# out = self.conv3(out)
		# return(out)

class GeneratorResNet(nn.Module):
	def __init__(self, in_channels=3, out_channels=3, filters=64, n_residual_blocks=23):
		super(GeneratorResNet, self).__init__()

		# First Layer
		self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, stride=1, padding=1)

		# Residual Blocks
		self.res_blocks = nn.Sequential(
			*[ResidualInResidualDenseBlock(filters) for _ in range(n_residual_blocks)]
		)

		# Second Conv Layer
		self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

		# Upsampling Layers
		scale_factor = 2
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
			nn.Conv2d(filters, filters, kernel_size=5, stride=(2,1), padding=2),
			nn.LeakyReLU(),
			nn.Conv2d(filters, filters, kernel_size=5, stride=(2,1), padding=2),
			nn.LeakyReLU(),
			nn.Conv2d(filters, out_channels, kernel_size=3, stride=1, padding=1)
		)

	def forward(self, x):
		out1 = self.conv1(x)
		out = self.res_blocks(out1)
		out2 = self.conv2(out)
		out = torch.add(out1, out2)
		out = self.upsampling(out)
		out = self.conv3(out)
		return(out)


# ADD HIGH FREQUENCY DISCRIMINATOR

class Discriminator(nn.Module):
	def __init__(self, input_shape):
		super(Discriminator, self).__init__()

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
			return(layers)

		layers = []
		in_filters = in_channels
		for i, out_filters in enumerate([64, 128, 256, 512]):
			layers.extend(discriminator_block(in_filters, out_filters, first_block=(i==0)))
			in_filters = out_filters

		layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

		self.model = nn.Sequential(*layers)

	def forward(self, img):
		return(self.model(img))


# mixer = MLPMixer(image_size=(256, 64), channels=3, patch_size=8, dim=192, depth=12, num_classes=1)
# batch_size = 8
# summary(mixer, input_size=(batch_size, 3, 256, 64))

# mixer = ResidualMLPMixer()
# batch_size = 8
# summary(mixer, input_size=(batch_size, 3, 256, 64))

# G_model = GeneratorResNet()
# batch_size = 8
# summary(G_model, input_size=(batch_size, 3, 256, 64))

G_model = GeneratorMixer()
batch_size = 8
summary(G_model, input_size=(batch_size, 3, 64, 256))

# D_model = Discriminator(input_shape=(3, 256, 256))
# batch_size = 64
# summary(D_model, input_size=(batch_size, 3, 256, 256))