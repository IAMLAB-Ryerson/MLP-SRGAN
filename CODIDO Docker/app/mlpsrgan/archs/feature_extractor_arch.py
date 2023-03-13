from torchvision.models import vgg19
from torch import nn

debug = False
if debug is True:
    from torchinfo import summary

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)

if debug is True:
    F_model = FeatureExtractor()
    batch_size = 8
    summary(F_model, input_size=(batch_size, 3, 256, 256))
