from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50
import torch.nn as nn
import torch

class FCN(nn.Module):
    def __init__(self, n_class=21):
        super(FCN, self).__init__()
        self.n_class = n_class
        self.fcn = fcn_resnet50(pretrained=False, num_classes=self.n_class)

    def forward(self, x):
        return self.fcn(x)['out']

class deeplabv3(nn.Module):
    def __init__(self, n_class=21):
        super(deeplabv3, self).__init__()
        self.n_class = n_class
        self.fcn = deeplabv3_resnet50(pretrained=False, num_classes=self.n_class)

    def forward(self, x):
        return self.fcn(x)['out']

if __name__ == '__main__':
    FCN = deeplabv3(2)
    # summary(model, (3, 512, 512), device="cpu")
    x = torch.randn(64, 3, 96, 96)
    out = FCN(x)
    print(out.shape)
