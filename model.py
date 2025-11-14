import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def get_unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1):
    """
    U-Net model from segmentation_models_pytorch.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,  
    )
    return model


# wrapper 
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, encoder_name='resnet34', encoder_weights='imagenet'):
        super().__init__()
        self.model = get_unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=n_classes
        )
    
    def forward(self, x):
        return self.model(x) # output are logits not sigmoid1