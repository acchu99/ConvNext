from torch import nn
from timm import create_model


class ConvNeXtModelOfficial(nn.Module):
    def __init__(self, n_labels, model_name, pretrained=True):
        super(ConvNeXtModelOfficial, self).__init__()
        self.convnext_model = create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=n_labels
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.convnext_model(x)
        x = self.sigmoid(x)
        return x