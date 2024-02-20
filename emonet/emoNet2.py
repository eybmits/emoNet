import torch.nn as nn
import torch.nn.functional as F

class emoNet(nn.Module):
    network_config = [64, 64, 'M', 64, 64, 'M', 128, 128, 'M']

    def __init__(self, num_of_channels, num_of_classes):
        super(emoNet, self).__init__()
        self.features = self._make_layers(num_of_channels, self.network_config)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1024),  # Erhöhte Layer-Größe
            nn.ELU(),  # ELU-Aktivierung
            nn.Dropout(0.5),  # Angepasste Dropout-Rate
            nn.Linear(1024, 512),  # Zusätzliche Layer
            nn.ELU(),  # ELU-Aktivierung
            nn.Dropout(0.5),  # Angepasste Dropout-Rate
            nn.Linear(512, num_of_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _make_layers(self, in_channels, cfg):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),  # Hinzugefügte BatchNorm
                           nn.ELU(inplace=True)]  # ELU-Aktivierung
                in_channels = x
        return nn.Sequential(*layers)