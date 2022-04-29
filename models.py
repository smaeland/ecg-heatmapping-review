import torch
import torch.nn as nn
import torch.nn.functional as F



class ResModule(nn.Module):

    def __init__(
        self,
        in_channels,
        conv1_filters=64,
        conv2_filters=32,
        conv1_size=50,
        conv2_size=50,
        sum_with_input=True
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=conv1_filters,
            kernel_size=conv1_size,
            padding='same'
        )
        self.bn1 = nn.BatchNorm1d(num_features=conv1_filters)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()  # need two separate ones for DeepLIFT
        self.conv2 = nn.Conv1d(
            in_channels=conv1_filters,
            out_channels=conv2_filters,
            kernel_size=conv2_size,
            padding='same'
        )
        self.bn2 = nn.BatchNorm1d(num_features=32)

        self.sum_with_input = sum_with_input


    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)       # TODO relu before add?
        
        if self.sum_with_input:
            out = torch.add(x, out)

        return out




class StevenNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.init_block = ResModule(12, conv1_size=8, conv2_size=3, sum_with_input=False)
        self.block1 = ResModule(32)
        self.block2 = ResModule(32)
        self.block3 = ResModule(32)
        self.block4 = ResModule(32)
        self.block5 = ResModule(32)
        self.block6 = ResModule(32)
        self.block7 = ResModule(32)
        self.block8 = ResModule(32)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32, 1)
    
    def forward(self, x):

        out = self.init_block(x)
        out = nn.AvgPool1d(kernel_size=2)(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc1(out)

        return out


class StandardConvNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=32, kernel_size=5, padding='same'),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, padding='same'),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding='same'),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_features=32, out_features=1)
        )
    
    def forward(self, x):
        return self.model(x)

