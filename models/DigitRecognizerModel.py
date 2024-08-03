import torch.nn as nn


class DigitRecognizerMLPModel(nn.Module):
    def __init__(self, dim=28, num_classes=10, activation="relu"):
        super(DigitRecognizerMLPModel, self).__init__()
        self.layer1 = nn.Linear(in_features=dim * dim, out_features=512)
        self.layer2 = nn.Linear(in_features=512, out_features=256)
        self.layer3 = nn.Linear(in_features=256, out_features=128)
        self.layer4 = nn.Linear(in_features=128, out_features=num_classes)
        self.dropout = nn.Dropout(0.1)
        self.batchNorm1 = nn.BatchNorm1d(num_features=512)
        self.batchNorm2 = nn.BatchNorm1d(num_features=256)
        self.batchNorm3 = nn.BatchNorm1d(num_features=128)
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        else:
            raise Exception("Unknown Activation Function")
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        b, _, h, w = input.shape
        input = input.reshape(b, h * w)
        input = self.batchNorm1(self.activation(self.layer1(input)))
        input = self.batchNorm2(self.dropout(self.activation(self.layer2(input))))
        input = self.batchNorm3(self.dropout(self.activation(self.layer3(input))))
        input = self.layer4(input)

        return self.softmax(input)


class DigitRecognizerCNNModel(nn.Module):
    def __init__(self, dim=28, num_classes=10, activation="relu"):
        super(DigitRecognizerCNNModel, self).__init__()
        self.layer1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1
        )
        self.layer2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1
        )
        self.linear = nn.Linear(in_features=6 * 6 * 64, out_features=num_classes)
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        else:
            raise Exception("Unknown Activation Function")
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        b, _, _, _ = input.shape
        input = self.activation(self.layer1(input))
        input = self.activation(self.layer2(input))
        input = input.reshape(b, -1)
        input = self.linear(input)

        return self.softmax(input)


class DigitRecognizerLargeCNNModel(nn.Module):
    def __init__(self, dim=28, num_classes=10, activation="relu"):
        super(DigitRecognizerLargeCNNModel, self).__init__()
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        else:
            raise Exception("Unknown Activation Function")
        self.softmax = nn.LogSoftmax(dim=1)
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(num_features=32),
            self.activation,
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(num_features=64),
            self.activation,
            nn.Dropout2d(0.1),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(num_features=128),
            self.activation,
            nn.Dropout2d(0.1),
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            self.activation,
            nn.Dropout2d(0.1),
        )
        self.maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.linear_layer1 = nn.Sequential(
            nn.Linear(in_features=256 * 3 * 3, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            self.activation,
            nn.Dropout(0.1),
        )
        self.linear_layer2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(num_features=512),
            self.activation,
            nn.Dropout(0.1),
        )
        self.linear_layer3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(num_features=256),
            self.activation,
            nn.Dropout(0.1),
        )
        self.linear_layer4 = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(num_features=128),
            self.activation,
            nn.Dropout(0.1),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=128, out_features=num_classes), self.softmax
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            self.linear_layer1,
            self.linear_layer2,
            self.linear_layer3,
            self.linear_layer4,
            self.output_layer,
        )

    def forward(self, input):
        input = self.conv_layer1(input)
        input = self.maxpool_layer(input)
        input = self.conv_layer2(input)
        input = self.maxpool_layer(input)
        input = self.conv_layer3(input)
        input = self.maxpool_layer(input)
        input = self.conv_layer4(input)
        input = self.maxpool_layer(input)
        input = self.flatten(input)
        output = self.classifier(input)

        return output


if __name__ == "__main__":
    import torch

    rand_input = torch.randn(64, 1, 28, 28)
    model = DigitRecognizerMLPModel()
    output = model(rand_input)
    print("Dimension of Output = ", output.shape)

    rand_input = rand_input.reshape(64, 1, 28, 28)
    model = DigitRecognizerCNNModel()
    output = model(rand_input)
    print("Dimension of Output = ", output.shape)

    rand_input = rand_input.reshape(64, 1, 28, 28)
    model = DigitRecognizerLargeCNNModel()
    output = model(rand_input)
    print("Dimension of Output = ", output.shape)
