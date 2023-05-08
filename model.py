from imports import *


if not os.path.exists('models'):
    os.mkdir('models')


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 1)
        self.conv3 = nn.Conv2d(64, 128, 4, 1)
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(1104000, 100)
        self.bn1 = nn.BatchNorm1d(100, eps=0.001, momentum=0.99)
        self.linear2 = nn.Linear(100, 50)
        self.bn2 = nn.BatchNorm1d(50, eps=0.001, momentum=0.99)
        self.linear3 = nn.Linear(50, 1)


    def forward(self, x):
        x = x.float()

        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = F.relu(self.conv3(x))
        #print(x.shape)

        x = self.flatten(x)

        #print(x.shape)

        x = self.bn1(F.relu(self.linear1(x)))
        x = self.bn2(F.relu(self.linear2(x)))
        x = self.linear3(x)

        return torch.sigmoid(x)

