import torch.nn as nn
import torch.nn.functional as F


BOARD_LENGTH = 8
BOARD_VECTOR_DEPTH = 12


# Double head architecture
class Alpha(nn.Module):
    def __init__(self):
        super(Alpha, self).__init__()

        # Shared body
        self.conv1 = nn.Conv2d(BOARD_VECTOR_DEPTH, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.dr1 = nn.Dropout()
        self.dr2 = nn.Dropout()

        # Policy head

        # Value head

    def forward(self, x):
        # x is assumed to be (batch_size, BOARD_VECTOR_DEPTH, BOARD_LENGTH, BOADR_LENGTH)

        # Common body
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.dr1(self.fc1(x)))
        x = F.relu(self.dr2(self.fc2(x)))

        # Policy head

        # Value head


def main():
    print("Hello from alpha-1!")


if __name__ == "__main__":
    main()
