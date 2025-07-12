import torch
import torch.nn as nn
import torch.nn.functional as F

BOARD_LENGTH = 8
BOARD_VECTOR_DEPTH = 12
POLICY_OUTPUT_SIZE = 4672  # You can change this based on move encoding


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

        # Policy head
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, POLICY_OUTPUT_SIZE)
        self.policy_dr = nn.Dropout()

        # Value head
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.value_dr1 = nn.Dropout()
        self.value_dr2 = nn.Dropout()

    def forward(self, x):
        # Shared body
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x_conv = F.relu(self.bn4(self.conv4(x)))  # Save for heads

        # Policy head
        x_policy = F.relu(self.policy_bn(self.policy_conv(x_conv)))
        x_policy = x_policy.view(-1, 2 * 8 * 8)
        x_policy = self.policy_fc(self.policy_dr(x_policy))
        x_policy = F.log_softmax(x_policy, dim=1)

        # Value head
        x_value = F.relu(self.value_bn(self.value_conv(x_conv)))
        x_value = x_value.view(-1, 8 * 8)
        x_value = F.relu(self.value_fc1(self.value_dr1(x_value)))
        x_value = torch.tanh(self.value_fc2(self.value_dr2(x_value)))

        return x_policy, x_value


def main():
    print("Hello from alpha-1!")


if __name__ == "__main__":
    main()
