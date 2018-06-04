import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitRecNet(nn.Module):
	def __init__(self):
		super(DigitRecNet, self).__init__()
		self.conv1 = torch.nn.Conv2d(in_channels=1,
									 out_channels=6,
									 kernel_size=5)
		self.conv2 = torch.nn.Conv2d(in_channels=6,
									 out_channels=16,
									 kernel_size=5)
		self.mxPool = torch.nn.MaxPool2d(kernel_size=2)

		self.output_size = 4 * 4 * 16
		self.fc1 = torch.nn.Linear(4 * 4 * 16, 120)
		self.fc2 = torch.nn.Linear(120, 84)
		self.fc3 = torch.nn.Linear(84, 10)
		self.fc = torch.nn.Linear(20 * 4 * 4, 10)

	def forward(self, x):
		x = self.mxPool(F.relu(self.conv1(x)))
		x = self.mxPool(F.relu(self.conv2(x)))
		x = x.resize(x.size()[0], self.output_size)
		# some thing more to do with x
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
#x = self.fc(x)
		return F.log_softmax(x)
