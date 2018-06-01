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
		self.fc1 = torch.nn.Linear(4 * 4 * 16, 120)
		self.fc2 = torch.nn.Linear(120, 84)
		self.fc3 = torch.nn.Linear(84, 10)

	def forward(self, x):
		print (x.size())
		x = self.mxPool(F.relu(self.conv1(x)))
		print (x.size())
		x = self.mxPool(F.relu(self.conv2(x)))
		print (x.size())
		x = x.resize(x.size()[0], 4 * 4 * 16)
		# some thing more to do with x
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		return x
