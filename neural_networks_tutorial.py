

"""
A typical training procedure for a neural network:
	Define the neural network
	Iterate over a dataset of inputs
	Process input through the network
	Compute the loss (how far is the output from being correct)
	Propagate gradients back into the network's parameters
	Update weights (weight = weight - learning_rate * gradient)
"""

# DEFINE NET ARCHITECTURE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		# 1 input image channel, 6 output channels, 5x5 square convolution kernel
		self.conv1 = nn.Conv2d(1,6,5)
		self.conv2 = nn.Conv2d(6,16,5)
		# An affine operation: y = Wx + b
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		# Max pooling over a (2,2) window
		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
		# If the size is a square you can only specify a single number 
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	# Backward function is automatically defined using autograd

	def num_flat_features(self, x):
		size = x.size()[1:] # All dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

net = Net()
print(net)

params = list(net.parameters()) # The learnable parameters of a model
print("Lenght: ", len(params))

#for i in range(0, len(params)-1):
	#print(params[i].size()) layer i's weight

# PROCESSING INPUTS AND CALLING BACKWARD
input = torch.randn(1, 1, 32, 32) # Random input (standar normal distribution)
out = net(input)
print("Output: ", out)

net.zero_grad() # Zero the gradient buffers of all parameters
out.backward(torch.randn(1,10)) # backprops with random gradients

# nn only supports inputs that are a mini-batch of samples
# e.g. nn.Covn2d will take in a 4D tensor (nSamples x nChannels x Height x Width)
# If you hace a single sample, se input.unsqueeze(0) to add a fake batch dimension


# COMPUTE LOSS
output = net(input)
target = torch.arange(1, 11) # Dummy target
target = target.view(1, -1) # Make it the same shape as output
criterion = nn.MSELoss() # Mean-Squared error
loss = criterion(output, target) # (y, ý)
print("Loss: ",loss) 

# PROPAGATE GRADIENTS BACK INTO THE NETWORK PARAMETERS
net.zero_grad()  # Clear the existing gradients though, else gradients will be 
				 # accumulated to existing gradients. 

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward() # backpropagate the error

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# UPDATING WEIGHTS
optimizer = optim.SGD(net.parameters(), lr=0.01) 
"""
SGD implementation
learning_rate = 0.01
for f in net.parameters():
	f.data.sub_(f.grad.data * learning_rate)
"""
# In the training loop
optimizer.zero_grad() # Manually set to zero gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # Does the update 