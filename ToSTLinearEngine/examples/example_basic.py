import torch
import torch.optim as optim
from tostlinear.model import ToSTLinear
from tostlinear.utils import create_sample_data, train_model
#Setting this up for you to be as performant as possible.
input_size = 10
output_size = 5
sample_data = create_sample_data(input_size)
model = ToSTLinear(input_size, output_size)
targets = torch.randn(1, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, sample_data, targets, optimizer)
