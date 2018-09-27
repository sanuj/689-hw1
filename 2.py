import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 8
n_letters = 1
n_categories = 1
learning_rate = 0.0005 # If you set this too high, it might explode. If too low, it might not learn
criterion = nn.MSELoss()
rnn = RNN(n_letters, n_hidden, n_categories)

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i].unsqueeze(0).unsqueeze(0), hidden)

    loss = criterion(output, category_tensor.unsqueeze(0).unsqueeze(0))
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

import time
import math

n_iters = 100

# Keep track of losses for plotting
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

import csv
train_data = []
with open('2train.csv') as train_file:
    reader = csv.reader(train_file)
    for row in reader:
        train_data.append(list(map(int, row)))

train_data = torch.tensor(train_data, dtype=torch.float)

for iter in range(1, n_iters + 1):
    curr_loss = 0
    for data in train_data:
        output, loss = train(data[3], data[:3])
        curr_loss += loss
    curr_loss /= len(train_data)
    all_losses.append(curr_loss)
    print("Iter: " + str(iter) + ", loss: " + str(curr_loss))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
print('Plotting figure.')
print(all_losses)
plt.plot(all_losses)
