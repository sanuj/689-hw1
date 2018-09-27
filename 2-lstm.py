import torch
import torch.nn as nn
import numpy as np
import csv
import visdom


class LSTMReg(nn.Module):
    def __init__(self, inp_size, hid_size, out_size):
        super(LSTMReg, self).__init__()
        self.rnn = nn.LSTM(inp_size, hid_size)
        self.res = nn.Linear(hid_size*2, out_size)

    def forward(self, inp, hid):
        _, hid = self.rnn(inp, hid)
        out = self.res(torch.cat([hid[0].view(1,-1), hid[1].view(1, -1)], 1))
        return out, hid


def train(out_no, in_nos):
    hidden = torch.randn(2, 1, 1, n_hid)
    lstm.zero_grad()

    for i in range(in_nos.size()[0]):
        out, hidden = lstm(in_nos[i].view(1, 1, -1), hidden)
    
    loss = criterion(out, out_no.view(1, 1))
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in lstm.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return out, loss.item()


def get_data(name):
    train_data = []
    with open(name) as train_file:
        reader = csv.reader(train_file)
        for row in reader:
            train_data.append(list(map(int, row)))

    return torch.tensor(train_data, dtype=torch.float)


def main():
    train_data = get_data('2train.csv')
    test_data = get_data('2test.csv')
    vis = visdom.Visdom()
    loss_plot = None

    for epoch in range(1, epochs + 1):
        curr_loss = 0
        for data in train_data:
            output, loss = train(data[3], data[:3])
            curr_loss += loss
        curr_loss /= len(train_data)

        print("Iter: " + str(epoch) + ", loss: " + str(curr_loss))
        if loss_plot is None:
            loss_plot = vis.line(Y=np.array([curr_loss]), X=np.array([epoch]), win=loss_plot)
        else:
            vis.line(Y=np.array([curr_loss]), X=np.array([epoch]), win=loss_plot, update='append')

        if epoch % 10 == 0:
            for data in test_data:
                out_no, in_nos = data[3], data[:3]
                hidden = torch.randn(2, 1, 1, n_hid)
                for i in range(in_nos.size()[0]):
                    out, hidden = lstm(in_nos[i].view(1, 1, -1), hidden)
                loss = criterion(out, out_no.view(1, 1))
                print(data, out, loss)


if __name__ == '__main__':
    epochs = 1000
    n_hid = 64
    n_in_no = 1
    n_out_no = 1
    learning_rate = 0.01  # If you set this too high, it might explode. If too low, it might not learn.
    criterion = nn.MSELoss()
    lstm = LSTMReg(n_in_no, n_hid, n_out_no)
    main()
