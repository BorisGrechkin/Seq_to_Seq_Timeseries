'''LSTM Model'''
import os
from os import environ

import torch
import torch.nn as nn
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.env')
load_dotenv(env_path)

input_size = int(environ.get('INPUT_SIZE'))
hidden_size = int(environ.get('HIDDEN_SIZE'))
output_size = int(environ.get('OUTPUT_SIZE'))
num_layers = int(environ.get('NUM_LAYERS'))
teacher_forcing_ratio = float(environ.get('TEACHER_FORCING_RATIO'))

torch.manual_seed(42)
torch.cuda.manual_seed(42)

class LSTMModel(nn.Module):


    def __init__(self, device):
        super(LSTMModel, self).__init__()

        self.device = device

        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fully_connected = nn.Linear(hidden_size, output_size)


    def forward(self, x, y=None):
        batch_size = x.size(0)
        seq_length = x.size(1)

        _, (hidden, cell) = self.encoder(x)

        outputs = torch.zeros(batch_size, seq_length, 1).to(self.device)
        dec_input = torch.zeros(batch_size, 1, 1).to(self.device)

        for time_step_index in range(seq_length):

            dec_output, (hidden, cell) = self.decoder(dec_input, (hidden, cell))
            output = self.fully_connected(dec_output.squeeze(1))
            outputs[:, time_step_index] = output

            if y is not None and torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = y[:, time_step_index].unsqueeze(1)
            else:
                dec_input = output.unsqueeze(1)

        return outputs