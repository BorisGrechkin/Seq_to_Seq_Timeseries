import os
from os import environ

import torch
import torch.nn as nn
from dotenv import load_dotenv

from Model.LSTM import LSTMModel
from Scripts.visualization import Visualization

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.env')
load_dotenv(env_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training using device:', device)

learning_rate = float(environ.get('LEARNING_RATE'))
num_epochs = int(environ.get('NUM_EPOCHS'))
model_save_path = environ.get('MODEL_SAVE_PATH')
img_save_path = environ.get('IMG_SAVE_PATH')

model = LSTMModel(device).to(device)

def train_model(train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_losses = []
        val_losses = []

        for x_data, y_data in train_loader:
            x_data = x_data.to(device)
            y_data = y_data.to(device)

            optimizer.zero_grad()
            outputs = model(x_data, y_data)
            loss = criterion(outputs, y_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_data, y_data in val_loader:
                x_data = x_data.to(device)
                y_data = y_data.to(device)

                outputs = model(x_data)
                val_loss += criterion(outputs, y_data).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    print(f'Best Loss: {best_loss:.6f}')
    print(f'Successfully completed learning!')
    Visualization.save_learning_loss(train_losses, val_losses, img_save_path)


