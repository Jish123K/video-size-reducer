import argparse

import pandas as pd

import numpy as np

import os

import joblib

import torch

import torch.nn as nn

import torch.optim as optim

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error

import ffmpeg

class VideoSizer():

    def __init__(self):

        self.load_data()

        self.build_model()

    def build_model(self):

        self.model = nn.Sequential(

            nn.Linear(len(self.train_dataset.columns) - 1, 64),

            nn.ReLU(),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, 16),

            nn.ReLU(),

            nn.Linear(16, 8),

            nn.ReLU(),

            nn.Linear(8, 4),

            nn.ReLU(),

            nn.Linear(4, 2),

            nn.ReLU(),

            nn.Linear(2, 1)

        )

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001)

        self.loss_fn = nn.MSELoss()

    def load_data(self):

        ds = pd.read_csv('dataset/dataset.csv')

        if ds['bitrate'].count() < 16:

            raise ValueError('Not enough data, please run ./generate_data.py')

        test_dataset = ds.sample(16)

        self.test_bitrate = test_dataset.pop('bitrate')

        train_dataset = ds.drop(test_dataset.index)

        self.train_bitrate = train_dataset.pop('bitrate')

        self.train_scaler = StandardScaler().fit(train_dataset)

        self.input_shape = (len(train_dataset.columns),)

        self.normalized_train_dataset = pd.DataFrame(

            self.train_scaler.transform(train_dataset),

            columns=train_dataset.columns

        )

        self.normalized_test_dataset = pd.DataFrame(

            self.train_scaler.transform(test_dataset),

            columns=test_dataset.columns

        )

    def train(self, epochs=2000):

        for epoch in range(epochs):

            self.model.train()

            self.optimizer.zero_grad()

            inputs = torch.Tensor(self.normalized_train_dataset.values)

            targets = torch.Tensor(self.train_bitrate.values.reshape(-1, 1))

            outputs = self.model(inputs)

            loss = self.loss_fn(outputs, targets)

            loss.backward()

            self.optimizer.step()

            if epoch % 100 == 0:

                print(f'Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}')

    def test(self):

        self.model.eval()

        inputs = torch.Tensor(self.normalized_test_dataset.values)

        targets = torch.Tensor(self.test_bitrate.values.reshape(-1, 1))

        outputs = self.model(inputs)

        mse = mean_squared_error(targets, outputs)

        mae = mean_absolute_error(targets, outputs)

        print(f'Test MSE: {mse:.4f}')

        print(f'Test MAE: {mae:.4f}')

    def save(self, path):

        joblib.dump(self.train_scaler, f'{path}/scaler.joblib')

        torch.save(self.model.state_dict(), f'{path}/model.pt')

    def estimate(self, target_size, video_length, h_aspect, v_aspect):

        input_data = pd.DataFrame(

            data={

                'output size in bytes': np.array([target_size]),

                'length in seconds': np.array([video_length]),

                'horizontal aspect': np.array([h_aspect]),

                'vertical aspect': np.array([v_aspect])

            },

            columns=['output size in bytes', 'length in seconds

