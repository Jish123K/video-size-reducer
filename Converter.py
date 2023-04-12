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
', 'horizontal aspect', 'vertical aspect']

)

normalized_input_data = normalize_dataframe(input_data, self.train_stats)

return int(round(self.model.predict(normalized_input_data)[0][0]))
def save_model(self, filename):

    self.model.save(filename)

def load_model(self, filename):

    self.model = keras.models.load_model(filename)

class VideoSizer:

def init(self):

self.load_data()

self.build_model()
  def build_model(self):

    self.model = keras.Sequential([

        layers.Dense(64, activation=tf.nn.relu, input_shape=self.input_shape),

        layers.Dense(32, activation=tf.nn.relu),

        layers.Dense(16, activation=tf.nn.relu),

        layers.Dense(8, activation=tf.nn.relu),

        layers.Dense(4, activation=tf.nn.relu),

        layers.Dense(2, activation=tf.nn.relu),

        layers.Dense(1)

    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    self.model.compile(loss=tf.losses.mean_squared_error, optimizer=optimizer,

                       metrics=[tf.losses.mean_squared_error, 'mean_absolute_error'])

def load_data(self):

    ds = pd.read_csv('dataset/dataset.csv')

    if ds['bitrate'].count() < 16:

        raise dataset.InadequateDataError('Not enough data, please run ./generate_data.py')

    test_dataset = ds.sample(16)

    self.test_bitrate = test_dataset.pop('bitrate')

    train_dataset = ds.drop(test_dataset.index)

    self.train_bitrate = train_dataset.pop('bitrate')

    self.train_stats = train_dataset.describe()

    self.input_shape = (len(train_dataset.keys()),)

    self.normalized_train_dataset = normalize_dataframe(train_dataset, self.train_stats)

    self.normalized_test_dataset = normalize_dataframe(test_dataset, self.train_stats)

def train(self, epochs=2000):

    return self.model.fit(self.normalized_train_dataset, self.train_bitrate, epochs=epochs)

def test(self):

    result = self.model.evaluate(self.normalized_test_dataset, self.test_bitrate)

    print(self.model.metrics_names)

    print(result)

def estimate(self, target_size, video_length, h_aspect, v_aspect):

    input_data = pd.DataFrame(

        data={

            'output size in bytes': np.array([target_size]),

            'length in seconds': np.array([video_length]),

            'horizontal aspect': np.array([h_aspect]),

            'vertical aspect': np.array([v_aspect])

        },

        columns=['output size in bytes', 'length in seconds', 'horizontal aspect', 'vertical aspect']

    )

    normalized_input_data = normalize_dataframe(input_data, self.train_stats)

    return int(round(self.model.predict(normalized_input_data)[0][0]))

def save_model(self, filename):

    self.model.save(filename)

def load_model(self, filename):

    self.model = keras.models.load_model(filename)

                   if name == 'main':

description = ('Make a video fit a specified file size restriction')

parser = argparse.ArgumentParser(description=description)

parser.add_argument('filename', help='path to input video file', type=str)

parser.add_argument('size', help='target file size in bytes', type=int)

parser.add_argument('-o', '--output', help='output file path', type=str, default=None, required=False)

parser.add_argument('-e', '--epochs', help='number of epochs for training', type=int, default=2000, required=False)

parser.add_argument('-d', '--dont-remember', help="don't remember the results of this transcode", type=bool, default=False, required=False)

parser.add_argument('-m', '--model', help='machine learning model to use', type=str, default='keras', required=False, choices=['keras', 'pytorch', 'scikit'])

arguments = parser.parse_args()
# Load video metadata

video_length, h_aspect, v_aspect = get_video_metadata(arguments.filename)

# Instantiate the appropriate ML model

if arguments.model == 'keras':

    model = KerasVideoSizeEstimator()

elif arguments.model == 'pytorch':

    model = PytorchVideoSizeEstimator()

elif arguments.model == 'scikit':

    model = ScikitVideoSizeEstimator()

else:

    raise ValueError('Invalid model type specified')

# Train the model on the dataset

model.train(arguments.epochs)

# Test the model

model.test()

# Estimate the bitrate required to achieve the target file size

estimated_bitrate = model.estimate(arguments.size, video_length, h_aspect, v_aspect)

# Print the estimated bitrate

print(f"Estimated bitrate: {estimated_bitrate} kbps")

# Transcode the video to the estimated bitrate

output_path, actual_size = transcode_video(arguments.filename, arguments.output, estimated_bitrate)

# Check if the actual file size is larger than the target file size

if actual_size > arguments.size:

    print("Actual file size is larger than target file size!")

# Store the transcoding results in the dataset

if not arguments.dont_remember:

    store_transcoding_results(actual_size, video_length, estimated_bitrate, h_aspect, v_aspect)

