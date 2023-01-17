

import nest_asyncio
nest_asyncio.apply()

# importing required libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_federated as tff
import seaborn as sns
import scipy.stats as stats
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from tensorflow import keras
from keras.models import load_model
from keras.models import clone_model
from tensorflow.keras.layers import Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Dense,Dropout,LSTM
# % matplotlib inline

df = pd.read_csv('Combined_Data_21062022.csv')
df.head()

"""## **Data Pre-processing**"""

# giving unique ID to each user
df['user_id'] = np.where(df['user_id'] == 'Darshan', 1, df['user_id'])
df['user_id'] = np.where(df['user_id'] == 'Pankhuri', 2, df['user_id'])
df['user_id'] = np.where(df['user_id'] == 'Talib', 3, df['user_id'])
df['user_id'] = np.where(df['user_id'] == 'Tamanna', 4, df['user_id'])

# dropping timestamp feature
df.drop(columns=['timestamp'], axis=1, inplace=True)

# calculate stats information
def get_stats(series):
  min = np.min(series)
  max = np.max(series)
  mean = np.mean(series)
  median = np.median(series)
  std = np.std(series)
  skew = stats.skew(series)
  kurtosis = stats.kurtosis(series)
  diff = max - min
  IQR = np.percentile(series, 75) - np.percentile(series, 25)
  pos_count = np.sum(np.array(series) >= 0)
  neg_count = np.sum(np.array(series) < 0)
  above_mean = np.sum(np.array(series) > np.mean(series))
  avg_abs_diff = np.mean(np.absolute(series - np.mean(series)))
  energy = sum(i*i for i in series)/100
  return min, max, mean, median, std, skew, kurtosis, diff, IQR, pos_count, neg_count, above_mean, avg_abs_diff, energy

# setting sliding window parameters
frame_size = 100
hop_size = 50

# defining sliding window
def get_frames(df, frame_size, hop_size):
  frames = []
  labels = []

  for i in range(0, len(df) - frame_size, hop_size):
    #temp = []
    # Raw sensor readings from Empatica watch
    w_acc_x = df['w_acc_x'].values[i: i + frame_size]
    w_acc_y = df['w_acc_y'].values[i: i + frame_size]
    w_acc_z = df['w_acc_z'].values[i: i + frame_size]
    w_bvp = df['w_bvp'].values[i: i + frame_size]

    # Raw sensor readings from mobile
    m_acc_x = df['m_acc_x'].values[i: i + frame_size]
    m_acc_y = df['m_acc_y'].values[i: i + frame_size]
    m_acc_z = df['m_acc_z'].values[i: i + frame_size]
    m_gyro_x = df['m_gyro_x'].values[i: i + frame_size]
    m_gyro_y = df['m_gyro_y'].values[i: i + frame_size]
    m_gyro_z = df['m_gyro_z'].values[i: i + frame_size]

    label = stats.mode(df['user_activity'][i: i + frame_size])[0][0]

    frames.append([w_acc_x, w_acc_y, w_acc_z, w_bvp, m_acc_x, m_acc_y, m_acc_z, m_gyro_x, m_gyro_y, m_gyro_z])
    labels.append(label)

  frames = np.asarray(frames, dtype=np.float32).reshape(-1, frame_size, 10)
  return frames, labels

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

X_df = df[['user_activity','user_id']] #df[df.columns.difference(['user_activity','user_id'])]

# splitting data into 4 sub-samples by user_id
df1 = df[df['user_id'] == 1].copy()
df2 = df[df['user_id'] == 2].copy()
df3 = df[df['user_id'] == 3].copy()
df4 = df[df['user_id'] == 4].copy()

# dropping user_id column for each df
df1.drop(columns=['user_id'], axis=1, inplace=True)
df2.drop(columns=['user_id'], axis=1, inplace=True)
df3.drop(columns=['user_id'], axis=1, inplace=True)
df4.drop(columns=['user_id'], axis=1, inplace=True)

# applying sliding window on each dataframe
X_df1, y_df1 = get_frames(df1, frame_size, hop_size)
X_df2, y_df2 = get_frames(df2, frame_size, hop_size)
X_df3, y_df3 = get_frames(df3, frame_size, hop_size)
X_df4, y_df4 = get_frames(df4, frame_size, hop_size)


# converting labels to series
y_df1 = pd.Series(y_df1)
y_df2 = pd.Series(y_df2)
y_df3 = pd.Series(y_df3)
y_df4 = pd.Series(y_df4)

y_df1 = pd.get_dummies(y_df1)
y_df2 = pd.get_dummies(y_df2)
y_df3 = pd.get_dummies(y_df3)
y_df4 = pd.get_dummies(y_df4)

# train test split for each dataframe
X_train_df1, X_test_df1, y_train_df1, y_test_df1 = train_test_split(X_df1, y_df1, test_size=0.20, stratify=y_df1, random_state=142)
X_train_df2, X_test_df2, y_train_df2, y_test_df2 = train_test_split(X_df2, y_df2, test_size=0.20, stratify=y_df2, random_state=142)
X_train_df3, X_test_df3, y_train_df3, y_test_df3 = train_test_split(X_df3, y_df3, test_size=0.20, stratify=y_df3, random_state=142)
X_train_df4, X_test_df4, y_train_df4, y_test_df4 = train_test_split(X_df4, y_df4, test_size=0.20, stratify=y_df4, random_state=142)

user1_train =tf.data.Dataset.from_tensor_slices((X_train_df1, y_train_df1)).batch(50)
user2_train =tf.data.Dataset.from_tensor_slices((X_train_df2, y_train_df2)).batch(50)
user3_train =tf.data.Dataset.from_tensor_slices((X_train_df3, y_train_df3)).batch(50)
user4_train =tf.data.Dataset.from_tensor_slices((X_train_df4, y_train_df4)).batch(50)

print('X_train_df1', X_train_df4.shape)
print('X_test_df1', X_test_df1.shape)
print('y_train_df1', y_train_df1.shape)
print('y_test_df1', y_test_df1.shape)


federated_train_data=[user1_train,user4_train,user3_train]

"""## **Federated Learning with Tensorflow -Fed Avg**"""

def create_keras_model(n_timesteps,n_features=10,n_outputs=11):
  return tf.keras.models.Sequential([
      tf.keras.layers.LSTM(100,input_shape=(n_timesteps,n_features)),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(100, activation='relu'),
      tf.keras.layers.Dense(n_outputs),
  ])

def model_fn():
  # create a new keras model here.
  keras_model = create_keras_model(n_timesteps=100)
  return tff.learning.from_keras_model(
      keras_model,
      input_spec= user1_train.element_spec,
      loss=tf.keras.losses.CategoricalCrossentropy(),
      metrics=[tf.keras.metrics.CategoricalAccuracy()])

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

# Initialize the state
state = iterative_process.initialize()

print(iterative_process.initialize.type_signature.formatted_representation())

#@test {"skip": true}
logdir = "training/"
summary_writer = tf.summary.create_file_writer(logdir)

federated_train_data[2]

NUM_ROUNDS = 10

with summary_writer.as_default():
  for round_num in range(1, NUM_ROUNDS):
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round {:2d}, metrics={}'.format(round_num, metrics))
    for name, value in metrics['train'].items():
      tf.summary.scalar(name, value, step=round_num)

# Commented out IPython magic to ensure Python compatibility.
# visualise data on tensorboard 
# %load_ext tensorboard
#@test {"skip": true}
!ls {logdir}
# %tensorboard --logdir {logdir} --port=0

"""### Federated Learning With Tensorflow - FedProx"""

iterative_process_prox = tff.learning.algorithms.build_weighted_fed_prox(
    model_fn,
    proximal_strength =1.0,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

NUM_ROUNDS = 100

with summary_writer.as_default():
  for round_num in range(1, NUM_ROUNDS):
    output = iterative_process_prox.next(prox_state, federated_train_data)
    prox_state = output.state
    print('round {:2d}, metrics={}'.format(round_num,output.metrics))
    for name, value in metrics['train'].items():
      tf.summary.scalar(name, value, step=round_num)

# initial model weight 
model_weights = iterative_process.get_model_weights(state)

# Initialize with current weights
state = iterative_process.initialize()