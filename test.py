import numpy as np
import pandas as pd
import os, datetime
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
print('Tensorflow version: {}'.format(tf.__version__))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import warnings
warnings.filterwarnings('ignore')

# HYPERPARAMETERS
batch_size = 32
seq_len = 128

d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256

data_path = '/home/jbao/data/53e8fa668e202f27.csv'

df = pd.read_csv(data_path, delimiter=',', usecols=['date', 'best_bid', 'best_offer', 'strike_price', 'impl_volatility', 'volume'])

# Replace 0 to avoid dividing by 0 later on
df['volume'].replace(to_replace=0, method='ffill', inplace=True)
df.sort_values('date', inplace=True)
df.tail()


fig = plt.figure(figsize=(15,10))
st = fig.suptitle("Example graph", fontsize=20)
st.set_y(0.92)

ax1 = fig.add_subplot(211)
ax1.plot(df['best_bid'], label='IBM Close Price')
ax1.set_xticks(range(0, df.shape[0], 1464))
ax1.set_xticklabels(df['date'].loc[::1464])
ax1.set_ylabel('Best Bid', fontsize=18)
ax1.legend(loc="upper left", fontsize=12)
