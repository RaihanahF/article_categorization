# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:46:25 2022

@author: Fatin
"""

import pandas as pd
import os
import numpy as np
import pickle
import tensorflow as tf
from articles_categorization_modules import ExploratoryDataAnalysis, ModelCreation
from articles_categorization_modules import ModelEvaluation
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
import datetime

TOKEN_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model', 'tokenizer_data.json')
OHE_SCALER_PATH = os.path.join(os.getcwd(), 'saved_model', 'ohe_scaler.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_model', 'model.h5')
LOG_PATH = os.path.join(os.getcwd(),'log')
URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'

# 1: Data Loading

df = pd.read_csv(URL)

# 2: Data Inspection

df.info()
df.describe()

print(df.nunique())
print(df.isna().sum())

# 3: Data Cleaning
# Remove special charaters and numbers

article = df['text']
category = df['category']

eda = ExploratoryDataAnalysis()

article = eda.remove_unwanted_characters(article)
article = eda.lower_split(article)

# 3: Features Selection
# 4: Data Vectorization

article = eda.category_tokenizer(article, TOKEN_SAVE_PATH)
article = eda.category_pad_sequence(article)

# 5: Preprocessing

# One hot encoder
ohe_scaler = OneHotEncoder(sparse=False)
category = ohe_scaler.fit_transform(np.expand_dims(category, axis=-1))
pickle.dump(ohe_scaler, open(OHE_SCALER_PATH, 'wb'))

# Calculate number of categories
nb_categories = len(np.unique(category, axis=0))

# Train test split
X_train, X_test, y_train, y_test = train_test_split(article, category, 
                                                    test_size=0.3, random_state=123)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

#%% Model Creation

mc = ModelCreation()

num_words = 10000

model = mc.lstm_layer(num_words, nb_categories)

tf.keras.utils.plot_model(model, to_file='model_test.png', show_shapes=True, show_layer_names=True)

log_dir = os.path.join(LOG_PATH,
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#%% Compile & Model Fitting

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

model.fit(X_train, y_train, epochs=5,
          validation_data=(X_test, y_test),
          callbacks=tensorboard_callback)

#%% Model Evaluation
# Append approach

# Preallocation of memory approach

predicted_advanced = np.empty([len(X_test), 2])

for index, test in enumerate(X_test):
    predicted_advanced[index,:] = model.predict(np.expand_dims(test, axis=0))

#%% Model Analysis

y_pred = np.argmax(predicted_advanced, axis=1)
y_true = np.argmax(y_test, axis=1)

me = ModelEvaluation()
me.report_metrics(y_true, y_pred)

#%% Model Deployment
model.save(MODEL_SAVE_PATH)
