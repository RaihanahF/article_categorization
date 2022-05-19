# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:59:26 2022

@author: Fatin
"""

import pandas as pd
import re
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Embedding, Bidirectional
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ExploratoryDataAnalysis():
    
    def __init__(self):
        
        pass
    
    def remove_unwanted_characters(self, data):
        
        for index, text in enumerate(data):
            data[index] = re.sub('[0-9]', '', text)
            data[index] = re.sub('<.*?>!@#', '', text)
            data[index] = re.sub('\s+', ' ', text)
            
        return data
    
    def lower_split(self, data):
        
        for index, text in enumerate(data):
            data[index] = re.sub('[^a-zA-Z]', ' ', text).lower().split()
        
        return data

    def category_tokenizer(self, data, token_save_path, 
                            num_words=10000, oov_token='<OOV>', prt=False):
        # tokenizer to vectorize the words
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(data)
        
        # to save the tokenizer for deployment purpose
        token_json = tokenizer.to_json()
        
        with open(token_save_path, 'w') as json_file:
            json.dump(token_json, json_file)
            
        # to observe the number of words
        word_index = tokenizer.word_index
        
        if prt == True:
            # to view the tokenized words
            # print(word_index)
            print(dict(list(word_index())[0:10]))
        
        # to vectorize the sequence of text
        data = tokenizer.texts_to_sequences(data)
        
        return data
    
    def category_pad_sequence(self, data):

        return pad_sequences(data, maxlen=200, padding='post', truncating='post')
    
    def remove_duplicates(self, df):
        
        return pd.DataFrame(df).drop_duplicates()
    
    
class ModelCreation():
    
    def __init__(self):
        pass
    
    def lstm_layer(self, num_words, nb_categories, 
                   embedding_output=64, nodes=32, dropout=0.2):
        
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        model.add(Bidirectional(LSTM(nodes, return_sequences=True)))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation='softmax'))
        model.summary()
        
        return model

    def simple_lstm_layer(self, num_words, nb_categories,
                          embedding_output=64, nodes=32, dropout=0.2):

        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        model.add(LSTM(nodes, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation='softmax'))
        model.summary()
        
        return model
            
class ModelEvaluation():
    def report_metrics(self, y_true, y_pred):
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        print(accuracy_score(y_true, y_pred))