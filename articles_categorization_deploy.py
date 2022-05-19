# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:21:48 2022

@author: Fatin
"""

import os
import json
from articles_categorization_modules import ExploratoryDataAnalysis
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np

MODEL_PATH = os.path.join(os.getcwd(), 'saved_model', 'model.h5')
JSON_PATH = os.path.join(os.getcwd(), 'saved_model', 'tokenizer_data.json')

#%% Model Loading
category_classifier = load_model(MODEL_PATH)
category_classifier.summary()

#%% Tokenizer Loading
with open(JSON_PATH, 'r') as json_file:
    token = json.load(json_file)
    
#%% EDA

category_dict = {0:'business', 1:'entertainment', 2:'politics', 3:'sport', 4:'tech'}

# 1: Load Data
new_article = ['<br \> A COOL and composed Shaqeem Eiman Shahyar turned hero to\
               take Malaysia into the men’s team badminton final. Malaysia came\
                from behind to level the score at 2-2 with Singapore and the \
                atmosphere was tense when Shaqeem stepped onto the court in the\
                decider but the 21-year-old displayed nerves of steel to beat \
                Joel Koh 21-10, 21-17 to deliver the winning point.<br \>']

#new_article = [input('Type new article\n')]
              
# 2: Clean Data
eda = ExploratoryDataAnalysis()
removed_chars = eda.remove_unwanted_characters(new_article)
cleaned_input = eda.lower_split(new_article)

# 3: Features Selection
# 4: Data Preprocessing

# Vectorize the new review
# Feed the token into keras
loaded_tokenizer = tokenizer_from_json(token)

# Vectorize the review into integers
new_article = loaded_tokenizer.texts_to_sequences(cleaned_input)
new_article = eda.category_pad_sequence(new_article)

# Model Prediction
outcome = category_classifier.predict(np.expand_dims(new_article, axis=-1))

print('category:' + category_dict[np.argmax(outcome)])

