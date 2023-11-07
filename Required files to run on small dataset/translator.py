######################################################################################################################################################################
#IMPORTS
######################################################################################################################################################################

import streamlit as st
import io
import base64
import pytesseract
import os
import random
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
import pandas as pd
import numpy as np
import string
from string import digits
try:
 from PIL import Image
except ImportError:
 import Image
from keras.models import load_model
import time
import re

######################################################################################################################################################################
#DATA
######################################################################################################################################################################

PATH='dataset_san_en.csv'
lines=pd.read_csv(PATH,encoding='utf-8')

# Lowercase all characters
lines.english=lines.english.apply(lambda x: x.lower())
lines.sanskrit=lines.sanskrit.apply(lambda x: x.lower())

# Remove quotes
lines.english=lines.english.apply(lambda x: re.sub("'", '', x))
lines.sanskrit=lines.sanskrit.apply(lambda x: re.sub("'", '', x))

exclude = set(string.punctuation) # Set of all special characters
# Remove all the special characters
lines.english=lines.english.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
lines.sanskrit=lines.sanskrit.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

# Remove all numbers from text
remove_digits = str.maketrans('', '', digits)
lines.english=lines.english.apply(lambda x: x.translate(remove_digits))
lines.sanskrit = lines.sanskrit.apply(lambda x: re.sub("[२३०८१५७९४६1234567890]", "", x))

# Remove extra spaces
lines.english=lines.english.apply(lambda x: x.strip())
lines.sanskrit=lines.sanskrit.apply(lambda x: x.strip())
lines.english=lines.english.apply(lambda x: re.sub(" +", " ", x))
lines.sanskrit=lines.sanskrit.apply(lambda x: re.sub(" +", " ", x))

# Add start and end tokens to target sequences
lines.english = lines.english.apply(lambda x : 'START_ '+ x + ' _END')

all_eng_words=set()
for eng in lines.english:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

all_san_words=set()
for san in lines.sanskrit:
    for word in san.split():
        if word not in all_san_words:
            all_san_words.add(word)

# Max Length of source sequence
lenght_list=[]
for l in lines.sanskrit:
    lenght_list.append(len(l.split(' ')))
max_length_src = np.max(lenght_list)

mean_length_src=np.mean(lenght_list)

# Max Length of target sequence
lenght_list=[]
for l in lines.english:
    lenght_list.append(len(l.split(' ')))
max_length_tar = np.max(lenght_list)

mean_length_tar=np.mean(lenght_list)

target_words = sorted(list(all_eng_words))
input_words = sorted(list(all_san_words))
num_decoder_tokens = len(all_eng_words)
num_encoder_tokens = len(all_san_words)

num_decoder_tokens += 1
num_encoder_tokens += 1

input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])
target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])

reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())

latent_dim = 50

######################################################################################################################################################################
#PREPROCESS INPUT TEXT
######################################################################################################################################################################

def preprocess(txt):
    txt=txt.lower()
    txt=re.sub("'",'',txt)
    exclude = set(string.punctuation)
    txt=''.join([x for x in txt if x not in exclude])
    txt = re.sub("[२३०८१५७९४६1234567890]", "", txt)
    txt=txt.strip()
    txt=re.sub(" +", " ", txt)
    #print(type(txt))
    return txt

######################################################################################################################################################################
#MODEL
######################################################################################################################################################################

encoder_inputs = Input(shape=(None,))
enc_emb =  Embedding(num_encoder_tokens, latent_dim, mask_zero = True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero = True)
dec_emb = dec_emb_layer(decoder_inputs)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model_path='fin.h5'
model.load_weights(model_path)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])


# Encode the input sequence to get the "thought vectors"
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2= dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

######################################################################################################################################################################
#BLUR IMAGE IDENTIFICATION
######################################################################################################################################################################

#Random Forest classifier

######################################################################################################################################################################
#SHARPENING
######################################################################################################################################################################

#sharpening kernel

######################################################################################################################################################################
#OCR
######################################################################################################################################################################

def get_lines(img):
    lines = pytesseract.image_to_string(Image.open(img), lang='hin')
    lines=preprocess(lines)
    return lines

######################################################################################################################################################################
#PREDICTION
######################################################################################################################################################################

def decode_sequence(input_seq,lenn):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
           len(decoded_sentence) > lenn*(mean_length_tar/mean_length_src)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

def get_input_data(sentence):
          encoder_input_data = np.zeros((1, max_length_src),dtype='float32')
          for t, word in enumerate(sentence.split()):
            try:
              encoder_input_data[0, t] = input_token_index[word] # encoder input seq
            except:
              encoder_input_data[0, t] = 0 # encoder input seq
          return encoder_input_data,len(sentence)

def get_translation(sentence):
  input_seq,n=get_input_data(sentence)
  decoded_sentence = decode_sequence(input_seq,n)
  return decoded_sentence[1:-5]


######################################################################################################################################################################
#UI
######################################################################################################################################################################

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('3.jpg')

option = st.selectbox('Select?',('Image', 'Text'),label_visibility="collapsed")

out=""
if option == "Image":
    img = st.file_uploader("Upload image...",label_visibility="collapsed")
    if img is not None:
        txt=get_lines(img)
        txt=preprocess(txt)
        print(txt)
        try:
            out=translater(txt)
        except:
         try:
            out=get_translation(txt)
       except:
            out='Couldn\'t find an appropriate translation'
        finally:
            print(out)
            



elif option == "Text":
    txt = st.text_area("Input Text",placeholder="Enter Text here...",label_visibility="collapsed")
    txt=preprocess(txt)
    if txt is not None:
        try:
            out=get_translation(txt)
            
        except:
            out='Couldn\'t find an appropriate translation'
        finally:
            print(out)
            



st.markdown("#")
st.markdown("#")
st.markdown("#")
st.markdown("#")
st.markdown("#")
st.markdown("#")
st.text_area("Output",out,label_visibility="collapsed")

######################################################################################################################################################################
#END
######################################################################################################################################################################
