
#Flask deployment


import flask
from flask import request,Flask,jsonify
import cv2 
import math
from age_detection_mod.age_detection import age_detector
from adult_video_cap_detection.cap import Cap_Model
from tensorflow.keras.layers import Input,LSTM,Bidirectional,BatchNormalization,Dense,Dropout
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as numpy
import cv2
import os
import transformers
from transformers import BertTokenizer
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from bs4 import BeautifulSoup
from urlib.request import urlopen
import urllib
from bs4 import BeautifulSoup
from urllib.request import urlopen

# Web Page data is fetched from example website mentioned below

html=urlopen('https://house.porn/best')                             
bs = BeautifulSoup(html, "html.parser")
titles = bs.find_all(['h1'])
captions=bs.find_all(['h2','h3','h4','h5','h6'])
for t in titles:
  text1=t.get_text().strip()

print(text1)
print('\n')

text2=[]
for t in captions:
  text2.append(t.get_text().strip())
print(text2)





app=Flask(__name__)


def custom_model(bert_base_layer):
    input_layer=Input(shape=(200,), dtype=tf.int32)
    sequence_output=bert_base_layer(input_layer)[0]
    cls_token=sequence_output[:,0,:]
    X = tf.keras.layers.BatchNormalization()(cls_token)
    X = tf.keras.layers.Dense(200, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    out = Dense(2, activation='softmax')(X)

    return tf.keras.Model(inputs=input_layer,outputs=out)


bert_layer= transformers.TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
page_model=custom_model(bert_layer)

page_model.load_weights('Toxicity page weights/Toxicity multilingual weights (All 104 languages)/multiligual_bert.h5')

def capmodel(vocablen):
  capmodel=Cap_Model(vocablen)
  x_in = tf.keras.layers.Input(shape=(60,), dtype=tf.float32)
  initializer=capmodel(x_in, training=False)
  return capmodel

capmodel=capmodel(39750)
capmodel.load_weights('adult_video_cap_detection/porn_detection.h5')



@app.route("/predict",methods=["POST","GET"])
def predict():
  n=np.random.randint(0,10)
  response={'input1': text1,'input2': text2[n]}
  if type(response['input1'])==np.str:
    bert_tokenizer=BertTokenizer('Toxicity page weights/Toxicity multilingual weights (All 104 languages)/vocab.txt')
    inputs=bert_tokenizer(response['input1'])['input_ids']
    response['out1']=page_model.predict([inputs]).argmax(-1)
    response['out1']=response['out1'].tolist()

  if type(response['input2'])==np.str:
    infile = open('adult_video_cap_detection/tokenizer.pkl','rb')
    tokenizer=pickle.load(infile)
    sequences=tokenizer.texts_to_sequences([response['input2']])
    inp2_sequences=tf.keras.preprocessing.sequence.pad_sequences(sequences,maxlen=60)
    response['out2']=capmodel.predict(inp2_sequences).argmax(-1)
    response['out2']=response['out2'].tolist()

  if response['input2'] is not None:
    response['out3']=age_detector('age_detection_mod/test_vids/vid7.mp4')  #video path is mentioned from the folder structure of local system
    
  return jsonify(response)
    

if __name__ == "__main__":
    app.run(debug=True)
