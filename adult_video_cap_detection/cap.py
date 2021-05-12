import tensorflow as tf

import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense,Input,Embedding,Bidirectional,LSTM,Lambda,Reshape
from tensorflow.keras.models import Model


class Cap_Model(tf.keras.Model):
    def __init__(self,vocablen,units=512):
        super(Cap_Model,self).__init__()
        self.embedding_layer=tf.keras.layers.Embedding(vocablen,100)
        self.lstm_layer=tf.keras.layers.LSTM(units,return_sequences=True)
        self.dense_layer_1=tf.keras.layers.Dense(units/2,activation='relu')
        self.dense_layer_2=tf.keras.layers.Dense(2,activation='softmax')
        self.dropout_layer=tf.keras.layers.Dropout(0.2)
        self.flat_layer=tf.keras.layers.Flatten()

    def call(self,inputs):
        x=self.embedding_layer(inputs)
        x=self.lstm_layer(x)
        x=self.flat_layer(x)
        x=self.dropout_layer(x)
        x=self.dense_layer_1(x)
        x=self.dense_layer_2(x)

        return x