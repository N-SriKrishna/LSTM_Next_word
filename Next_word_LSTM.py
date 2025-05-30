import nltk
nltk.download("gutenberg")
from nltk.corpus import gutenberg
import pandas as pd

data=gutenberg.raw("shakespeare-hamlet.txt")
with open("hamlet.txt","w") as file:
    file.write(data)

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

with open("hamlet.txt","r") as file:
    text=file.read().lower()

tokenizer = Tokenizer()  #creating index for words
tokenizer.fit_on_texts([text])
total_words=len(tokenizer.word_index)+1

input_sequences=[]
for line in text.split('\n'):
    token_list=tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
        n_gram_sequence=token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_seq_len=max([len(x) for x in input_sequences])
padded_sequences=pad_sequences(input_sequences,maxlen=max_seq_len)
input_sequences=np.array(padded_sequences)

import tensorflow as tf
x,y=input_sequences[:,:-1],input_sequences[:,-1]    
y=tf.keras.utils.to_categorical(y,num_classes=total_words)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout

model=Sequential()
model.add(Embedding(total_words,100,input_length=max_seq_len-1))
model.add(LSTM(150,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words,activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

history=model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test),verbose=1)

def predict_next_word(model,tokenizer,text,max_seq_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_seq_len:
        token_list=token_list[-(max_seq_len-1):]
    token_list=pad_sequences([token_list],maxlen=max_seq_len-1,padding="pre")
    predicted = model.predict(token_list,verbose=1)
    predicted_word_index=np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None

import pickle
model.save("Next_word_LSTM.keras")
with open("tokenizer.pkl","wb") as file:
    pickle.dump(tokenizer,file)



