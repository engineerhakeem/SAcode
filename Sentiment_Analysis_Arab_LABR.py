# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk

#from pandas import DataFrame, read_csv
#import matplotlib.pyplot as plt
#from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from nltk.stem.isri import ISRIStemmer
#from keras.utils.np_utils import to_categorical
import re
from pyarabic.araby import strip_tashkeel



#Input data files are available in the "../input/" directory.
#For example, running this (by clicking run or pressing Shift+Enter) 
#will list the files in the input directory

file1 = r"D:\ReserchCourse\datasets\ar-embeddings-master\ar-embeddings-master\datasets\LABR-book-reviews.csv"
data1 = pd.read_csv(file1, encoding='utf8')


# Keeping only the neccessary columns
data1 = data1[['txt','sentiment']]


#Next, I am dropping the 'Neutral' sentiments as my goal was to only differentiate
#positive and negative tweets.
#data1 = data1[data1.polarity != 0]


#data1 = data1.replace('Positive',1)
data1 = data1.replace(0,-1)

  
print(data1)
#data1 = data1[data1.sentiment != 'Both']

# Download Arabic stop words
nltk.download('stopwords')
# Extract Arabic stop words
arb_stopwords = set(nltk.corpus.stopwords.words("arabic"))
print (arb_stopwords)
# Initialize Arabic stemmer
st = ISRIStemmer()

data1['txt'] = data1['txt'].apply(lambda x: x.strip_tashkeel())
#After that, I am filtering the tweets so only valid texts and words remain. 
#data['text'] = data['text'].apply(lambda x: x.lower())
#data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print(np.size(data1[ data1['sentiment'] == 1])) # method to know size of data
print(data1[ data1['sentiment'] == -1].size)    # another method to know size of data

#for idx,row in data.iterrows():
 #   row[0] = row[0].replace('rt',' ')
    
#print(data)
    
#Then, I define the number of max features as 2000 and use Tokenizer to vectorize and convert text
#into Sequences 
#so the Network can deal with it as input.    
max_fatures1 = 2000
tokenizer1 = Tokenizer(num_words=max_fatures1, split=' ')
tokenizer1.fit_on_texts(data1['txt'].values)
print(tokenizer1.word_counts)
print(tokenizer1.document_count)
print(tokenizer1.word_index)
print(tokenizer1.word_docs)
#data1['txt'] = data1['txt'].map(lambda x: [w for w in x if w not in arb_stopwords])
#print (data1['txt'])
X1 = tokenizer1.texts_to_sequences(data1['txt'].values)
#print(X1)
X1 = pad_sequences(X1)
print(X1)

#Next, I compose the LSTM Network. Note that embed_dim, lstm_out, batch_size,
#droupout_x variables are hyperparameters,
#their values are somehow intuitive, can be and must be played with in order to achieve good results. 
#Please also note that I am using softmax as activation function. 
#The reason is that our Network is using categorical crossentropy,
#and softmax is just the right activation method for that.
embed_dim1 = 128
lstm_out1 = 196

model = Sequential()
model.add(Embedding(max_fatures1, embed_dim1,input_length = X1.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out1, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

#Hereby I declare the train and test dataset.
Y1 = pd.get_dummies(data1['sentiment']).values
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1, test_size = 0.33, random_state = 42)
print(X1_train.shape,Y1_train.shape)
print(X1_test.shape,Y1_test.shape)

#Here we train the Network.
#We should run much more than 7 epoch, but I would have to wait forever for kaggle, so it is 7 for now.
batch_size1 = 32
model.fit(X1_train, Y1_train, epochs = 7, batch_size=batch_size1, verbose = 2)

model.save("D:\\ReserchCourse\\datasets\\datasets\\LABR-book-reviews.h5")
new_model = load_model("D:\\ReserchCourse\\datasets\\datasets\\LABR-book-reviews.h5")
#Extracting a validation set, and measuring score and accuracy.
validation_size1 = 1500

X1_validate = X1_test[-validation_size1:]
Y1_validate = Y1_test[-validation_size1:]
X1_test = X1_test[:-validation_size1]
Y1_test = Y1_test[:-validation_size1]
score1,acc1 = new_model.evaluate(X1_test, Y1_test, verbose = 2, batch_size = batch_size1)
print("score1: %.2f" % (score1))
print("acc1: %.2f" % (acc1))



#As it was requested by the crowd, I extended the kernel with a prediction example,
#and also updated the API calls to Keras 2.0. Please note that the network performs poorly.
#Its because the training data is very unbalanced (pos: 4472, neg: 16986),
#you should get more data, use other dataset, use pre-trained model,
#or weight classes to achieve reliable predictions.

#I have created this kernel when I knew much less about LSTM & ML. 
#It is a really basic, beginner level kernel, yet it had a huge audience in the past year. 
#I had a lot of private questions and requests regarding this notebook and
#I tried my best to help and answer them .
#In the future I am not planning to answer custom questions and support/enhance this kernel in any ways.
#Thank you my folks :)

twt = ['كتاب مبهم ']
#vectorizing the tweet by the pre-fitted tokenizer instance
twt = tokenizer1.texts_to_sequences(twt)
print (twt)

#padding the tweet to have exactly the same shape as `embedding_2` input
twt = pad_sequences(twt, maxlen=1440, dtype='int32', value=0)

sentiment = new_model.predict(twt,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")