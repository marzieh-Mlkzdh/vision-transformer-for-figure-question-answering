# To extract the features in questions, first it is necessary to devide each question into tokens that are more important ( .e.g. omitting the prepositions 
# or conjunctives), then all the questions will get into the LSTM network for training, following by a dense layer.
# features created in LSTM itself are our target for embedding.

from keras.datasets import imdb
import pandas as pd
import numpy as np
from numpy import savetxt
from keras.layers import LSTM, Activation, Dropout, Dense, Input
from keras.layers.embeddings import Embedding
from keras.models import Model
import string
import re
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.sequence import pad_sequences
import keras
from sklearn.model_selection import train_test_split


data = pd.read_csv(r'...\qa_pairs.csv')
data.info()



data['question_string'] = data['question_string'].str.lower()


stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", 
             "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during",
             "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", 
             "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
             "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
             "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", 
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
             "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
             "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
             "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've",
             "your", "yours", "yourself", "yourselves" ]




def remove_stopwords(data):
    data['question_string without stopwords'] = data['question_string'].apply(lambda x : ' '.join([word for word in x.split() if word not in (stopwords)]))
    return data

def remove_tags(string):
    result = re.sub('<.*?>','',string)
    return result
    
data_without_stopwords = remove_stopwords(data)
data_without_stopwords['clean_question_string']= data_without_stopwords['question_string without stopwords'].apply(lambda cw : remove_tags(cw))
data_without_stopwords['clean_question_string'] = data_without_stopwords['clean_question_string'].str.replace('[{}]'.format(string.punctuation), ' ')




question_string_list = []
for i in range(len(data['question_string'])):
    question_string_list.append(data['question_string'][i])
 
    answer = data_without_stopwords['answer']




y = np.array(list(answer))

X_train, X_test,Y_train, Y_test = train_test_split(question_string_list, y, test_size=0.2, random_state = 45)




tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

words_to_index = tokenizer.word_index


# for the tokenizer section,  we prefer to use GloVe tokenizer for it's high accuracy and undrestandable structure.

def read_glove_vector(glove_vec):
    with open(glove_vec, 'r', encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)



    return word_to_vec_map




word_to_vec_map = read_glove_vector(r'C:\Users\sepehr\glove.6B.50d.txt')

maxLen = 150




vocab_len = len(words_to_index)
embed_vector_len = word_to_vec_map['answer'].shape[0]

emb_matrix = np.zeros((vocab_len, embed_vector_len))

for word, index in words_to_index.items():
    embedding_vector = word_to_vec_map.get(word)
    if embedding_vector is not None:
        emb_matrix[index - 1, :] = embedding_vector

embedding_layer = Embedding(input_dim=vocab_len, output_dim=embed_vector_len, input_length=maxLen, weights = [emb_matrix], trainable=False)



X_train_indices = tokenizer.texts_to_sequences(X_train)

X_train_indices = pad_sequences(X_train_indices, maxlen=maxLen, padding='post')



from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping

model = keras.Sequential()
model.add(layers.Embedding(input_dim=vocab_len, output_dim=embed_vector_len, input_length=maxLen, weights = [emb_matrix], trainable=False))
model.add(layers.SpatialDropout1D(0.2))
model.add(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



#history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
history = model.fit(X_train_indices, Y_train, batch_size=32, epochs=50 , validation_split=0.1)

model.summary()



layer_name = 'lstm'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
LSTM_output = intermediate_layer_model.predict(data)
np.savetxt('/content/drive/MyDrive/QuestionFeatures.csv', LSTM_output, delimiter=',')

