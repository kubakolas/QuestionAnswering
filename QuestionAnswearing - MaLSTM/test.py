from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint


# File paths
TRAIN_CSV = 'C:/Users/Kociuba/Desktop/qa/QuestionAnswering/QuestionAnswering/trainGoodBad.csv'
TEST_CSV = 'C:/Users/Kociuba/Desktop/qa/QuestionAnswering/QuestionAnswering/trainGoodBad.csv'
EMBEDDING_FILE = 'file.txt'
MODEL_SAVING_DIR = 'C:/Users/Kociuba/Downloads/'


# Ładowanie danych testowych i treningowych, oba typy CVS
# Układ danych to pytanie1, pytanie2, czy_jest_duplikatem - treningowe
# Układ danych to pytanie1, pytanie2 - testowe
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# Preprocesing tekstu - nas niedotyczy bo mamy juz obrobiony tekst
stops = set(stopwords.words('english'))


def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


# Tworzenie niezbędnych danych dla użycia w2v

# Słownik dla naszych ztokenizowanych list słów - <słowo, jego dentyfikator>
vocabulary = dict()

# Słowa jakie nie znalazły sie w naszym słowniku - ponieważ nie istneją w podanym korpusie.
# Słownik naszych słów jakie występują w pytaniach jest porównywany z tym w naszym korpusie word2vec
# Jeśli słowa nie ma w korpusie, to musimy je wyrzucić bo nie ma dla niego wektorowej reprezentacji
inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding

# Ładowanie korpusu GoogleNews
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=False, unicode_errors='ignore')

questions_cols = ['question1', 'question2']

# Iteracja nad naszymi zbiorami danych aby wyrzucić niepotrzebe słowa
for dataset in [train_df, test_df]:
    for index, row in dataset.iterrows():

        # Iterate through the text of both questions of the row
        for question in questions_cols:

            q2n = []  # q2n -> question numbers representation
            for word in text_to_word_list(row[question]):

                # Check for unwanted words
                if word in stops and word not in word2vec.vocab:
                    continue
                # Tworzenie słownika <id, slowo> oraz takiego ze slownik[id] = slowo
                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary) #dodanie słowa do słownika i danie mu klucza równego dł. inverse vocabulary
                    q2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    q2n.append(vocabulary[word])

            # Replace questions as word to question as number representation
            dataset.set_value(index, question, q2n)


print(vocabulary)
print(inverse_vocabulary)
# /\ Przerobione nasze pytania nie będa posiadać w sobie słowa o indexie 0 ponieważ
# || jest on przeznaczony na 0 padding - co kolwiek to znaczy

# Tworzenie macierzy dla naszego słownika i wektorów
embedding_dim = 25
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
embeddings[0] = 0  # So that the padding will be ignored

# Wpoisywanie wartosci do macierzy
for word, index in vocabulary.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)

del word2vec

# Ustalenie makxymalnej dlugości z naszych zbiorów - najdłuższe pytanie
max_seq_length = max(train_df.question1.map(lambda x: len(x)).max(),
                     train_df.question2.map(lambda x: len(x)).max(),
                     test_df.question1.map(lambda x: len(x)).max(),
                     test_df.question2.map(lambda x: len(x)).max())

# Podział na zbiór treningowy i walidacyjny
validation_size = 2000
training_size = len(train_df) - validation_size

# Wskazanie co jest wejsciem - jakie nazwy kolumn z naszego pliku csv
# Co jest wyjsciem - kolumna czy jest dluplikatem
X = train_df[questions_cols]
Y = train_df['is_duplicate']

# Wskazanie co jest wejsciem - jakie nazwy kolumn z naszego pliku csv
# Co jest wyjsciem - kolumna czy jest dluplikatem
X_test = test_df[questions_cols]
Y_test = test_df['is_duplicate']

#Rozpicie zbiru danych na podzbiory walidaci i testowy
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

# Rozbicie na dwie strony, bo dwa wejscia sieci syjamskich
X_train = {'left': X_train.question1, 'right': X_train.question2}
X_validation = {'left': X_validation.question1, 'right': X_validation.question2}
X_test = {'left': X_test.question1, 'right': X_test.question2}

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values
Y_test = Y_test.values

# Uzupełniene zerami, tam gdzie brakuje słów do pełnej długośći
for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)


# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

assert X_test['left'].shape == X_test['right'].shape
assert len(X_test['left']) == len(Y_test)

# Model variables
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 1

# Definiowanie funkcji jaka będzie wykorzystywana u nas w warstwie łączenia
# Czyli w momencie porównania pytań - MaLSTM similarity function
def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

# Warstwy widoczne
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

# Warstwa z wektorami dal całego słownika
embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Pack it all up into a model
malstm = Model([left_input, right_input], [malstm_distance])

# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy']) ##wlasna f bledu keras - negativ sampling(na przykladzie
# gensim.models.vord2vec lub keyedvectors skipgram
# operetional intencity -ile operacji na 1 bajt danych

# Start training
training_start_time = time()

malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, nb_epoch=n_epoch,
                            validation_data=([X_validation['left'], X_validation['right']], Y_validation))

predict = malstm.predict([X_test['left'], X_test['right']])
print('#################', predict[0])
malstm.save('my.h5')

print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))


#malstm_tested = malstm.evaluate([X_test['left'], X_test['right']], Y_test, batch_size=batch_size)
#Evaluate the model on the test data using `evaluate`
#print('\n# Evaluate on test data')
#print('test loss, test acc:', malstm_tested)
# Plot accuracy
plt.plot(malstm_trained.history['acc'])
plt.plot(malstm_trained.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()