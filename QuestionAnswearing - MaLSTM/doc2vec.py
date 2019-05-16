from time import time
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Lambda
from keras import backend as K
from keras.optimizers import Adadelta
from keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import datetime

# link do paczki z plikami https://drive.google.com/file/d/1cjAuU9dg8KsyDxCr53hQENQfZba2ueV3/view?usp=sharing

questions = pickle.load(open("questionsVecTrain.p", "rb"))
answers = pickle.load(open("answersVecTrain.p", "rb"))
labels = pickle.load(open("labelsTrain.p", "rb"))

max_seq_length = 200
validation_size = 16000
training_size = len(questions) - validation_size
indices = list(range(0, len(questions) - 1))
shuffled_indices= np.random.permutation(indices)

q_t = []
a_t = []
l_t = []

for index in shuffled_indices:
    q_t.append(questions[index])
    a_t.append(answers[index])
    l_t.append(labels[index])

questions = q_t
answers = a_t
labels = l_t

val_q = questions[0:validation_size]
val_a = answers[0:validation_size]
val_l = labels[0:validation_size]

del questions[0:validation_size]
del answers[0:validation_size]
del labels[0:validation_size]

# tst_q = questions[0:validation_size]
# tst_a = answers[0:validation_size]
# tst_l = labels[0:validation_size]
#
# del questions[0:validation_size]
# del answers[0:validation_size]
# del labels[0:validation_size]

#zbior treningowy

q_train = np.random.randn( len(questions), 200)
a_train = np.random.randn( len(questions), 200)
l_train = np.random.randn( len(questions), 1)

i = 0
while i < len(questions):
    q_train[i, :] = questions[i]
    a_train[i, :] = answers[i]
    l_train[i, 0] = labels[i]
    i += 1

#zbior walidacyjny

q_valid = np.random.randn( len(val_q), 200)
a_valid = np.random.randn( len(val_q), 200)
l_valid = np.random.randn( len(val_q), 1)

i = 0
while i < len(val_q):
    q_valid[i, :] = val_q[i]
    a_valid[i, :] = val_a[i]
    l_valid[i, 0] = val_l[i]
    i += 1

# zbior testowy - na koncu wczytujemy osobny lecz mozna wydzielic z treningowego tym kodem ponizej
# q_test = np.random.randn( len(tst_q), 200)
# a_test = np.random.randn( len(tst_q), 200)
# l_test = np.random.randn( len(tst_q), 1)
#
# i = 0
# while i < len(tst_q):
#     q_test[i, :] = tst_q[i]
#     a_test[i, :] = tst_a[i]
#     l_test[i, 0] = tst_l[i]
#     i += 1

q_train = q_train.reshape(len(questions), 200, 1)
a_train = a_train.reshape(len(questions), 200,1 )

q_valid = q_valid.reshape(len(val_q), 200, 1)
a_valid = a_valid.reshape(len(val_q), 200, 1)

# q_test = q_test.reshape(len(tst_q), 200, 1)
# a_test = q_test.reshape(len(tst_q), 200, 1)

n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 10

def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

left_input = Input(shape=(200,1,))
right_input = Input(shape=(200,1,))
shared_lstm = LSTM(n_hidden, input_shape=(1, 200))
left_output = shared_lstm(left_input)
right_output = shared_lstm(right_input)

malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
malstm = Model([left_input, right_input], [malstm_distance])
optimizer = Adadelta(clipnorm=gradient_clipping_norm)
malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])


training_start_time = time()
malstm_trained = malstm.fit([q_train, a_train], l_train, batch_size=batch_size, epochs=n_epoch,
                            validation_data=([q_valid, a_valid], l_valid))

malstm.save('my.h5')
print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))
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


#malstm = load_model('my.h5', custom_objects={'exponent_neg_manhattan_distance': exponent_neg_manhattan_distance})

questionTest = pickle.load(open("questionVecTest.p", "rb"))
answerTest = pickle.load(open("answersVecTest.p", "rb"))
labelsTest = pickle.load(open("labelsTest.p", "rb"))

q_test = np.random.randn( len(questionTest), 200)
a_test = np.random.randn( len(questionTest), 200)
l_test = np.random.randn( len(questionTest), 1)

i = 0
while i < len(questionTest):
    q_test[i, :] = questionTest[i]
    a_test[i, :] = answerTest[i]
    l_test[i, 0] = labelsTest[i]
    i += 1

q_test = q_test.reshape(len(questionTest), 200, 1)
a_test = a_test.reshape(len(questionTest), 200,1)

eval = malstm.evaluate(x=[q_test, a_test], y=l_test, batch_size=64)
predict = malstm.predict([q_test, a_test])

print('\n# Predict on test data')
print(predict)

print('\n# Evaluate on test data')
print(eval)