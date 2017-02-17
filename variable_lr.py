'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
import keras.callbacks as K
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler

#history=[]


# learning rate schedule
#def step_decay(epoch):
#	initial_lrate = 0.0082
#	epochs_drop = 1.0
#	lrate = initial_lrate * np.exp(history.losses[-1])
#	return lrate
sd=[]
class LossHistory(K.Callback):
    def on_train_begin(self, logs={}):
        self.losses = [1,1]

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        sd.append(step_decay(len(self.losses)))
        print('learning rate:', step_decay(len(self.losses)))
        print('derivative of loss:', 2*np.sqrt((self.losses[-1])))



def step_decay(losses):
    if float((np.array(temp_history.losses[-1])))<3.0:
        lrate=0.060*np.exp(np.array(temp_history.losses[-1]))
        return lrate
    else:
        lrate=0.01
        return lrate

        

batch_size = 128
nb_classes = 10
nb_epoch = 40

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(100, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

sgd = SGD(lr=0.0, momentum=0.8)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
#def step_decay(losses):
#    if float(history.losses[-1])<4:
#        lrate=0.1*np.exp(history.losses)
#        momentum=0.8
#        decay_rate=2e-6
#        return lrate
#    else:
#        lrate=0.1
#        return lrate

temp_history = LossHistory()
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate, temp_history]

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test), callbacks=callbacks_list)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])