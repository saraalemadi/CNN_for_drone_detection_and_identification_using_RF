#########################################################################
# Author by Sara Al-Emadi
############################## Libraries ################################
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense
from keras.layers import Conv1D,MaxPooling1D, Flatten
from keras.layers import Conv2D, GlobalMaxPooling1D
from keras.layers.core import Reshape
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense, Dropout
############################## Functions ###############################
def decode(datum):
    y = np.zeros((datum.shape[0],1))
    for i in range(datum.shape[0]):
        y[i] = np.argmax(datum[i])
    return y
def encode(datum):
    return to_categorical(datum)
############################# Parameters ###############################
np.random.seed(1)
K                    = 10
inner_activation_fun = 'relu'
outer_activation_fun = 'sigmoid'
optimizer_loss_fun   = 'mse'
optimizer_algorithm  = 'adam'
number_inner_layers  = 3
number_inner_neurons = 256
number_epoch         = 100
batch_length         = 50
show_inter_results   = 2
############################### Loading ##################################
print("Loading Data ...")
Data = np.loadtxt("RF_Data.csv", delimiter=",")
############################## Splitting #################################
print("Preparing Data ...")
x = np.transpose(Data[0:2047,:])
Label_1 = np.transpose(Data[2048:2049,:]); Label_1 = Label_1.astype(int);
Label_2 = np.transpose(Data[2049:2050,:]); Label_2 = Label_2.astype(int);
Label_3 = np.transpose(Data[2050:2051,:]); Label_3 = Label_3.astype(int);
y = encode(Label_1)
################################ Main ####################################
cvscores    = []
cnt         = 0
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)
for train, test in kfold.split(x, decode(y)):
    cnt = cnt + 1
    print(cnt)
    cnn = Sequential()
    print(x.shape[1],1)
    print('x_train shape:', x[train].shape)
    cnn = Sequential()
    cnn.add(Reshape((x.shape[1], 1), input_shape=(x.shape[1], )))
    
    cnn.add(Conv1D(32,3,  activation='relu',padding='same'))
    cnn.add(MaxPooling1D(3))
    cnn.add(Conv1D(64,3, activation='relu',padding='same'))
    cnn.add(MaxPooling1D(3))
    cnn.add(Conv1D(128,3,  activation='relu',padding='same'))
    cnn.add(MaxPooling1D(3))
    cnn.add(Conv1D(128,3,  activation='relu',padding='same'))
    cnn.add(MaxPooling1D(3))
    cnn.add(Dropout(0.25))
    cnn.add(Flatten())
    cnn.add(Dense(256, activation = inner_activation_fun))
    
    cnn.add(Dense(y.shape[1],activation='sigmoid'))
    
    print('Compiling')
    cnn.compile(loss = optimizer_loss_fun, optimizer = optimizer_algorithm, metrics =         ['accuracy'])
    print('Compilation is complete')
    print('fitting the model')
    #cnn.fit(x[train], y[train], epochs = number_epoch, batch_size = batch_length, verbose = show_inter_results)
    cnn.fit(x[train], y[train], batch_size=batch_length , epochs=number_epoch , validation_data=(x[test], y[test]),verbose=show_inter_results)
    #cnn.fit(x[train], y[train],batch_size=batch_length,epochs=number_epoch,verbose = show_inter_results)
    print(cnn.summary())

    print('fitting complete \n Evaluating:')
    scores = cnn.evaluate(x[test], y[test], verbose = show_inter_results)
    print(scores[1]*100)
    cvscores.append(scores[1]*100)
    print('Predicting the final results')
    y_pred = cnn.predict(x[test])
    np.savetxt("Classification_Results/Results_9%s.csv" % cnt, np.column_stack((y[test], y_pred)), delimiter=",", fmt='%s')
#########################################################################
