# Author Sara Al-Emadi
############################## Libraries ################################
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense
from keras import optimizers
from keras.layers import Conv1D,MaxPooling1D, Flatten,AveragePooling1D
from keras.layers import Conv2D, GlobalMaxPooling1D
from keras.layers.core import Reshape
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense, Dropout
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999,amsgrad=False)
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
number_epoch         = 350
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
y = encode(Label_2)
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
    cnn.add(AveragePooling1D(3))

    cnn.add(Conv1D(64,3, activation='relu',padding='same'))
    cnn.add(AveragePooling1D(3))

    cnn.add(Conv1D(64,3,  activation='relu',padding='same'))
    cnn.add(Conv1D(128,3, activation='relu',padding='same'))
    cnn.add(Conv1D(128,3, activation='relu',padding='same'))
    cnn.add(AveragePooling1D(3))

    cnn.add(Conv1D(256,3, activation='relu',padding='same'))
    cnn.add(Dropout(0.2))

    cnn.add(Flatten())
    cnn.add(Dense(256, activation = inner_activation_fun))
    cnn.add(Dense(y.shape[1],activation='softmax'))
    
    print(cnn.summary())

    print('Compiling')
    cnn.compile(loss = 'categorical_crossentropy', optimizer = optimizer_algorithm, metrics =         ['accuracy'])
    print('Compilation is complete')
    print('fitting the model')
    #cnn.fit(x[train], y[train], epochs = number_epoch, batch_size = batch_length, verbose = show_inter_results)
    cnn.fit(x[train], y[train], batch_size=batch_length , epochs=number_epoch ,verbose=show_inter_results)
    #cnn.fit(x[train], y[train],batch_size=batch_length,epochs=number_epoch,verbose = show_inter_results)
    #print(cnn.summary())

    print('fitting complete \n Evaluating:')
    scores = cnn.evaluate(x[test], y[test], verbose = show_inter_results)
    print(scores[1]*100)
    cvscores.append(scores[1]*100)
    print('Predicting the final results')
    y_pred = cnn.predict(x[test])
    rounded_predictions = cnn.predict_classes(x[test],verbose=1)
    rounded_labels=np.argmax(y[test], axis=1)
    print('Precision: ',precision_score(rounded_labels, rounded_predictions, average="macro"))
    print('Recall: ',recall_score(rounded_labels, rounded_predictions, average="macro"))
    print('F1_Score: ',f1_score(rounded_labels, rounded_predictions, average="macro"))
    print('Accuracy: ',accuracy_score(rounded_labels, rounded_predictions))
    cm = confusion_matrix(rounded_labels, rounded_predictions)
    print(cm)
    np.savetxt("Classification_Results/Results_new_k_label_1_%s.csv" % cnt, np.column_stack((y[test], y_pred)), delimiter=",", fmt='%s')
#########################################################################
