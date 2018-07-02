from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pandas as pd

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

features = list(set(list(train.columns)) - set(['ID', 'target']))
train_nor = minmax_scale(train[features], axis = 0)
test_nor = minmax_scale(test[features], axis = 0)

ncol = train_nor.shape[1]
X_train, X_valid, Y_train, Y_valid = train_test_split(train_nor, train['target'], train_size = 0.8)

encoding_dimension = 10
input_dim = Input(shape = (ncol, ))

# Encoder Layers
encoded1 = Dense(300, activation = 'relu')(input_dim)
encoded2 = Dense(250, activation = 'relu')(encoded1)
encoded3 = Dense(100, activation = 'relu')(encoded2)
encoded4 = Dense(encoding_dimension, activation = 'relu')(encoded3)

# Decoder Layers
decoded1 = Dense(100, activation = 'relu')(encoded4)
decoded2 = Dense(250, activation = 'relu')(decoded1)
decoded3 = Dense(300, activation = 'relu')(decoded2)
decoded4 = Dense(ncol, activation = 'sigmoid')(decoded3)

# Combine Encoder and Deocder layers
autoencoder = Model(inputs = input_dim, outputs = decoded4)

# Compile the Model
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

autoencoder.summary()

autoencoder.fit(X_train, X_train, nb_epoch = 10, batch_size = 128, shuffle = False, validation_data = (X_valid, X_valid))

encoder = Model(inputs = input_dim, outputs = encoded4)
encoded_input = Input(shape = (encoding_dimension, ))

encoded_train = pd.DataFrame(encoder.predict(train_nor))
encoded_train = encoded_train.add_prefix('col_')

encoded_test = pd.DataFrame(encoder.predict(test_nor))
encoded_test = encoded_test.add_prefix('col_')

encoded_train['target']=train['target']
encoded_test['ID']=test['ID']

encoded_train.to_csv('auto_encoder_train.csv',index=False)
encoded_test.to_csv('auto_encoder_test.csv',index=False)



