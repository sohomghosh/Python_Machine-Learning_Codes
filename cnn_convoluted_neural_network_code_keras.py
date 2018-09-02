#Reference: https://www.kaggle.com/drissaitlabsir27/spooky-glove-cnn-bilstm
def keras_cnn():
    # create model
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=3,padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=3,padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=NB_CLASSES, activation='softmax'))
    # Compile model
    #opti = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    #opti = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    #opti = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    #opti = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    opti = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(optimizer=opti,
                  loss='categorical_crossentropy',
                  metrics=['acc']
                 )
    return model


# Training the model
print("model fitting - cnn")
model_cnn = keras_cnn()
# summarize the model
model_cnn.summary()
# fit the model
history_cnn = model_cnn.fit(s_train,
                            l_train, 
                            validation_data=(s_val, l_val),
                            epochs = 10,
                            batch_size = 1024,
                            callbacks = callbacks_list,
                            verbose=2
                           )

# evaluate the model
loss, accuracy = model_cnn.evaluate(train_sequences_pad, dummy_labels, verbose=2)
print('Accuracy: %f' % (accuracy*100))
print('Loss: %f' % (loss))

predictions = model_cnn.predict(test_sequences_pad)
columns = list(le.classes_)
prediction = pd.DataFrame(predictions,columns = columns)
prediction.insert(loc = 0, column = 'id', value = test_ids)
prediction.head()
