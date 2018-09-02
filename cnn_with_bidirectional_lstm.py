#Reference: https://www.kaggle.com/drissaitlabsir27/spooky-glove-cnn-bilstm
def cnn_bi_lstm():
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(TimeDistributed(Dense(64)))
    model.add(Bidirectional(LSTM(32,recurrent_dropout=0.2,dropout=0.2)))
    model.add(Dense(NB_CLASSES, activation='softmax'))
    opti = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opti, 
                  metrics=['acc']
                 )
    return model

# Training the model
print("model fitting - cnn_bidirectional_lstm")
model_cnn_bilstm = cnn_bi_lstm()
# summarize the model
model_cnn_bilstm.summary()
# fit the model
history_cnn_bi_lstm = model_cnn_bilstm.fit(s_train,
                                           l_train, 
                                           validation_data=(s_val, l_val),
                                           epochs = 10,
                                           batch_size = 1024,
                                           callbacks = callbacks_list,
                                           verbose=2
                                          )

print("*****"*5)
print("Losses:\n", accuracy_losses_lr.losses)
print("*****"*5)
print("Accuracies:\n", accuracy_losses_lr.accuracy)
print("*****"*5)
print("Learning rates:\n", accuracy_losses_lr.lr)
print("*****"*5)

# evaluate the model
loss, accuracy = model_cnn_bilstm.evaluate(train_sequences_pad, dummy_labels, verbose=2)
print('Accuracy: %f' % (accuracy*100))
print('Loss: %f' % (loss))

predictions = model_cnn_bilstm.predict(test_sequences_pad)
columns = list(le.classes_)
prediction = pd.DataFrame(predictions,columns = columns)
prediction.insert(loc = 0, column = 'id', value = test_ids)
prediction.head()
