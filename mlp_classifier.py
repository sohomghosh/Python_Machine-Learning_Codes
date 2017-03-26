from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(hidden_layer_sizes=(15,4),alpha=1e-5, random_state=1)
clf.fit(X_train, Y_train)
probabs=clf.predict_proba(X_test)
Y_predict=clf.predict(X_test)
np.set_printoptions(threshold=np.inf)
