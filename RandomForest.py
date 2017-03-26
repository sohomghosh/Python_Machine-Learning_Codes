from sklearn import ensemble
clf=ensemble.RandomForestClassifier(n_estimators=5000, criterion='entropy',max_depth=3,max_features='auto')
clf.fit(X_train, Y_train)
probabs=clf.predict_proba(X_test)
Y_predict=clf.predict(X_test)
np.set_printoptions(threshold=np.inf)
