from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans

mdl = FuzzyKMeans(k=7, m=2)
mdl.fit(np.concatenate([x_km,x_km_test],axis = 0))
train_fcm_pred = list(mdl.labels_)[0:x_km.shape[0]]
test_fcm_pred = list(mdl.labels_)[x_km.shape[0]:]

