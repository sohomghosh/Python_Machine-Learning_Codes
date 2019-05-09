X = data[features]
y = pd.get_dummies(data['target']).as_matrix()
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
clf.fit(X, y)
ans = clf.predict_proba(X)
output_df = pd.DataFrame({})
cnt = 1
for col in [i[:,1] for i in ans]:
  output_df[str(cnt)] = col
  cnt = cnt + 1

#Extracting indices/column_names of the instances having maximum balue in a row
output_df.idxmax(axis = 1)
