#Source: https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

def one_hot_encode_fns(col_to_encode,train):
	integer_encoded = label_encoder.fit_transform(train[col_to_encode].values).reshape(train.shape[0], 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	onehot_encoded_df = pd.DataFrame(onehot_encoded)
	onehot_encoded_df.index = train.index
	onehot_encoded_df.columns = [col_to_encode + "one_hot" + str(i) for i in range(onehot_encoded_df.shape[1])]
	return onehot_encoded_df


col_to_encode = "C6"
#train[col_to_encode] has values like True, False, True
onehot_df = one_hot_encode_fns(col_to_encode,train)
train = pd.concat([train, onehot_df], axis = 1)
del train[col_to_encode]
