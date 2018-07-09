import pickle
import xgboost as xgb

#Save a model
pickle.dump(xgb1, open("xgb1.pickle.dat", "wb"))

#To load the saved model
loaded_model = pickle.load(open("xgb1.pickle.dat", "rb"))
