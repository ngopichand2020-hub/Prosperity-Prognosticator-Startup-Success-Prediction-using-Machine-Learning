import joblib
from sklearn.metrics import classification_report

from preprocess import load_data, clean_data, encode_data, split_data


model = joblib.load("../models/model.pkl")

df = load_data("../data/startup_success_prediction.csv")

df = clean_data(df)

df = encode_data(df)

X, y = split_data(df)

predictions = model.predict(X)

print(classification_report(y, predictions))
