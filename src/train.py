import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocess import load_data, clean_data, encode_data, select_features



# Load and preprocess data
df = load_data("../data/startup_success_prediction.csv")

df = clean_data(df)

df = encode_data(df)

X, y = select_features(df)


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train multiple models

lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Accuracy comparison

lr_acc = accuracy_score(y_test, lr.predict(X_test))
dt_acc = accuracy_score(y_test, dt.predict(X_test))
rf_acc = accuracy_score(y_test, rf.predict(X_test))

print("Logistic Regression Accuracy:", lr_acc)
print("Decision Tree Accuracy:", dt_acc)
print("Random Forest Accuracy:", rf_acc)


# Hyperparameter tuning for Random Forest

params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, 15]
}

grid = GridSearchCV(RandomForestClassifier(), params)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Final accuracy

final_acc = accuracy_score(y_test, best_model.predict(X_test))

print("Best Model Accuracy after tuning:", final_acc)


# Save model and features

joblib.dump(best_model, "../models/model.pkl")

joblib.dump(X.columns.tolist(), "../models/features.pkl")

print("Best model saved successfully")
