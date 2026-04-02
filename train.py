import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load data
df = pd.read_csv('data/Employee.csv')

# Convert target
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Features
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Encode
X = pd.get_dummies(X, drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost
model = XGBClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

# Save columns (VERY IMPORTANT for app)
pickle.dump(X.columns, open('columns.pkl', 'wb'))