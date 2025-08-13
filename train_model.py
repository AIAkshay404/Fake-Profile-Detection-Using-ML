# model_training.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('fake_profiles_extended.csv')
X = df.drop('label', axis=1)
y = df['label']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'model.pkl')  # ✅ Correct name and method
print("✅ Model saved as model.pkl")
