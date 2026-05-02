import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pickle
import os

os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("data/hr_data.csv")

# Drop unnecessary column
df.drop(['EmployeeNumber'], axis=1, inplace=True)

# Convert target
df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})

# ✅ ADD YOUR CODE HERE (IMPORTANT)
selected_features = [
    'Age',
    'MonthlyIncome',
    'JobLevel',
    'YearsAtCompany',
    'OverTime'
]

df = df[selected_features + ['Attrition']].copy()

# Convert OverTime to numeric
df['OverTime'] = df['OverTime'].map({'Yes':1, 'No':0})

#print(df.head())
#print(df.dtypes)

# -----------------------------
# Features & target
# -----------------------------
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("Training Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance
feature_importance = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.savefig("model/feature_importance.png")
plt.close()

importance_df.to_csv("model/feature_importance.csv", index=False)

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))

print("Model trained and saved successfully!")