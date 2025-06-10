import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
FileNotFoundError: ...
df = pd.read_csv(r"C:\Users\CSE\Desktop\ml gender gap\data.csv")


# Encode categorical column
df['C_api'] = LabelEncoder().fit_transform(df['C_api'])

# Feature/target split
X = df.drop(columns=['gender'])
y = df['gender']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ XGBoost Accuracy: {accuracy:.4f}")

# F1 Score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"🎯 F1 Score (weighted): {f1:.4f}")

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
