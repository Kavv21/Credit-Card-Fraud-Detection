import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


df = pd.read_excel("data/Final_Payment.xlsx")
df.columns = df.columns.str.strip()

fraud_column = None
for col in df.columns:
    if df[col].nunique() == 2 and sorted(df[col].unique()) == [0, 1]:
        fraud_column = col
        break

if fraud_column is None:
    raise KeyError("No valid fraud column found! Please check dataset structure.")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)



X = df.drop(columns=[fraud_column])
y = df[fraud_column]


X = pd.get_dummies(X, drop_first=True)


rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X, y)

selector = SelectFromModel(rf_selector, threshold="median", prefit=True)
X_selected = selector.transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.3, stratify=y, random_state=42
)


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

def adjust_threshold(model, X_test, threshold=0.6):
    probs = model.predict_proba(X_test)[:, 1]
    return (probs >= threshold).astype(int)

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_resampled, y_train_resampled)

y_lr_pred = adjust_threshold(lr, X_test)

print("\nLogistic Regression Results:")
print(classification_report(y_test, y_lr_pred))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_lr_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

f1 = f1_score(y_test, y_lr_pred)
recall = recall_score(y_test, y_lr_pred)
precision = precision_score(y_test, y_lr_pred)

metrics = ['F1 Score', 'Recall', 'Precision']
scores = [f1, recall, precision]

plt.figure(figsize=(8, 5))
bars = plt.barh(metrics, scores, color=['skyblue', 'lightgreen', 'salmon'])
plt.xlim(0.0, 1.0)
plt.title("KPI Metrics - Logistic Regression with SMOTE & Feature Selection")

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.3f}', va='center')

plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
