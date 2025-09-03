import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK


data = pd.read_excel("data/Final_Payment.xlsx")

# Feature Engineering
data['hour'] = (data['step'] - 1) % 24
data = pd.get_dummies(data, columns=['type'])
data.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True, errors='ignore')


X = data.drop('isFraud', axis=1)
y = data['isFraud']
X.fillna(X.mean(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, stratify=y, random_state=42
)


imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


search_space = {
    'n_estimators': hp.choice('n_estimators', [300, 500, 700]),
    'max_depth': hp.choice('max_depth', [20, 50, None]),
}

def objective(params):
    model = ExtraTreesClassifier(**params, class_weight='balanced', random_state=42)
    accuracy = cross_val_score(model, X_train_smote, y_train_smote, cv=3).mean()
    return {'loss': -accuracy, 'status': STATUS_OK}

trials = Trials()
best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=30, trials=trials)


best_model = ExtraTreesClassifier(**best_params, class_weight='balanced', random_state=42)
best_model.fit(X_train_smote, y_train_smote)


y_pred = best_model.predict(X_test)


print("Optimized Extra Trees Model (TPE + SMOTE)")
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Optimized Extra Trees - TPE + SMOTE')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics = ['Precision', 'Recall', 'F1-Score']
values = [precision, recall, f1]

plt.figure(figsize=(8, 5))
sns.barplot(x=metrics, y=values, palette='Set2')
plt.ylim(0, 1)
plt.title('Precision, Recall and F1-Score')
plt.ylabel('Score')
plt.tight_layout()
plt.show()
