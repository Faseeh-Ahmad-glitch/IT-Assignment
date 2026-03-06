import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

data = load_breast_cancer()
X_scaled = StandardScaler().fit_transform(data.data)
y = data.target

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

models = {
    'SVM': GridSearchCV(SVC(), {'C': [1, 10], 'gamma': [0.1, 0.01]}, cv=5).fit(X_train, y_train).best_estimator_,
    'LogReg': LogisticRegression().fit(X_train, y_train),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
}

print('--- Final Comparison (Best in Class) ---')
for name, model in models.items():
    pred = model.predict(X_test)
    print(f'\n{name} - Accuracy: {accuracy_score(y_test, pred):.2%}, F1: {f1_score(y_test, pred):.4f}')