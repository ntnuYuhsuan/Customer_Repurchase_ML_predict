import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from collections import Counter
# import random

raw = pd.read_csv('post_PCA.csv')
scaler = joblib.load('Models/scaler.joblib') # trained MinMaxScaler

def feature_filter(model, df):
    # 如果模型具有係數屬性，例如邏輯迴歸
    if hasattr(model, 'coef_'):
        importance = model.coef_[0]
        feature_importance = pd.DataFrame({'Feature': df.drop(['Y1_repurchase', 'REPUR_FLG'], axis=1).columns, 'Importance': importance})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        print("Feature Importance:")
        print(feature_importance[:3])
    
    # 如果模型具有特徵重要性屬性，例如隨機森林和XGBoost
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': df.drop(['Y1_repurchase', 'REPUR_FLG'], axis=1).columns, 'Importance': importance})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        print("Feature Importance:")
        print(feature_importance[:3])

random_state = 313
df = raw
# X = df.drop(['Y1_repurchase', 'recency_m'], axis=1)
# X = df.drop(['Y1_repurchase', 'REPUR_FLG'], axis=1)
X = df.drop(['Y1_repurchase'], axis=1)
y = df['Y1_repurchase']
# y = df['REPUR_FLG']
# y = df.index.isin(analysis[analysis['cluster'] == 0].index)
# 標準化特徵
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)
X_scaled = scaler.transform(X)
# joblib.dump(scaler, 'Models/scaler.joblib')
# joblib.dump(scaler, 'Models/RF_scaler.joblib')

rus = RandomUnderSampler(random_state=random_state)
smote = SMOTE(sampling_strategy=0.2,random_state=random_state,)
# smote = SMOTE(random_state=random_state)

# X_test_real, y_test_real = rus.fit_resample(X_scaled, y)
# X_scaled, X_test, y_scaled, y_test = train_test_split(X_scaled, y, test_size=0.4, stratify=y, random_state=random_state)
X_scaled, X_test, y_scaled, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state)
# X_test, y_test = rus.fit_resample(X_test, y_test)

X_scaled, y_scaled = smote.fit_resample(X_scaled, y_scaled)
# X_scaled, y = smote.fit_resample(X_scaled, y)
X_train, y_train = rus.fit_resample(X_scaled, y_scaled)

# X_scaled, y = rus.fit_resample(X_scaled, y)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state)

models = {
    'Logistic Regression': LogisticRegression(C=1000, max_iter=1000),
    'Random Forest': RandomForestClassifier(max_depth=80, n_estimators=300),
    # 'Random Forest': RandomForestClassifier(max_depth=80, n_estimators=200),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    # 'KNN': KNeighborsClassifier(n_neighbors=9),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_split=3),
    # 'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_split=2),
    'XGBoost': XGBClassifier(learning_rate=0.15, max_depth=4, n_estimators=200),
    # 'XGBoost': XGBClassifier(learning_rate=0.13, max_depth=3, n_estimators=150),
    # 'SVM': SVC(C=500, gamma=0.1, kernel='linear', probability=True), # probability=True
    # 'SVM': SVC(C=800, gamma=0.1, kernel='linear', probability=True),
    'SVD+Logistic Regression': make_pipeline(TruncatedSVD(n_components=50), LogisticRegression(C=1000, max_iter=1000))
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model: {name}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy:.2f}")
    
    feature_filter(model, df)
    
    # joblib.dump(model, f"Models/RF_{name.replace(' ', '_')}_{accuracy:.2f}.joblib")
    print()