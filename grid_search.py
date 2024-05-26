import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
raw = pd.read_csv('post_PCA.csv')
scaler = joblib.load('Models/scaler.joblib') # trained MinMaxScaler
random_state = 313
df = raw
# X = df.drop(['Y1_repurchase', 'REPUR_FLG'], axis=1)
X = df.drop(['Y1_repurchase'], axis=1)
y = df['Y1_repurchase']
X_scaled = scaler.transform(X)
rus = RandomUnderSampler(random_state=random_state)
X_scaled, y = rus.fit_resample(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier(),
    'SVM': SVC(),
    'SVD+Logistic Regression': make_pipeline(TruncatedSVD(), LogisticRegression())
}
param_grid = {
    # 'Logistic Regression': {'C': [500, 1000]},
    # 'Random Forest': {'n_estimators': [150, 200, 300, 450], 'max_depth': [40, 50, 60, 80]},
    # 'KNN': {'n_neighbors': [3, 5, 7, 9]},
    # 'Decision Tree': {'max_depth': [8, 10, 12], 'min_samples_split': [2, 3, 4]},
    # 'XGBoost': {'learning_rate': [0.13, 0.15, 0.2, 0.5], 
    #             'n_estimators': [150, 200, 250], 'max_depth': [3, 4, 5]},
    # 'SVM': {'C': [350, 500, 800],'kernel': ['linear', 'rbf', 'poly'],
    #           'gamma': [0.1, 0.15, 0.08]},
    'SVD+Logistic Regression': {'logisticregression__C': [200, 300, 500, 1000]}
}

# 使用網格搜索進行參數優化
for name, model in models.items():
    if name in param_grid:
        grid_search = GridSearchCV(model, param_grid[name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(f"Best parameters for {name}: {best_params}")
        print(f"Best cross-validation accuracy: {best_score:.2f}")