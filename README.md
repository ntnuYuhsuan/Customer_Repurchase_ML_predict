# Customer_Repurchase_ML_predict

```
numpy                     1.26.4
pandas                    2.2.2
joblib                    1.4.2
scikit-learn              1.4.2
imbalanced-learn          0.12.2
xgboost                   2.0.3
jupyter                   1.0.0
```

### 路徑設定

raw = pd.read_csv('post_PCA.csv')

```base
python customer_repurchase_classific.py
```

## Result

```
Model: Logistic Regression
[[19690   205]
 [    1   104]]
              precision    recall  f1-score   support

       False       1.00      0.99      0.99     19895
        True       0.34      0.99      0.50       105

    accuracy                           0.99     20000
   macro avg       0.67      0.99      0.75     20000
weighted avg       1.00      0.99      0.99     20000

Accuracy: 0.99
Feature Importance:
      Feature  Importance
5   recency_m  395.362403
33  label_PC9  201.174420
25  label_PC1  178.359125

Model: Random Forest
[[19821    74]
 [   97     8]]
              precision    recall  f1-score   support

       False       1.00      1.00      1.00     19895
        True       0.10      0.08      0.09       105

    accuracy                           0.99     20000
   macro avg       0.55      0.54      0.54     20000
weighted avg       0.99      0.99      0.99     20000

Accuracy: 0.99
Feature Importance:
      Feature  Importance
28  label_PC4    0.102721
4   ternure_m    0.080565
27  label_PC3    0.067778

Model: KNN
[[18105  1790]
 [   62    43]]
              precision    recall  f1-score   support

       False       1.00      0.91      0.95     19895
        True       0.02      0.41      0.04       105

    accuracy                           0.91     20000
   macro avg       0.51      0.66      0.50     20000
weighted avg       0.99      0.91      0.95     20000

Accuracy: 0.91

Model: Decision Tree
[[17251  2644]
 [   49    56]]
              precision    recall  f1-score   support

       False       1.00      0.87      0.93     19895
        True       0.02      0.53      0.04       105

    accuracy                           0.87     20000
   macro avg       0.51      0.70      0.48     20000
weighted avg       0.99      0.87      0.92     20000

Accuracy: 0.87
Feature Importance:
      Feature  Importance
28  label_PC4    0.297387
4   ternure_m    0.214886
33  label_PC9    0.133854

Model: XGBoost
[[19779   116]
 [   42    63]]
              precision    recall  f1-score   support

       False       1.00      0.99      1.00     19895
        True       0.35      0.60      0.44       105

    accuracy                           0.99     20000
   macro avg       0.67      0.80      0.72     20000
weighted avg       0.99      0.99      0.99     20000

Accuracy: 0.99
Feature Importance:
                Feature  Importance
28            label_PC4    0.120474
16          AHb_POL_CNT    0.064725
20  product_density_his    0.063762

Model: SVD+Logistic Regression
[[19694   201]
 [    1   104]]
              precision    recall  f1-score   support

       False       1.00      0.99      0.99     19895
        True       0.34      0.99      0.51       105

    accuracy                           0.99     20000
   macro avg       0.67      0.99      0.75     20000
weighted avg       1.00      0.99      0.99     20000

Accuracy: 0.99
```