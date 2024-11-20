import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_scores, f1_score, classification_report, confusion_matrix,
    roc_curve, auc
)

import warnings
warnings.filterwarnings('ignore')

sns.set(style="whitegrid")


data = pd.read_csv("revenue-budget-financial-plan-qtr1-1 (1).csv")

print("Veri Setinin İlk 5 Satırı:")
print(data.head())

print("\nVeri Setinin Bilgisi:")
print(data.info())

print("\nEksik Değerlerin Sayısı:")
print(data.isnull().sum())



target = 'Year 4 Revenue Amount'

if target in data.columns:
    y = data[target]
    X = data.drop([
        'Publication Date', 'Fiscal Year', 'Revenue Category Name', 'Revenue Class Name',
        'Revenue Source Name', 'Revenue Structure Description', 'Adopted Budget Amount',
        'Current Modified Budget Amount', 'Year 1 Revenue Amount', 'Year 2 Revenue Amount',
        'Year 3 Revenue Amount', 'Year 4 Revenue Amount'
    ], axis=1)
else:
    raise ValueError(f"Hedef değişken '{target}' sütunu bulunamadı.")

print("\nHedef Değişken ve Özellikler Belirlendi.")
print("Özellikler:")
print(X.columns.tolist())

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print("\nKategorik Değişkenler:")
print(categorical_cols)

encoder = OrdinalEncoder()
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

print("\nOrdinal Encoding Sonrası Özelliklerin İlk 5 Satırı:")
print(X.head())


constant_columns = [col for col in X.columns if X[col].nunique() <= 1]
print("\nSabit Sütunlar:", constant_columns)

if constant_columns:
    X = X.drop(columns=constant_columns)
    print("\nSabit Sütunlar Kaldırıldıktan Sonra X'in Boyutu:", X.shape[1])
else:
    print("\nSabit sütun bulunamadı.")


median_value = y.median()
y_binary = y.apply(lambda x: 1 if x > median_value else 0)

print("\nHedef Değişken Binarizasyonu Yapıldı.")
print(y_binary.value_counts())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nÖzellikler Ölçeklendirildi.")


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print("\nEğitim Seti Boyutu:", X_train.shape)
print("Test Seti Boyutu:", X_test.shape)


mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_results = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})
mi_results = mi_results.sort_values(by='MI_Score', ascending=False)

top_n = 20
significant_features = mi_results['Feature'].head(top_n).tolist()
print("\nEn Önemli 20 Özellik:")
print(significant_features)

plt.figure(figsize=(12, 8))
sns.barplot(x='MI_Score', y='Feature', data=mi_results.head(top_n), palette='viridis')
plt.title('En Önemli 20 Özelliğin Mutual Information Skorları')
plt.xlabel('Mutual Information Skoru')
plt.ylabel('Özellikler')
plt.show()


selected_indices = [X.columns.get_loc(f) for f in significant_features]
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

print("\nSeçilen Özelliklerle Eğitim Seti Boyutu:", X_train_selected.shape)
print("Seçilen Özelliklerle Test Seti Boyutu:", X_test_selected.shape)

selected_features_df = pd.DataFrame(X_train_selected, columns=significant_features)
plt.figure(figsize=(15, 12))
correlation_matrix_selected = selected_features_df.corr()
sns.heatmap(correlation_matrix_selected, cmap='coolwarm', linewidths=0.5, annot=True)
plt.title('Seçilen 10 Özellik Arasındaki Korelasyon Matrisi')
plt.xlabel('Özellikler')
plt.ylabel('Özellikler')
plt.show()


lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_selected, y_train)
y_pred_lr = lr_model.predict(X_test_selected)

accuracy_lr = accuracy_scores(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print("\nModel Performansları (Logistic Regression):")
print(f"Accuracy: {accuracy_lr:.4f}")
print(f"F1 Score: {f1_lr:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

y_pred_proba_lr = lr_model.predict_proba(X_test_selected)[:, 1]
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred_proba_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc_lr:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Logistic Regression')
plt.legend(loc="lower right")
plt.show()

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_selected, y_train)
y_pred_dt = dt_model.predict(X_test_selected)

accuracy_dt = accuracy_scores(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print("\nModel Performansları (Decision Tree Classifier):")
print(f"Accuracy: {accuracy_dt:.4f}")
print(f"F1 Score: {f1_dt:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

y_pred_proba_dt = dt_model.predict_proba(X_test_selected)[:, 1]
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, y_pred_proba_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, color='green', lw=2, label=f'ROC curve (area = {roc_auc_dt:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Decision Tree Classifier')
plt.legend(loc="lower right")
plt.show()

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_selected, y_train)
y_pred_rf = rf_model.predict(X_test_selected)

accuracy_rf = accuracy_scores(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("\nModel Performansları (Random Forest Classifier):")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

y_pred_proba_rf = rf_model.predict_proba(X_test_selected)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='purple', lw=2, label=f'ROC curve (area = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Random Forest Classifier')
plt.legend(loc="lower right")
plt.show()

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_selected, y_train)
y_pred_gb = gb_model.predict(X_test_selected)

accuracy_gb = accuracy_scores(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)

print("\nModel Performansları (Gradient Boosting Classifier):")
print(f"Accuracy: {accuracy_gb:.4f}")
print(f"F1 Score: {f1_gb:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_gb))

y_pred_proba_gb = gb_model.predict_proba(X_test_selected)[:, 1]
fpr_gb, tpr_gb, thresholds_gb = roc_curve(y_test, y_pred_proba_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_gb, tpr_gb, color='orange', lw=2, label=f'ROC curve (area = {roc_auc_gb:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Gradient Boosting Classifier')
plt.legend(loc="lower right")
plt.show()


param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)
grid_rf.fit(X_train_selected, y_train)

print("\nEn İyi Parametreler (Random Forest Classifier):", grid_rf.best_params_)
print("En İyi Cross-Validation F1 Skoru:", grid_rf.best_score_)

best_rf_model = grid_rf.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test_selected)

accuracy_best_rf = accuracy_scores(y_test, y_pred_best_rf)
f1_best_rf = f1_score(y_test, y_pred_best_rf)

print("\nModel Performansları (En İyi Random Forest Classifier):")
print(f"Accuracy: {accuracy_best_rf:.4f}")
print(f"F1 Score: {f1_best_rf:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best_rf))

y_pred_proba_best_rf = best_rf_model.predict_proba(X_test_selected)[:, 1]
fpr_best_rf, tpr_best_rf, thresholds_best_rf = roc_curve(y_test, y_pred_proba_best_rf)
roc_auc_best_rf = auc(fpr_best_rf, tpr_best_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_best_rf, tpr_best_rf, color='cyan', lw=2, label=f'ROC curve (area = {roc_auc_best_rf:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - En İyi Random Forest Classifier')
plt.legend(loc="lower right")
plt.show()


models = ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier',
          'Gradient Boosting Classifier', 'En İyi Random Forest Classifier']
f1_scores = [f1_lr, f1_dt, f1_rf, f1_gb, f1_best_rf]
accuracy_scores = [accuracy_lr, accuracy_dt, accuracy_rf, accuracy_gb, accuracy_best_rf]

plt.figure(figsize=(14, 7))
sns.barplot(x=models, y=f1_scores, palette='viridis')
plt.title('Modellerin F1 Skorları')
plt.xlabel('Modeller')
plt.ylabel('F1 Skoru')
plt.ylim(0, 1)
plt.show()

plt.figure(figsize=(14, 7))
sns.barplot(x=models, y=accuracy_scores, palette='magma')
plt.title('Modellerin Accuracy Skorları')
plt.xlabel('Modeller')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()


importances = best_rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': significant_features,
    'Importance': importances
})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='magma')
plt.title('En İyi Random Forest Classifier Özellik Önemi')
plt.xlabel('Özellik Önemi')
plt.ylabel('Özellikler')
plt.show()

