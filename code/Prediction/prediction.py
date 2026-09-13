# ============================================================
# 1. IMPORT THƯ VIỆN
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ============================================================
# 2. ĐỌC DỮ LIỆU
# ============================================================

file_path = "data/processed/diabetic_cleaned_data.csv"
df = pd.read_csv(file_path)
print("=" * 60)
print("THÔNG TIN DỮ LIỆU")
print("=" * 60)

print("Kích thước dữ liệu:", df.shape)
print("\nCác cột:")
print(df.columns.tolist())

print("\nPhân bố target:")
print(df["target_30days"].value_counts())

print("\nTỷ lệ target:")
print(df["target_30days"].value_counts(normalize=True))

# ============================================================
# 3. XÁC ĐỊNH TARGET VÀ FEATURE
# ============================================================
patient_id = df["patient_nbr"]

# y = biến mục tiêu cần dự đoán
y = df["target_30days"]

# X = các biến đầu vào
X = df.drop(
    columns=["target_30days", "patient_nbr"]
)

print("\n" + "=" * 60)
print("X VÀ Y")
print("=" * 60)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================================================
# 4. CHIA TRAIN / TEST THEO BỆNH NHÂN
# ============================================================

# GroupShuffleSplit giúp đảm bảo:
# Một bệnh nhân chỉ xuất hiện ở TRAIN hoặc TEST, không xuất hiện ở cả hai.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=patient_id))

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

patient_train = patient_id.iloc[train_idx]
patient_test = patient_id.iloc[test_idx]

print("\n" + "=" * 60)
print("TRAIN / TEST")
print("=" * 60)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("\ny_train:")
print(y_train.value_counts())
print("\ny_test:")
print(y_test.value_counts())

# ============================================================
# 5. KIỂM TRA PATIENT LEAKAGE
# ============================================================
common_patients = set(patient_train) & set(patient_test)

print("\n" + "=" * 60)
print("KIỂM TRA PATIENT LEAKAGE")
print("=" * 60)
print("Số bệnh nhân trùng giữa Train và Test:",
      len(common_patients))

if len(common_patients) == 0:
    print("OK - Không có patient leakage.")
else:
    print("WARNING - Có patient leakage!")

# ============================================================
# 6. XÁC ĐỊNH CỘT CATEGORICAL VÀ NUMERICAL
# ============================================================

categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

# Các cột ID này thực chất là biến phân loại
id_as_categorical = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id"
]

# Chuyển 3 cột ID từ numerical sang categorical
for col in id_as_categorical:
    if col in numerical_cols:
        numerical_cols.remove(col)
    if col not in categorical_cols:
        categorical_cols.append(col)

print("\n" + "=" * 60)
print("CÁC LOẠI BIẾN")
print("=" * 60)

print("Số categorical columns:", len(categorical_cols))
print("Số numerical columns:", len(numerical_cols))

print("\nCategorical:")
print(categorical_cols)

print("\nNumerical:")
print(numerical_cols)

# ============================================================
# 7. PREPROCESSING
# ============================================================

# -----------------------------
# Numerical
# -----------------------------
numerical_transformer = Pipeline(
    steps = [
        ("imputer", SimpleImputer(strategy = "median")),
        ("scaler", StandardScaler())
        ]
    )


# -----------------------------
# Categorical
# -----------------------------
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ]
)

# -----------------------------
# Kết hợp preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)


# ============================================================
# 8. MODEL 1 - LOGISTIC REGRESSION
# ============================================================

# Logistic Regression được sử dụng làm baseline model.
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))
    ]
)

print("\n" + "=" * 60)
print("TRAIN LOGISTIC REGRESSION")
print("=" * 60)

logistic_model.fit(X_train, y_train)
print("Logistic Regression training hoàn tất.")


# ============================================================
# 9. MODEL 2 - RANDOM FOREST
# ============================================================

# Random Forest là model nâng cao hơn và có thể cung cấp Feature Importance.
random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1))
    ]
)

print("\n" + "=" * 60)
print("TRAIN RANDOM FOREST")
print("=" * 60)
random_forest_model.fit(X_train, y_train)
print("Random Forest training hoàn tất.")

# ============================================================
# 10. DỰ ĐOÁN
# ============================================================

# -----------------------------
# Logistic Regression
# -----------------------------
y_pred_lr = logistic_model.predict(X_test)
y_prob_lr = logistic_model.predict_proba(X_test)[:, 1]

# -----------------------------
# Random Forest
# -----------------------------

y_prob_rf = random_forest_model.predict_proba(X_test)[:, 1]

# Giảm threshold để tăng khả năng phát hiện bệnh nhân có nguy cơ
threshold = 0.3
y_pred_rf = (y_prob_rf >= threshold).astype(int)

# ============================================================
# 11. HÀM ĐÁNH GIÁ MODEL
# ============================================================

def evaluate_model(model_name, y_true, y_pred, y_prob):
    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    # Trong tất cả bệnh nhân model dự đoán đúng bao nhiêu %
    accuracy = accuracy_score(y_true, y_pred)
    
    # Trong nhưng người model dự đoán sẽ tái nhập viện, có bao nhiêu người thực sự tái nhập viện
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    # Trong tất cả những người thực sự tái nhập viện < 30 ngày, model phát hiện được bao nhiêu người
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # Cho biết model cân bằng giữa precision và recall tốt đến mức nào
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Đánh giá khả năng model phân biệt người có nguy cơ và không có nguy cơ
    roc_auc = roc_auc_score(y_true, y_prob)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0)    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc
    }


# ============================================================
# 12. ĐÁNH GIÁ 2 MODEL
# ============================================================

result_lr = evaluate_model("Logistic Regression", y_test, y_pred_lr, y_prob_lr)
result_rf = evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)


# ============================================================
# 13. SO SÁNH MODEL
# ============================================================

results = pd.DataFrame([result_lr, result_rf])

print("\n" + "=" * 60)
print("SO SÁNH MODEL")
print("=" * 60)

print(results)


# ============================================================
# 14. VẼ BIỂU ĐỒ SO SÁNH MODEL
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC"
]

results_plot = results.set_index("Model")[metrics]

results_plot.plot(kind="bar", figsize=(10, 6))

plt.title("So sánh hiệu quả các mô hình")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()

# ============================================================
# 15. CONFUSION MATRIX - LOGISTIC REGRESSION
# ============================================================

cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_lr, annot=True, fmt="d")

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# ============================================================
# 16. CONFUSION MATRIX - RANDOM FOREST
# ============================================================

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt="d")

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ============================================================
# 17. ROC CURVE
# ============================================================

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

auc_lr = roc_auc_score(y_test, y_prob_lr)
auc_rf = roc_auc_score(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 18. FEATURE IMPORTANCE - RANDOM FOREST
# ============================================================

# Lấy Random Forest sau preprocessing
rf_model = random_forest_model.named_steps["model"]
rf_preprocessor = (random_forest_model.named_steps["preprocessor"])

# Lấy tên các feature sau One-Hot Encoding
feature_names = (rf_preprocessor.get_feature_names_out())
feature_importances = rf_model.feature_importances_

# Tạo DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
})

# Sắp xếp giảm dần
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP 20 FEATURE QUAN TRỌNG")
print("=" * 60)
print(importance_df.head(20))


# ============================================================
# 19. VẼ TOP 20 FEATURE IMPORTANCE
# ============================================================

top_features = importance_df.head(20)
plt.figure(figsize=(10, 8))
sns.barplot(data=top_features, x="Importance", y="Feature")

plt.title("Top 20 Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ============================================================
# 20. LƯU KẾT QUẢ
# ============================================================

results.to_csv("figures/model_comparison.csv", index=False)

importance_df.to_csv("figures/random_forest_feature_importance.csv", index=False)

print("\n" + "=" * 60)
print("HOÀN TẤT")
print("=" * 60)

print("Đã hoàn thành:")
print("1. Chia Train/Test theo bệnh nhân")
print("2. Preprocessing")
print("3. Logistic Regression")
print("4. Random Forest")
print("5. Đánh giá model")
print("6. Confusion Matrix")
print("7. ROC Curve")
print("8. Feature Importance")