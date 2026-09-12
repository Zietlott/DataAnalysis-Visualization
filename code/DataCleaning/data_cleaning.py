import pandas as pd
import numpy as np

# 1. ĐỌC DỮ LIỆU
file_path = "data/raw/diabetic_data.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip() # loại bỏ khoảng trắng ở đầu/cuối tên cột
df = df.replace('?', np.nan)
print(
    f"\n[1] Kích thước dữ liệu ban đầu: {df.shape[0]} dòng, {df.shape[1]} cột"
)

# 2. KIỂM TRA THÔNG TIN DỮ LIỆU
print("\n--- THÔNG TIN DỮ LIỆU ---")
df.info()

# 3. KIỂM TRA MISSING VALUES
missing_report = pd.DataFrame({
    'Missing_Count': df.isnull().sum(),
    'Percentage (%)': (
        df.isnull().mean() * 100
    ).round(2)
})

missing_report = missing_report[
    missing_report['Missing_Count'] > 0
].sort_values('Percentage (%)',ascending=False)
print("\n--- BÁO CÁO MISSING VALUES ---")
print(missing_report)

# 4. KIỂM TRA DUPLICATE
duplicate_count = df.duplicated().sum()
print(
    f"\nSố dòng trùng lặp hoàn toàn: "
    f"{duplicate_count}"
)
df = df.drop_duplicates()

# 5. KIỂM TRA SỐ LƯỢNG LẦN KHÁM CỦA MỖI BỆNH NHÂN
# Giữ lại tất cả các lần khám của bệnh nhân.
# Mỗi encounter được xem là một quan sát trong dữ liệu.
print(
    f"\nSố bệnh nhân duy nhất (patient_nbr): "
    f"{df['patient_nbr'].nunique()}"
)
print(
    f"Tổng số lượt khám (encounter) đang giữ: "
    f"{df.shape[0]}"
)

# 6. LOẠI CÁC TRƯỜNG HỢP KHÔNG PHÙ HỢP
# Nhóm cần loại (bệnh nhân đã mất / chuyển hospice) -> không có ý nghĩa để dự đoán tái nhập viện:
#   11 = Expired
#   13 = Hospice / home
#   14 = Hospice / medical facility
#   19 = Expired at home (Medicaid, hospice)
#   20 = Expired in a medical facility (Medicaid, hospice)
#   21 = Expired, place unknown (Medicaid, hospice)
expired_ids = [11, 13, 14, 19, 20, 21]
df = df[~df['discharge_disposition_id'].isin(expired_ids)]

# 7. LOẠI CÁC CỘT KHÔNG CẦN THIẾT
cols_to_drop = [
    # ID
    'encounter_id',

    # Missing nhiều
    'payer_code',
    'medical_specialty',

    # Missing rất cao
    'weight',
    'max_glu_serum',
    'A1Cresult',

    # Không có / rất ít sự biến thiên
    'examide',
    'citoglipton',
    'acetohexamide',
    'troglitazone',
    'metformin-rosiglitazone',
    'metformin-pioglitazone',
    'glimepiride-pioglitazone'
]

df = df.drop(columns = [c for c in cols_to_drop if c in df.columns])

# 8. XỬ LÝ MISSING VALUES CÒN LẠI
categorical_missing = ['race', 'diag_1', 'diag_2', 'diag_3', 'gender']

# Chuẩn hóa giá trị gender không hợp lệ thành missing
df['gender'] = df['gender'].replace('Unknown/Invalid', np.nan)

# Thay missing bằng Unknown
for col in categorical_missing:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# 9. KIỂM TRA OUTLIER
id_like_cols = [
    'patient_nbr',
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id'
]

numerical_cols = df.select_dtypes(include = ['int64','float64']).columns
numerical_cols = [c for c in numerical_cols if c not in id_like_cols]

print("\n--- THỐNG KÊ BIẾN SỐ (đã loại các cột mã ID phân loại) ---")
print(df[numerical_cols].describe())

# ------------------------------------------------------------------------------
# Phát hiện outlier bằng phương pháp IQR (Interquartile Range)
# Quy tắc: 1 giá trị được coi là outlier nếu nằm ngoài khoảng
# [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
# ------------------------------------------------------------------------------

outlier_report = []
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    n_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]

    outlier_report.append({
        'Cột': col,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Cận dưới': lower_bound,
        'Cận trên': upper_bound,
        'Số outlier': n_outliers,
        'Tỷ lệ (%)': round(n_outliers / df.shape[0] * 100, 2)
    })

outlier_report_df = pd.DataFrame(outlier_report)
print("\n--- BÁO CÁO OUTLIER THEO PHƯƠNG PHÁP IQR ---")
print(outlier_report_df)

# Lưu ý: với bộ dữ liệu y tế này, các outlier (VD: thời gian nằm viện dài,
# số lần xét nghiệm nhiều) thường phản ánh tình trạng bệnh nặng thực tế
# chứ không phải lỗi nhập liệu, nên KHÔNG tự động xoá — chỉ ghi nhận để
# tham khảo khi phân tích/mô hình hoá ở các bước sau.

# 10. TẠO BIẾN MỤC TIÊU
df['target_30days'] = (df['readmitted'] == '<30').astype(int)
df = df.drop(columns=['readmitted'])

# 11. KIỂM TRA DỮ LIỆU SAU XỬ LÝ
print(f"\nTổng số ô NaN còn lại: {df.isnull().sum().sum()}"
)

print("\nTỷ lệ target_30days:")
print(df['target_30days'].value_counts(normalize=True))

# 12. XUẤT FILE DỮ LIỆU SẠCH
output_filename = "data/processed/diabetic_cleaned_data.csv"
df.to_csv(output_filename, index=False)

print("\nXỬ LÝ THÀNH CÔNG!")
print(f"File dữ liệu sạch: {output_filename}")
print(f"Tổng số dòng: {df.shape[0]}")
print(f"Tổng số cột: {df.shape[1]}")
print(f"Số biến đặc trưng: {df.shape[1] - 1}")