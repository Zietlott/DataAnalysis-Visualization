# ==============================================================================
# BÀI TIỂU LUẬN: GIẢM TỶ LỆ TÁI NHẬP VIỆN BỆNH NHÂN TIỂU ĐƯỜNG
# Bước 1: Thu thập, Khám phá và Tiền xử lý Dữ liệu
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================================
# 1. CẤU HÌNH BIỂU ĐỒ DÙNG CHUNG
# ==============================================================================

plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

CUSTOM_PALETTE = [
    "#2b5c8f",
    "#d95f02",
    "#7570b3"
]

sns.set_palette(CUSTOM_PALETTE)

print("--- Thư viện và cấu hình biểu đồ đã sẵn sàng ---")


# ==============================================================================
# 2. ĐỌC DỮ LIỆU
# ==============================================================================

file_path = "diabetic_data.csv"

df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

df = df.replace('?', np.nan)

print(
    f"\n[1] Kích thước dữ liệu ban đầu: "
    f"{df.shape[0]} dòng, {df.shape[1]} cột"
)


# ==============================================================================
# 3. KIỂM TRA THÔNG TIN DỮ LIỆU
# ==============================================================================

print("\n--- THÔNG TIN KIỂU DỮ LIỆU ---")

df.info()


# ==============================================================================
# 4. KIỂM TRA MISSING VALUES
# ==============================================================================

missing_report = pd.DataFrame({
    'Missing_Count': df.isnull().sum(),
    'Percentage (%)': (
        df.isnull().mean() * 100
    ).round(2)
})

missing_report = missing_report[
    missing_report['Missing_Count'] > 0
].sort_values(
    'Percentage (%)',
    ascending=False
)

print("\n--- BÁO CÁO MISSING VALUES ---")

print(missing_report)


# ==============================================================================
# 5. KIỂM TRA DUPLICATE
# ==============================================================================

duplicate_count = df.duplicated().sum()

print(
    f"\nSố dòng trùng lặp hoàn toàn: "
    f"{duplicate_count}"
)

df = df.drop_duplicates()


# ==============================================================================
# 6. XỬ LÝ BỆNH NHÂN CÓ NHIỀU LẦN KHÁM
# ------------------------------------------------------------------------------
# Giữ nguyên logic: sắp xếp theo encounter_id tăng dần rồi lấy lần khám ĐẦU TIÊN
# của mỗi patient_nbr (đúng theo cách làm gốc của Strack et al., tránh việc các
# lần tái khám sau của cùng 1 bệnh nhân làm sai lệch/phụ thuộc lẫn nhau).
# Lưu ý: bước này CHỈ hoạt động đúng sau khi mục 7 đã được sửa lỗi mã 18
# (xem giải thích bên dưới) — nếu không, một bệnh nhân có lần khám đầu tiên
# bị gắn nhầm là "expired" (do mã 18) sẽ bị loại oan toàn bộ hồ sơ.
# ==============================================================================

df = df.sort_values(
    'encounter_id'
).drop_duplicates(
    subset='patient_nbr',
    keep='first'
)


# ==============================================================================
# 7. LOẠI CÁC TRƯỜNG HỢP KHÔNG PHÙ HỢP
# ------------------------------------------------------------------------------
# LỖI CŨ: expired_ids = [11, 13, 14, 18, 19, 20, 21]
# Mã 18 trong discharge_disposition_id nghĩa là "NULL" (không xác định nơi
# xuất viện) — KHÔNG liên quan đến tử vong/hospice, nên không được gộp vào
# nhóm "expired". Gộp nhầm khiến ta loại oan các bệnh nhân có dữ liệu tái
# nhập viện hoàn toàn hợp lệ.
#
# Nhóm đúng cần loại (bệnh nhân đã mất / chuyển hospice -> không có ý nghĩa
# để dự đoán tái nhập viện):
#   11 = Expired
#   13 = Hospice / home
#   14 = Hospice / medical facility
#   19 = Expired at home (Medicaid, hospice)
#   20 = Expired in a medical facility (Medicaid, hospice)
#   21 = Expired, place unknown (Medicaid, hospice)
# ==============================================================================

expired_ids = [
    11, 13, 14,
    19, 20, 21
]

df = df[
    ~df['discharge_disposition_id'].isin(
        expired_ids
    )
]


# ==============================================================================
# 8. LOẠI CÁC CỘT KHÔNG CẦN THIẾT
# ==============================================================================

cols_to_drop = [

    # ID
    'encounter_id',
    'patient_nbr',

    # Missing nhiều
    'weight',
    'payer_code',
    'medical_specialty',

    # Missing rất cao
    'max_glu_serum',
    'A1Cresult',

    # Không có / rất ít sự biến thiên (gần như hằng số)
    'examide',
    'citoglipton',
    'acetohexamide',
    'troglitazone'
]

df = df.drop(
    columns=[
        c for c in cols_to_drop
        if c in df.columns
    ]
)


# ==============================================================================
# 9. XỬ LÝ MISSING VALUES CÒN LẠI
# ------------------------------------------------------------------------------
# LỖI CŨ: chỉ xử lý race, diag_1, diag_2, diag_3.
# THIẾU: cột 'gender' có giá trị literal "Unknown/Invalid" (không phải NaN)
# nên không bị isnull() bắt được, nhưng vẫn là dữ liệu rác cần loại bỏ
# (thường chỉ vài dòng, không ảnh hưởng đáng kể đến kích thước tập dữ liệu).
# ==============================================================================

categorical_missing = [
    'race',
    'diag_1',
    'diag_2',
    'diag_3'
]

for col in categorical_missing:

    if col in df.columns:

        df[col] = df[col].fillna(
            'Unknown'
        )

# Xử lý riêng cho gender: loại các dòng có giá trị không hợp lệ
if 'gender' in df.columns:

    df = df[
        df['gender'] != 'Unknown/Invalid'
    ]


# ==============================================================================
# 10. KIỂM TRA OUTLIER
# ------------------------------------------------------------------------------
# LỖI CŨ: numerical_cols lấy toàn bộ cột int64/float64, nhưng admission_type_id,
# discharge_disposition_id, admission_source_id thực chất là MÃ PHÂN LOẠI
# (categorical id), không phải biến liên tục. Tính mean/std/describe trên
# chúng là vô nghĩa và dễ khiến hiểu sai là có "outlier".
# ==============================================================================

id_like_cols = [
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id'
]

numerical_cols = df.select_dtypes(
    include=[
        'int64',
        'float64'
    ]
).columns

numerical_cols = [
    c for c in numerical_cols
    if c not in id_like_cols
]

print(
    "\n--- THỐNG KÊ BIẾN SỐ (đã loại các cột mã ID phân loại) ---"
)

print(
    df[numerical_cols].describe()
)


# ==============================================================================
# 11. TẠO BIẾN MỤC TIÊU
# ==============================================================================

df['target_30days'] = (
    df['readmitted'] == '<30'
).astype(int)

df = df.drop(
    columns=['readmitted']
)


# ==============================================================================
# 12. KIỂM TRA DỮ LIỆU SAU XỬ LÝ
# ==============================================================================

print(
    f"\nTổng số ô NaN còn lại: "
    f"{df.isnull().sum().sum()}"
)

print(
    "\nTỷ lệ target_30days:"
)

print(
    df['target_30days']
    .value_counts(normalize=True)
)


# ==============================================================================
# 13. XUẤT FILE DỮ LIỆU SẠCH
# ==============================================================================

output_filename = (
    "diabetes_cleaned_data.csv"
)

df.to_csv(
    output_filename,
    index=False
)

print(
    f"\nXỬ LÝ THÀNH CÔNG!"
)

print(
    f"File dữ liệu sạch: "
    f"{output_filename}"
)

print(
    f"Tổng số dòng: "
    f"{df.shape[0]}"
)

print(
    f"Tổng số cột: "
    f"{df.shape[1]}"
)

print(
    f"Số biến đặc trưng: "
    f"{df.shape[1] - 1}"
)