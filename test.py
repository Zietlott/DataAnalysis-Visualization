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
# 6. GIỮ MỘT LẦN ĐIỀU TRỊ ĐẠI DIỆN CHO MỖI BỆNH NHÂN
# ==============================================================================

df = df.sort_values(
    'encounter_id'
).drop_duplicates(
    subset='patient_nbr',
    keep='first'
)


# ==============================================================================
# 7. LOẠI CÁC TRƯỜNG HỢP KHÔNG PHÙ HỢP
# ==============================================================================

expired_ids = [
    11, 13, 14,
    18, 19, 20, 21
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

    # Không có / rất ít sự biến thiên
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


# ==============================================================================
# 10. KIỂM TRA OUTLIER
# ==============================================================================

numerical_cols = df.select_dtypes(
    include=[
        'int64',
        'float64'
    ]
).columns

print(
    "\n--- THỐNG KÊ BIẾN SỐ ---"
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