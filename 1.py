# ==============================================================================
# BÀI TIỂU LUẬN: GIẢM TỶ LỆ TÁI NHẬP VIỆN BỆNH NHÂN TIỂU ĐƯỜNG
# NGƯỜI 1: THU THẬP - KHÁM PHÁ - TIỀN XỬ LÝ - LÀM SẠCH DỮ LIỆU
# ==============================================================================


# ==============================================================================
# 1. KHAI BÁO THƯ VIỆN
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================================
# 2. CẤU HÌNH BIỂU ĐỒ DÙNG CHUNG CHO CẢ NHÓM
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

print("=" * 80)
print("CẤU HÌNH BIỂU ĐỒ ĐÃ SẴN SÀNG")
print("=" * 80)


# ==============================================================================
# 3. ĐỌC DỮ LIỆU
# ==============================================================================

file_path = "diabetic_data.csv"

df = pd.read_csv(file_path)

# Chuẩn hóa tên cột
df.columns = df.columns.str.strip()

# Chuyển ký hiệu '?' thành NaN
df = df.replace('?', np.nan)

print("\n" + "=" * 80)
print("1. THÔNG TIN DỮ LIỆU BAN ĐẦU")
print("=" * 80)

print(f"Số dòng : {df.shape[0]:,}")
print(f"Số cột  : {df.shape[1]:,}")


# ==============================================================================
# 4. KIỂM TRA KIỂU DỮ LIỆU
# ==============================================================================

print("\n" + "=" * 80)
print("2. KIỂM TRA KIỂU DỮ LIỆU")
print("=" * 80)

df.info()


# ==============================================================================
# 5. KIỂM TRA DUPLICATE THỰC SỰ
# ==============================================================================

print("\n" + "=" * 80)
print("3. KIỂM TRA DỮ LIỆU TRÙNG LẶP")
print("=" * 80)

duplicate_count = df.duplicated().sum()

print(f"Số dòng trùng lặp hoàn toàn: {duplicate_count:,}")

if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"Đã loại bỏ {duplicate_count:,} dòng trùng lặp.")
else:
    print("Không phát hiện dòng trùng lặp hoàn toàn.")


# ==============================================================================
# 6. KIỂM TRA MISSING VALUES BAN ĐẦU
# ==============================================================================

print("\n" + "=" * 80)
print("4. KIỂM TRA DỮ LIỆU KHUYẾT THIẾU")
print("=" * 80)

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

if missing_report.empty:
    print("Không có dữ liệu khuyết thiếu.")
else:
    print(missing_report)


# ==============================================================================
# 7. LOẠI CÁC TRƯỜNG HỢP KHÔNG PHÙ HỢP
# ==============================================================================

print("\n" + "=" * 80)
print("5. XỬ LÝ CÁC TRƯỜNG HỢP KHÔNG PHÙ HỢP")
print("=" * 80)

# Các mã discharge disposition liên quan đến tử vong
# hoặc chuyển sang chăm sóc giảm nhẹ/hospice
expired_ids = [
    11, 13, 14,
    18, 19, 20, 21
]

before_expired = len(df)

df = df[
    ~df['discharge_disposition_id'].isin(expired_ids)
].copy()

removed_expired = before_expired - len(df)

print(
    f"Số encounter bị loại: "
    f"{removed_expired:,}"
)

print(
    f"Số dòng còn lại: "
    f"{len(df):,}"
)


# ==============================================================================
# 8. LOẠI CÁC CỘT KHÔNG CẦN THIẾT
# ==============================================================================

print("\n" + "=" * 80)
print("6. LOẠI CÁC CỘT KHÔNG CẦN THIẾT")
print("=" * 80)

cols_to_drop = [

    # --------------------------------------------------------------------------
    # ID - không mang ý nghĩa dự đoán
    # --------------------------------------------------------------------------
    'encounter_id',
    'patient_nbr',

    # --------------------------------------------------------------------------
    # Các cột có tỷ lệ missing rất cao
    # --------------------------------------------------------------------------
    'weight',
    'payer_code',
    'medical_specialty',

    # --------------------------------------------------------------------------
    # Các xét nghiệm có tỷ lệ missing rất cao
    # --------------------------------------------------------------------------
    'max_glu_serum',
    'A1Cresult',

    # --------------------------------------------------------------------------
    # Các thuốc không có hoặc gần như không có sự biến thiên
    # --------------------------------------------------------------------------
    'examide',
    'citoglipton',
    'acetohexamide',
    'troglitazone'
]

existing_cols_to_drop = [
    col for col in cols_to_drop
    if col in df.columns
]

df = df.drop(
    columns=existing_cols_to_drop
)

print("Các cột đã loại bỏ:")

for col in existing_cols_to_drop:
    print(f" - {col}")


# ==============================================================================
# 9. XỬ LÝ MISSING VALUES CÒN LẠI
# ==============================================================================

print("\n" + "=" * 80)
print("7. XỬ LÝ MISSING VALUES CÒN LẠI")
print("=" * 80)

categorical_missing = [
    'race',
    'diag_1',
    'diag_2',
    'diag_3'
]

for col in categorical_missing:

    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

print("Đã thay thế missing values của các biến phân loại bằng 'Unknown'.")


# ==============================================================================
# 10. KIỂM TRA CÁC CỘT KHÔNG CÓ SỰ BIẾN THIÊN
# ==============================================================================

print("\n" + "=" * 80)
print("8. KIỂM TRA CÁC BIẾN KHÔNG CÓ SỰ BIẾN THIÊN")
print("=" * 80)

low_variance_cols = [
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]

if low_variance_cols:
    print("Các cột không có sự biến thiên:")
    for col in low_variance_cols:
        print(f" - {col}")
else:
    print("Không còn cột nào chỉ có một giá trị duy nhất.")


# ==============================================================================
# 11. KIỂM TRA KIỂU DỮ LIỆU SAU KHI LÀM SẠCH
# ==============================================================================

print("\n" + "=" * 80)
print("9. KIỂM TRA KIỂU DỮ LIỆU SAU TIỀN XỬ LÝ")
print("=" * 80)

print(df.dtypes)


# ==============================================================================
# 12. KIỂM TRA CÁC BIẾN SỐ VÀ OUTLIER
# ==============================================================================

print("\n" + "=" * 80)
print("10. KIỂM TRA CÁC BIẾN SỐ")
print("=" * 80)

numerical_cols = df.select_dtypes(
    include=['int64', 'float64']
).columns

print(
    df[numerical_cols].describe().T
)


# ------------------------------------------------------------------------------
# Kiểm tra outlier bằng IQR
# ------------------------------------------------------------------------------

print("\n" + "=" * 80)
print("11. KIỂM TRA OUTLIER BẰNG IQR")
print("=" * 80)

outlier_report = []

for col in numerical_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_count = (
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ).sum()

    outlier_percentage = (
        outlier_count / len(df) * 100
    )

    outlier_report.append({
        'Variable': col,
        'Outlier_Count': outlier_count,
        'Outlier_Percentage (%)': round(
            outlier_percentage, 2
        )
    })

outlier_report = pd.DataFrame(outlier_report)

print(outlier_report.sort_values(
    'Outlier_Percentage (%)',
    ascending=False
))


# ==============================================================================
# 13. TẠO BIẾN MỤC TIÊU
# ==============================================================================

print("\n" + "=" * 80)
print("12. TẠO BIẾN MỤC TIÊU")
print("=" * 80)

# 1 = tái nhập viện trong vòng 30 ngày
# 0 = không tái nhập viện trong 30 ngày

df['target_30days'] = (
    df['readmitted'] == '<30'
).astype(int)

# Không cần giữ biến readmitted gốc nữa
df = df.drop(
    columns=['readmitted']
)

print(
    "Đã tạo biến mục tiêu: target_30days"
)

print("\nQuy ước:")
print("1 = Tái nhập viện trong vòng 30 ngày")
print("0 = Không tái nhập viện trong vòng 30 ngày")


# ==============================================================================
# 14. KIỂM TRA PHÂN BỐ TARGET
# ==============================================================================

print("\n" + "=" * 80)
print("13. PHÂN BỐ BIẾN MỤC TIÊU")
print("=" * 80)

target_count = df['target_30days'].value_counts()

target_percentage = (
    df['target_30days']
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

target_report = pd.DataFrame({
    'Count': target_count,
    'Percentage (%)': target_percentage
})

print(target_report)


# ==============================================================================
# 15. KIỂM TRA MISSING SAU CÙNG
# ==============================================================================

print("\n" + "=" * 80)
print("14. KIỂM TRA MISSING SAU TIỀN XỬ LÝ")
print("=" * 80)

remaining_missing = df.isnull().sum()

remaining_missing = remaining_missing[
    remaining_missing > 0
]

total_nan = df.isnull().sum().sum()

print(
    f"Tổng số ô NaN còn lại: {total_nan:,}"
)

if total_nan == 0:
    print("✓ Dataset không còn dữ liệu khuyết thiếu.")
else:
    print("Các cột vẫn còn missing:")
    print(remaining_missing)


# ==============================================================================
# 16. KIỂM TRA DUPLICATE SAU CÙNG
# ==============================================================================

print("\n" + "=" * 80)
print("15. KIỂM TRA DUPLICATE SAU TIỀN XỬ LÝ")
print("=" * 80)

final_duplicate_count = df.duplicated().sum()

print(
    f"Số dòng trùng lặp hoàn toàn còn lại: "
    f"{final_duplicate_count:,}"
)


# ==============================================================================
# 17. KIỂM TRA KÍCH THƯỚC DATASET CUỐI
# ==============================================================================

print("\n" + "=" * 80)
print("16. KÍCH THƯỚC DATASET SAU TIỀN XỬ LÝ")
print("=" * 80)

print(
    f"Số dòng: {df.shape[0]:,}"
)

print(
    f"Số cột: {df.shape[1]:,}"
)

print(
    f"Số biến đặc trưng: {df.shape[1] - 1:,}"
)

print(
    "Biến mục tiêu: target_30days"
)


# ==============================================================================
# 18. XUẤT DATASET SẠCH
# ==============================================================================

output_filename = "diabetes_cleaned_data.csv"

df.to_csv(
    output_filename,
    index=False
)

print("\n" + "=" * 80)
print("17. XUẤT DATASET")
print("=" * 80)

print(
    f"✓ Dataset sạch đã được lưu thành công:"
)

print(
    f"  {output_filename}"
)


# ==============================================================================
# 19. XUẤT BÁO CÁO MISSING
# ==============================================================================

missing_report.to_csv(
    "missing_value_report.csv",
    index=True
)

print(
    "✓ Đã lưu báo cáo missing:"
)

print(
    "  missing_value_report.csv"
)


# ==============================================================================
# 20. XUẤT BÁO CÁO OUTLIER
# ==============================================================================

outlier_report.to_csv(
    "outlier_report.csv",
    index=False
)

print(
    "✓ Đã lưu báo cáo outlier:"
)

print(
    "  outlier_report.csv"
)


# ==============================================================================
# 21. HOÀN TẤT
# ==============================================================================

print("\n" + "=" * 80)
print("HOÀN TẤT QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU")
print("=" * 80)

print(
    f"Dataset cuối cùng: "
    f"{df.shape[0]:,} dòng × {df.shape[1]:,} cột"
)

print(
    f"Target: target_30days"
)

print(
    "File dùng chung cho nhóm: "
    "diabetes_cleaned_data.csv"
)

print("=" * 80)