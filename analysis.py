# ==============================================================================
# BÀI TIỂU LUẬN: BÀI TOÁN GIẢM TỶ LỆ TÁI NHẬP VIỆN BỆNH NHÂN TIỂU ĐƯỜNG
# Bước 1: Thu thập, Tiền xử lý và Làm sạch Dữ liệu (Dành cho Người 1)
# ==============================================================================

# 1. KHAI BÁO THƯ VIỆN & CẤU HÌNH BIỂU ĐỒ DÙNG CHUNG CHO NHÓM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình Format biểu đồ thống nhất cho cả nhóm
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
CUSTOM_PALETTE = ["#2b5c8f", "#d95f02", "#7570b3"]
sns.set_palette(CUSTOM_PALETTE)

print("--- Thư viện và Cấu hình Biểu đồ đã sẵn sàng ---")

# ==============================================================================
# 2. ĐỌC DỮ LIỆU & LÀM SẠCH BAN ĐẦU
# ==============================================================================
file_path = "diabetic_data.csv"
df = pd.read_csv(file_path)

# Chuẩn hóa khoảng trắng ở tên cột và thay ký tự '?' thành NaN
df.columns = df.columns.str.strip()
df = df.replace('?', np.nan)

print(f"\n[1] Kích thước dữ liệu ban đầu: {df.shape[0]} dòng, {df.shape[1]} cột")

# BÁO CÁO DỮ LIỆU KHUYẾT THIẾU (MISSING VALUES)
missing_report = pd.DataFrame({
    'Missing_Count': df.isnull().sum(),
    'Percentage (%)': (df.isnull().mean() * 100).round(2)
})
missing_report = missing_report[missing_report['Missing_Count'] > 0].sort_values(
    'Percentage (%)', 
    ascending=False
)

print("\n--- BÁO CÁO DỮ LIỆU KHUYẾT THIẾU (MISSING VALUES) ---")
print(missing_report)

# ==============================================================================
# 3. LỌC TRÙNG LẶP VÀ XỬ LÝ DỮ LIỆU NHIỄU/KHUYẾT THIẾU
# ==============================================================================
# a. Giữ lại đợt nhập viện đầu tiên của từng bệnh nhân (lọc trùng patient_nbr)
df = df.sort_values('encounter_id').drop_duplicates(subset='patient_nbr', keep='first')

# b. Loại bỏ các bệnh nhân tử vong hoặc chuyển sang chăm sóc xoa dịu (Hospice)
expired_ids = [11, 13, 14, 19, 20, 21]
df = df[~df['discharge_disposition_id'].isin(expired_ids)]

# c. Bỏ các cột thiếu quá nhiều dữ liệu (>30%) hoặc các ID định danh không dùng huấn luyện
cols_to_drop = ['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# d. Điền dữ liệu khuyết thiếu cho các cột phân loại
df['race'] = df['race'].fillna('Unknown')
df['diag_1'] = df['diag_1'].fillna('Unknown')
df['diag_2'] = df['diag_2'].fillna('Unknown')
df['diag_3'] = df['diag_3'].fillna('Unknown')
df['max_glu_serum'] = df['max_glu_serum'].fillna('None')
df['A1Cresult'] = df['A1Cresult'].fillna('None')

print(f"\n[2] Kích thước sau khi lọc trùng và loại bỏ nhiễu: {df.shape[0]} dòng, {df.shape[1]} cột")

# ==============================================================================
# 4. TẠO BIẾN MỤC TIÊU (TARGET VARIABLE)
# ==============================================================================
# 1: Tái nhập viện < 30 ngày, 0: Không tái nhập viện hoặc > 30 ngày
df['target_30days'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
df = df.drop(columns=['readmitted'])

# ==============================================================================
# 5. XUẤT FILE DỮ LIỆU SẠCH HOÀN CHỈNH
# ==============================================================================
output_filename = "diabetes_cleaned_data.csv"
df.to_csv(output_filename, index=False)

print(f"\nXỬ LÝ THÀNH CÔNG! File dữ liệu sạch đã được lưu tại: {output_filename}")
print(f"Tổng số dòng: {df.shape[0]}, Tổng số cột đặc trưng: {df.shape[1]}")
print("Tỷ lệ biến mục tiêu (target_30days):")
print(df['target_30days'].value_counts(normalize=True))