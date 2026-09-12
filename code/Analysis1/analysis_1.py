# Bước 2: Phân tích Thực trạng & Trực quan hóa Dữ liệu (Descriptive Analytics)

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Khắc phục lỗi font hiển thị trên Terminal Windows
if sys.platform == "win32":
  sys.stdout.reconfigure(encoding="utf-8")

# 2. Thiết lập đường dẫn động chuẩn dự án
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent

# Tìm file dữ liệu sạch
DATA_PATH = BASE_DIR / "data" / "processed" / "diabetic_cleaned_data.csv"
if not DATA_PATH.exists():
  DATA_PATH = BASE_DIR / "data" / "processed" / "diabetes_cleaned_data.csv"

FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 3. Đọc dữ liệu
print("=" * 65)
print(f"Đang đọc dữ liệu từ: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Nạp dữ liệu thành công: {df.shape[0]:,} dòng, {df.shape[1]} cột.")
print("=" * 65)



# 4. TÍNH TOÁN & IN THÔNG TIN CƠ BẢN THEO YÊU CẦU RA TERMINAL
total_cases = len(df)
readmit_cases = int(df["target_30days"].sum())
non_readmit_cases = total_cases - readmit_cases
pct_readmit = (readmit_cases / total_cases) * 100
pct_non_readmit = (non_readmit_cases / total_cases) * 100

print("\n--- 1. THỰC TRẠNG TÁI NHẬP VIỆN (<= 30 NGÀY) ---")
print(f"Tổng số ca quan sát: {total_cases:,} ca")
print(
    f"- Tái nhập viện <= 30 ngày:  {readmit_cases:,} ca ({pct_readmit:.2f}%)"
)
print(
    f"- Không tái nhập viện:      {non_readmit_cases:,} ca"
    f" ({pct_non_readmit:.2f}%)"
)

print("\n--- 2. PHÂN BỐ BỆNH NHÂN THEO CÁC BIẾN CHÍNH ---")

# a. Giới tính
print("[a] Phân bố Giới tính:")
gender_counts = df["gender"].value_counts()
for g, cnt in gender_counts.items():
  print(f"    + {g:<15}: {cnt:,} ca ({cnt/total_cases*100:.2f}%)")

# b. Thời gian nằm viện
stay_mean = df["time_in_hospital"].mean()
stay_median = df["time_in_hospital"].median()
stay_std = df["time_in_hospital"].std()
print(
    f"[b] Thời gian nằm viện (ngày): Trung bình = {stay_mean:.2f} | Trung vị = {stay_median:.0f} | Độ lệch = {stay_std:.2f}"
)

# c. Số lượng thuốc
med_mean = df["num_medications"].mean()
med_median = df["num_medications"].median()
print(
    f"[c] Số lượng thuốc kê đơn:   Trung bình = {med_mean:.2f} | Trung vị = {med_median:.0f} loại"
)

# d. Số lần nhập viện trước đó
if "number_inpatient" in df.columns:
  inp_mean = df["number_inpatient"].mean()
  inp_zero = (df["number_inpatient"] == 0).sum()
  pct_zero = inp_zero / total_cases * 100

  print(
    f"[d] Số lần nhập viện trước: Trung bình = {inp_mean:.2f} lần "
    f"({pct_zero:.1f}% chưa từng nhập viện)"
    )

# e. Độ tuổi
print("[e] Phân bố Nhóm tuổi (Top 3 nhóm cao nhất):")
age_dist = df["age"].value_counts(normalize=True) * 100
for a, pct in age_dist.head(3).items():
  print(f"    + Nhóm {a:<10}: {pct:.2f}%")
print("=" * 65)

# 5. CẤU HÌNH BIỂU ĐỒ
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PRIMARY_COLOR = "#2b5c8f"
HIGHLIGHT_COLOR = "#d95f02"

# HÌNH 1: DONUT CHART (Hiển thị số ca cụ thể từng trường hợp)
plt.figure(figsize=(6.5, 6.5))
sizes = [non_readmit_cases, readmit_cases]
# Nhãn hiển thị cả tên và số ca cụ thể
chart1_labels = [
    f"Không tái nhập viện\n({non_readmit_cases:,} ca)",
    f"Tái nhập viện\n({readmit_cases:,} ca)",
]

wedges, texts, autotexts = plt.pie(
    sizes,
    labels=chart1_labels,
    autopct="%1.2f%%",
    startangle=90,
    colors=[PRIMARY_COLOR, HIGHLIGHT_COLOR],
    pctdistance=0.75,
    wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    textprops=dict(fontweight="bold", fontsize=10),
)
autotexts[0].set_color("white")
autotexts[1].set_color("white")

plt.title("1. Tỷ lệ tái nhập viện trong 30 ngày", fontweight="bold", fontsize=13)
plt.text(
    0,
    0,
    f"Tổng ca\n{total_cases:,}",
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "chart1_readmission_donut.png", dpi=300)
plt.close()

# HÌNH 2: HISTOGRAM (Có cả Trung bình và Trung vị)
plt.figure(figsize=(7.5, 5))
sns.histplot(
    df["time_in_hospital"],
    bins=14,
    discrete=True,
    color=PRIMARY_COLOR,
    edgecolor="black",
    alpha=0.85,
)

plt.axvline(
    stay_median,
    color=HIGHLIGHT_COLOR,
    linestyle="--",
    linewidth=2,
    label=f"Trung vị: {stay_median:.0f} ngày",
)
plt.axvline(
    stay_mean,
    color="crimson",
    linestyle="-.",
    linewidth=2,
    label=f"Trung bình: {stay_mean:.2f} ngày",
)

plt.title(
    "2. Phân bố thời gian nằm viện của bệnh nhân",
    fontweight="bold",
    fontsize=13,
)
plt.xlabel("Số ngày nằm viện (ngày)", fontsize=11)
plt.ylabel("Số lượng bệnh nhân (ca)", fontsize=11)
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "chart2_time_in_hospital_hist.png", dpi=300)
plt.close()

# HÌNH 3: LINE CHART 
plt.figure(figsize=(8, 5))
age_order = [
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]
existing_ages = [a for a in age_order if a in df["age"].values]
age_rate = (
    df.groupby("age")["target_30days"].mean().reindex(existing_ages) * 100
)

plt.plot(
    age_rate.index,
    age_rate.values,
    marker="o",
    color=HIGHLIGHT_COLOR,
    linewidth=2.5,
    markersize=7,
)

# Hiển thị nhãn giá trị và tự động căn trục y tránh bị cắt đỉnh
max_y_val = age_rate.max()
plt.ylim(0, max_y_val + 2.5)

for i, val in enumerate(age_rate.values):
  plt.text(
      i, val + 0.35, f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold"
  )

plt.title(
    "3. Tỷ lệ tái nhập viện theo nhóm tuổi", fontweight="bold", fontsize=13
)
plt.xlabel("Nhóm tuổi", fontsize=11)
plt.ylabel("Tỷ lệ tái nhập viện (%)", fontsize=11)
plt.xticks(rotation=35)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "chart3_age_trend_line.png", dpi=300)
plt.close()

print(f"Hoàn thành! Đã xuất đủ 3 biểu đồ vào thư mục: {FIGURES_DIR}")