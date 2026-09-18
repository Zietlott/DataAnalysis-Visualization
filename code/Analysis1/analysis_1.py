from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Khắc phục lỗi font hiển thị trên terminal Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 1. ĐỌC DỮ LIỆU
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "diabetic_cleaned_data.csv"

# Nếu tên file thực tế khác thì thử tên dự phòng
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "data" / "processed" / "diabetes_cleaned_data.csv"

FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Đang đọc dữ liệu từ: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Nạp dữ liệu thành công: {df.shape[0]:,} dòng, {df.shape[1]} cột.\n")

# 2. BIẾN MỤC TIÊU VÀ KIỂM TRA
if "target_30days" not in df.columns and "readmitted" in df.columns:
    df["target_30days"] = (df["readmitted"] == "<30").astype(int)

required_columns = ["target_30days", "time_in_hospital", "gender", "age"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Không tìm thấy các cột cần thiết: {missing_columns}")

# 3. THỐNG KÊ MÔ TẢ CƠ BẢN
total_cases = len(df)
readmit_cases = int(df["target_30days"].sum())
non_readmit_cases = total_cases - readmit_cases

pct_readmit = readmit_cases / total_cases * 100
pct_non_readmit = non_readmit_cases / total_cases * 100

gender_counts = df["gender"].value_counts()
largest_gender = gender_counts.idxmax()
largest_gender_pct = (gender_counts.max() / total_cases) * 100

stay_mean = df["time_in_hospital"].mean()
stay_median = df["time_in_hospital"].median()
stay_mode = df["time_in_hospital"].mode()[0]
stay_mode_count = (df["time_in_hospital"] == stay_mode).sum()

print("THỰC TRẠNG TÁI NHẬP VIỆN TRONG 30 NGÀY")
print(f"Tổng số ca quan sát: {total_cases:,} ca")
print(f"- Tái nhập viện < 30 ngày: {readmit_cases:,} ca ({pct_readmit:.2f}%)")
print(f"- Không tái nhập viện <= 30 ngày: {non_readmit_cases:,} ca ({pct_non_readmit:.2f}%)")

print("\nPHÂN BỐ THEO GIỚI TÍNH:")
for gender, count in gender_counts.items():
    percentage = count / total_cases * 100
    print(f"- {gender}: {count:,} ca ({percentage:.2f}%)")

print("\nTHỜI GIAN NẰM VIỆN:")
print(f"- Trung bình: {stay_mean:.2f} ngày")
print(f"- Trung vị: {stay_median:.0f} ngày")

# 4. CẤU HÌNH BIỂU ĐỒ
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# CHART 1 - DONUT CHART (TỶ LỆ TÁI NHẬP VIỆN TRONG 30 NGÀY)
plt.figure(figsize=(6.5, 6))
chart1_labels = [
    f"Không tái nhập viện\n({non_readmit_cases:,} ca)",
    f"Tái nhập viện\n({readmit_cases:,} ca)"
]
sizes = [non_readmit_cases, readmit_cases]

wedges, texts, autotexts = plt.pie(
    sizes,
    labels=chart1_labels,
    autopct="%1.2f%%",
    startangle=90,
    colors=["#2b5c8f", "#d95f02"],
    pctdistance=0.75,
    wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    textprops=dict(fontweight="bold", fontsize=10)
)

autotexts[0].set_color("white")
autotexts[1].set_color("white")

plt.title("1. Tỷ lệ tái nhập viện trong 30 ngày", fontweight="bold", fontsize=13)
plt.text(0, 0, f"Tổng ca\n{total_cases:,}", ha="center", va="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "chart1_readmission_donut.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nINSIGHT BIỂU ĐỒ 1")
print(f"Tỷ lệ tái nhập viện trong 30 ngày là {pct_readmit:.2f}%, trong khi {pct_non_readmit:.2f}% bệnh nhân không tái nhập viện.")
print(f"Điều này cho thấy cứ 100 ca quan sát có khoảng {pct_readmit:.0f} ca tái nhập viện trong vòng 30 ngày.")

# CHART 2 - HISTOGRAM (PHÂN BỐ THỜI GIAN NẰM VIỆN)
plt.figure(figsize=(8.5, 5))
plt.hist(
    df["time_in_hospital"],
    bins=range(int(df["time_in_hospital"].min()), int(df["time_in_hospital"].max()) + 2),
    edgecolor="black",
    color="#2b5c8f",
    alpha=0.85
)

plt.title("2. Phân bố thời gian nằm viện của bệnh nhân", fontweight="bold", fontsize=13)
plt.xlabel("Thời gian nằm viện (ngày)", fontsize=11)
plt.ylabel("Số lượng bệnh nhân (ca)", fontsize=11)
plt.xticks(range(int(df["time_in_hospital"].min()), int(df["time_in_hospital"].max()) + 1))
plt.tight_layout()
plt.savefig(FIGURES_DIR / "chart2_time_in_hospital_hist.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nINSIGHT BIỂU ĐỒ 2")
print(f"Thời gian nằm viện trung bình là {stay_mean:.2f} ngày và trung vị là {stay_median:.0f} ngày.")
print(f"Thời gian nằm viện xuất hiện nhiều nhất là {stay_mode} ngày, với {stay_mode_count:,} bệnh nhân.")

# CHART 3 - LOLLIPOP CHART (TỶ LỆ TÁI NHẬP VIỆN THEO NHÓM TUỔI)
plt.figure(figsize=(10, 5.5))

age_order = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", 
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
]
existing_ages = [age for age in age_order if age in df["age"].values]
age_readmit_rate = (df.groupby("age")["target_30days"].mean().reindex(existing_ages) * 100).fillna(0)

# Lấy giá trị x và y
x_labels = age_readmit_rate.index.astype(str)
y_values = age_readmit_rate.values

# 1. Vẽ thân kẹo (đường thẳng đứng)
plt.vlines(
    x=x_labels, 
    ymin=0, 
    ymax=y_values, 
    color="#2b5c8f", 
    linewidth=3.5, 
    alpha=0.8
)

# 2. Vẽ đỉnh kẹo (chấm tròn)
plt.plot(
    x_labels, 
    y_values, 
    "o", 
    markersize=12, 
    color="#d95f02"
)

plt.title("3. Tỷ lệ tái nhập viện trong 30 ngày theo nhóm tuổi (Lollipop Chart)", fontsize=13, fontweight="bold")
plt.xlabel("Nhóm tuổi", fontsize=11)
plt.ylabel("Tỷ lệ tái nhập viện (%)", fontsize=11)
plt.xticks(rotation=45, ha="right")

# Ghi chú % lên đầu mỗi kẹo mút
for i, yval in enumerate(y_values):
    plt.text(
        i, 
        yval + 0.5, 
        f"{yval:.1f}%", 
        ha="center", 
        va="bottom", 
        fontsize=9, 
        fontweight="bold"
    )

plt.ylim(0, age_readmit_rate.max() + 3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "chart3_readmission_by_age_lollipop.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nINSIGHT BIỂU ĐỒ 3")
highest_age_group = age_readmit_rate.idxmax()
highest_age_rate = age_readmit_rate.max()
print(f"Nhóm tuổi có tỷ lệ tái nhập viện cao nhất là {highest_age_group} với tỷ lệ lên tới {highest_age_rate:.1f}%.")
print("Dữ liệu (Lollipop Chart) cho thấy độ tuổi càng cao có xu hướng ảnh hưởng trực tiếp đến khả năng tái nhập viện.")

# CHART 4 - BAR CHART (TỶ LỆ TÁI NHẬP VIỆN THEO SỐ LẦN NHẬP VIỆN TRƯỚC ĐÓ)
plt.figure(figsize=(8.5, 5.5))
if "number_inpatient" in df.columns:
    df["inpatient_group"] = df["number_inpatient"].apply(lambda x: ">= 4" if x >= 4 else str(x))
    inp_order = ["0", "1", "2", "3", ">= 4"]
    existing_inp = [i for i in inp_order if i in df["inpatient_group"].values]
    
    inp_readmit_rate = (df.groupby("inpatient_group")["target_30days"].mean().reindex(existing_inp) * 100).fillna(0)

    bars_inp = plt.bar(
        inp_readmit_rate.index,
        inp_readmit_rate.values,
        edgecolor="black",
        color="#d95f02"
    )

    plt.title("4. Tỷ lệ tái nhập viện theo số lần nhập viện trước đó", fontsize=13, fontweight="bold")
    plt.xlabel("Số lần nhập viện trước đó", fontsize=11)
    plt.ylabel("Tỷ lệ tái nhập viện (%)", fontsize=11)

    for bar in bars_inp:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.ylim(0, inp_readmit_rate.max() + 5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "chart4_readmission_by_inpatient.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nINSIGHT BIỂU ĐỒ 4")
    highest_inp_group = inp_readmit_rate.idxmax()
    highest_inp_rate = inp_readmit_rate.max()
    print(f"Bệnh nhân từng nhập viện {highest_inp_group} lần trở lên có nguy cơ tái nhập viện cao nhất ({highest_inp_rate:.1f}%).")
    print("Có thể thấy rõ xu hướng tỷ lệ thuận: số lần nhập viện tuyến trước càng nhiều, rủi ro tái nhập viện sớm càng tăng mạnh.")

# 5. TỔNG KẾT INSIGHT - TV2
print("\nTỔNG KẾT PHÂN TÍCH THỰC TRẠNG - TV2")
print(f"- Tỷ lệ tái nhập viện chung (< 30 ngày): {pct_readmit:.2f}%.")
print(f"- Thời gian nằm viện: Trung bình {stay_mean:.2f} ngày; Phổ biến nhất là {stay_mode} ngày.")
print(f"- Nhóm giới tính chiếm đa số: {largest_gender} ({largest_gender_pct:.2f}%).")
print(f"- Độ tuổi rủi ro cao nhất: Nhóm {highest_age_group} có tỷ lệ tái nhập viện lên tới {highest_age_rate:.1f}%.")
if "number_inpatient" in df.columns:
    print(f"- Tiền sử nhập viện: Bệnh nhân từng nhập viện {highest_inp_group} lần trở lên có tỷ lệ tái phát cực cao ({highest_inp_rate:.1f}%).")

