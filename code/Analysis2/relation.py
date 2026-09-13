import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("diabetes_cleaned_data.csv")

# 1. TIME_IN_HOSPITAL ↔ NUM_MEDICATIONS

x = df["time_in_hospital"]
y = df["num_medications"]

# Tính hệ số tương quan Pearson
correlation = x.corr(y)

# Vẽ Scatter Plot
plt.figure(figsize=(10, 6))

plt.scatter(
    x,
    y,
    alpha=0.15,
    s=15
)

# Tạo đường xu hướng
coef = np.polyfit(x, y, 1)
line = np.poly1d(coef)

x_line = np.linspace(x.min(), x.max(), 100)

plt.plot(
    x_line,
    line(x_line),
    linewidth=2
)

# Tiêu đề và tên trục
plt.title("Scatter Plot: Time in Hospital vs Number of Medications")
plt.xlabel("Time in Hospital (days)")
plt.ylabel("Number of Medications")

# Hiển thị correlation
plt.text(
    0.98,
    0.05,
    f"Pearson correlation = {correlation:.3f}",
    transform=plt.gca().transAxes,
    ha="right",
    fontsize=11
)

plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

print("Correlation: ", correlation)


# 2. CÁC BIẾN ↔ TARGET_30DAYS

# Biến mục tiêu
target = "target_30days"

# Các biến số muốn phân tích
features = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]

# Tính correlation với target
correlations = (
    df[features + [target]]
    .corr()[target]
    .drop(target)
)

# Sắp xếp từ cao xuống thấp
correlations = correlations.sort_values(ascending=False)

# Vẽ Bar Plot
plt.figure(figsize=(10, 6))

plt.bar(
    correlations.index,
    correlations.values
)

plt.title("Correlation of Variables with 30-Day Readmission")
plt.xlabel("Variables")
plt.ylabel("Correlation with target_30days")

plt.xticks(
    rotation=45,
    ha="right"
)

# Đường tại correlation = 0
plt.axhline(
    y=0,
    linewidth=1
)

plt.grid(
    axis="y",
    alpha=0.3
)

# Hiển thị giá trị trên mỗi cột
for i, value in enumerate(correlations.values):

    plt.text(
        i,
        value,
        f"{value:.3f}",
        ha="center",
        va="bottom" if value >= 0 else "top"
    )

plt.tight_layout()
plt.show()

# In kết quả
print("\nCorrelation với target_30days:")
print(correlations)

# 3. AGE ↔ TIME_IN_HOSPITAL
# Vẽ Box Plot

df.boxplot(
    column="time_in_hospital",
    by="age",
    grid=False
)

plt.title("Time in Hospital Across Age Groups")
plt.suptitle("")

plt.xlabel("Age Group")
plt.ylabel("Time in Hospital (days)")

plt.xticks(
    rotation=45
)

plt.tight_layout()
plt.show()
