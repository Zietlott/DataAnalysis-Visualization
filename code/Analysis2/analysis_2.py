import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Đọc dữ liệu
df = pd.read_csv("diabetes_cleaned_data.csv")


# 1. TIME_IN_HOSPITAL ↔ NUM_MEDICATIONS

x = df["time_in_hospital"]
y = df["num_medications"]

# Tính hệ số tương quan Pearson
correlation = x.corr(y)

# Vẽ Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.15, s=15)

# Đường xu hướng
coef = np.polyfit(x, y, 1)
line = np.poly1d(coef)
x_line = np.linspace(x.min(), x.max(), 100)

plt.plot(x_line, line(x_line), linewidth=2)

plt.title("Scatter Plot: time_in_hospital vs num_medications")
plt.xlabel("time_in_hospital (days)")
plt.ylabel("num_medications")

# Hiển thị correlation
plt.text(
    0.98, 0.05,
    f"Pearson correlation = {correlation:.3f}",
    transform=plt.gca().transAxes,
    ha="right"
)

plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

print("Correlation:", round(correlation, 3))
# Insight
print("\n--- Insight: time_in_hospital ↔ num_medications ---")

print(
    f"Giữa time_in_hospital và num_medications có hệ số tương quan Pearson "
    f"là {correlation:.3f}, cho thấy mức độ tương quan "
    f"{'dương' if correlation > 0 else 'âm'}."
)

print(
    f"Khi time_in_hospital tăng, num_medications có xu hướng "
    f"{'tăng' if correlation > 0 else 'giảm'} theo."
)


# 2. target_30days ↔ number_inpatient / emergency / outpatient


variables = [
    "number_inpatient",
    "number_emergency",
    "number_outpatient"
]

for var in variables:

    # Gom nhóm từ 7 lần trở lên thành 7+
    df["group"] = df[var].apply(
        lambda x: "7+" if x >= 7 else str(int(x))
    )

    # Tính tỷ lệ tái nhập viện
    rate = df.groupby("group")["target_30days"].mean() * 100

    # Tính số lượng bệnh nhân (Count) của từng nhóm
    count = df.groupby("group").size()

    # Sắp xếp nhóm
    order = ["0", "1", "2", "3", "4", "5", "6", "7+"]

    rate = rate.reindex(
        [x for x in order if x in rate.index]
    )

    count = count.reindex(
        [x for x in order if x in count.index]
    )

    # Vẽ biểu đồ cột
    plt.figure(figsize=(9, 5))
    bars = plt.bar(rate.index, rate.values)

    plt.title(f"Tỷ lệ tái nhập viện trong 30 ngày theo {var}")
    plt.xlabel(var)
    plt.ylabel("Tỷ lệ tái nhập viện trong 30 ngày  (%)")
    plt.ylim(0, rate.max() + 5)

    # Hiển thị tỷ lệ + số lượng bệnh nhân trên đầu mỗi cột
    for bar, value, n in zip(bars, rate.values, count.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.3,
            f"{value:.1f}%\n(n={n})",
            ha="center"
        )

    plt.tight_layout()
    plt.show()

    # Insight
    print(f"\n--- Insight: {var} ---")

    # Nhóm 0 lần
    rate_0 = df[df[var] == 0]["target_30days"].mean() * 100
    count_0 = (df[var] == 0).sum()

    # Nhóm >=2 lần
    rate_2plus = df[df[var] >= 2]["target_30days"].mean() * 100
    count_2plus = (df[var] >= 2).sum()

    # Chênh lệch tỷ lệ
    difference = rate_2plus - rate_0

    print(
        f"Nhóm >=2 lần có {count_2plus} bệnh nhân, "
        f"tỷ lệ tái nhập viện {rate_2plus:.1f}%, "
        f"trong khi nhóm 0 lần có {count_0} bệnh nhân, "
        f"tỷ lệ là {rate_0:.1f}%."
    )

    print(
        f"Chênh lệch tỷ lệ giữa nhóm >=2 lần và nhóm 0 lần "
        f"là {difference:+.1f} điểm phần trăm."
    )

    # Nhóm có tỷ lệ cao nhất
    max_group = rate.idxmax()
    max_rate = rate.max()
    max_count = count.loc[max_group]

    print(
        f"Nhóm {max_group} có tỷ lệ tái nhập viện cao nhất "
        f"({max_rate:.1f}%), với {max_count} bệnh nhân."
    )

# Xóa cột group tạm thời
df.drop(columns=["group"], inplace=True)


# 3. AGE ↔ TIME_IN_HOSPITAL


# Tính median theo nhóm tuổi
median_age = df.groupby("age")["time_in_hospital"].median()

# Vẽ Box Plot
df.boxplot(
    column="time_in_hospital",
    by="age",
    grid=False
)

plt.title("time_in_hospital Across age Groups")
plt.suptitle("")
plt.xlabel("age")
plt.ylabel("time_in_hospital (days)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Insight cho Boxplot

print("\n--- Insight: age ↔ time_in_hospital ---")

max_age = median_age.idxmax()
min_age = median_age.idxmin()

max_median = median_age.max()
min_median = median_age.min()

print(
    f"Nhóm tuổi {max_age} có thời gian nằm viện trung vị cao nhất "
    f"({max_median:.1f} ngày), trong khi nhóm tuổi {min_age} "
    f"có median thấp nhất ({min_median:.1f} ngày)."
)

print(
    f"Chênh lệch median giữa hai nhóm là "
    f"{max_median - min_median:.1f} ngày."
)


# 4. KẾT LUẬN TỔNG HỢP


print("\n" + "=" * 60)
print("KẾT LUẬN TỔNG HỢP")
print("=" * 60)

print(
    "1. Thời gian nằm viện dài hơn có xu hướng đi kèm với số lượng "
    "thuốc sử dụng nhiều hơn."
)

print(
    "2. Tiền sử sử dụng dịch vụ y tế có mối liên hệ với tỷ lệ "
    "tái nhập viện trong 30 ngày."
)

print(
    "3. Độ tuổi cao có mối liên hệ với sự khác biệt về thời gian "
    "điều trị giữa các nhóm tuổi."
)