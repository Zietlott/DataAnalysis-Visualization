from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# 1. CẤU HÌNH TRANG
# =========================================================

st.set_page_config(
    page_title="Dashboard tái nhập viện bệnh nhân tiểu đường",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# 2. ĐƯỜNG DẪN
# =========================================================

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "diabetic_cleaned_data.csv"
)

FIGURES_DIR = BASE_DIR / "figures"

MODEL_RESULTS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_results.csv"
)

FEATURE_IMPORTANCE_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "feature_importance.csv"
)


# =========================================================
# 2.1 BẢNG MÀU DÙNG CHUNG (chỉ phục vụ trực quan hoá,
#     KHÔNG ảnh hưởng đến số liệu/tính toán)
# =========================================================

PALETTE_AGE = px.colors.qualitative.Bold          # 10 màu rực cho 10 nhóm tuổi
PALETTE_GROUP = px.colors.qualitative.Set2         # màu cho nhóm inpatient/emergency/outpatient
PALETTE_GENDER = ["#2563eb", "#f97316", "#9ca3af"]  # Nữ / Nam / Không rõ

COLOR_NONREADMIT = "#3b82f6"   # xanh dương - không tái nhập
COLOR_READMIT = "#dc2626"      # đỏ - tái nhập (nhóm cần chú ý)

KPI_ACCENTS = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed"]
INFO_ACCENTS = {"tv2": "#2563eb", "tv3": "#7c3aed", "tv4": "#f97316"}


def bold(text: str) -> str:
    """Bọc text bằng thẻ <b> - Plotly render được HTML tag này ở
    title/annotation/label mà không ảnh hưởng tới dữ liệu gốc."""
    return f"<b>{text}</b>"


# =========================================================
# 3. CSS
# =========================================================

st.html(
    """
    <style>

    /* =====================================================
       TOÀN TRANG
       ===================================================== */

    .stApp {
        background-color: #f5f7fb;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
        font-weight: 800 !important;
    }

    p {
        color: #374151;
        font-weight: 500;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px 14px;
        margin-bottom: 18px;
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #d1d5db;
    }

    .sidebar-icon {
        font-size: 28px;
        line-height: 1;
    }

    .sidebar-title {
        font-size: 16px;
        font-weight: 800;
        color: #111827 !important;
        letter-spacing: 0.5px;
    }

    .sidebar-subtitle {
        font-size: 12px;
        color: #4b5563 !important;
        margin-top: 3px;
        font-weight: 600;
    }

    .filter-header {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: #ffffff !important;
        padding: 11px 13px;
        border-radius: 10px;
        font-size: 14px;
        font-weight: 800;
        margin-bottom: 17px;
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: #ffffff !important;
        font-weight: 700;
    }

    [data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 700;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] {
        border-radius: 10px;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        border-radius: 10px;
        min-height: 42px;
    }

    [data-testid="stSidebar"] .stSlider {
        padding-top: 4px;
    }


    /* =====================================================
       HERO
       ===================================================== */

    .hero {
        background: linear-gradient(
            135deg,
            #dbeafe,
            #ede9fe
        );
        padding: 30px 34px;
        border-radius: 17px;
        margin-bottom: 26px;
        border: 1px solid #bfdbfe;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
    }

    .hero h1 {
        color: #1e3a8a !important;
        font-size: 31px;
        font-weight: 800;
        margin: 0 0 8px 0;
    }

    .hero p {
        color: #374151 !important;
        font-size: 15px;
        font-weight: 700;
        margin: 0;
    }


    /* =====================================================
       SECTION
       ===================================================== */

    .section-title {
        color: #111827 !important;
        font-size: 25px;
        font-weight: 800;
        margin-top: 18px;
        margin-bottom: 5px;
        border-left: 6px solid #2563eb;
        padding-left: 12px;
    }

    .section-desc {
        color: #4b5563 !important;
        font-size: 15px;
        font-weight: 600;
        line-height: 1.6;
        margin-bottom: 18px;
    }


    /* =====================================================
       KPI
       ===================================================== */

    .kpi-card {
        background: #ffffff;
        border-radius: 15px;
        padding: 20px 21px;
        min-height: 132px;
        border: 1px solid #e5e7eb;
        border-left-width: 6px;
        border-left-style: solid;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        transition: transform 0.15s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
    }

    .kpi-label {
        color: #4b5563 !important;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    .kpi-value {
        color: #111827 !important;
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
        line-height: 1.1;
    }

    .kpi-note {
        color: #6b7280 !important;
        font-size: 13px;
        font-weight: 700;
        margin-top: 7px;
    }


    /* =====================================================
       INSIGHT
       ===================================================== */

    .info-card {
        background: #ffffff;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #e5e7eb;
        border-top-width: 5px;
        border-top-style: solid;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        min-height: 230px;
    }

    .info-card h4 {
        color: #111827 !important;
        margin-top: 3px;
        margin-bottom: 9px;
        font-size: 18px;
        font-weight: 800;
    }

    .info-card p {
        color: #374151 !important;
        font-size: 14px;
        font-weight: 600;
        line-height: 1.7;
        margin-bottom: 10px;
    }

    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: #dbeafe;
        color: #1e40af !important;
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .badge-tv2 { background: #dbeafe; color: #1e40af !important; }
    .badge-tv3 { background: #ede9fe; color: #5b21b6 !important; }
    .badge-tv4 { background: #ffedd5; color: #9a3412 !important; }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {
        text-align: center;
        color: #6b7280 !important;
        font-size: 13px;
        font-weight: 700;
        padding: 28px 0 10px 0;
    }


    /* =====================================================
       CAPTION
       ===================================================== */

    [data-testid="stCaptionContainer"] {
        text-align: center !important;
        color: #111111 !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }

    [data-testid="stCaptionContainer"] p {
        text-align: center !important;
        color: #111111 !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }

    </style>
    """
)


# =========================================================
# 4. ĐỌC DỮ LIỆU
# =========================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)

    if (
        "target_30days" not in df.columns
        and "readmitted" in df.columns
    ):
        df["target_30days"] = (
            df["readmitted"] == "<30"
        ).astype(int)

    return df


df = load_data()

if df is None:
    st.error(
        f"Không tìm thấy file dữ liệu:\n{DATA_PATH}"
    )
    st.stop()


# =========================================================
# 5. SIDEBAR - BỘ LỌC
# =========================================================

st.sidebar.html(
    """
    <div class="sidebar-header">

        <div class="sidebar-icon">
            🩺
        </div>

        <div>
            <div class="sidebar-title">
                DASHBOARD
            </div>

            <div class="sidebar-subtitle">
                Diabetes Readmission
            </div>
        </div>

    </div>
    """
)

st.sidebar.html(
    """
    <div class="filter-header">
        🔎 BỘ LỌC DỮ LIỆU
    </div>
    """
)


# ---------------------------------------------------------
# Filter nhóm tuổi
# ---------------------------------------------------------

st.sidebar.markdown("**👤 Nhóm tuổi**")

if "age" in df.columns:

    age_values = sorted(
        df["age"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    age_values = []


age_options = ["Tất cả"] + age_values

selected_age = st.sidebar.selectbox(
    "age_filter",
    age_options,
    index=0,
    label_visibility="collapsed"
)


# ---------------------------------------------------------
# Filter giới tính
# ---------------------------------------------------------

st.sidebar.markdown("**⚧ Giới tính**")

if "gender" in df.columns:

    gender_values = sorted(
        df["gender"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    gender_values = []


gender_options = ["Tất cả"] + gender_values

selected_gender = st.sidebar.selectbox(
    "gender_filter",
    gender_options,
    index=0,
    label_visibility="collapsed"
)


# ---------------------------------------------------------
# Filter số lần nhập viện trước đó
# ---------------------------------------------------------

st.sidebar.markdown(
    "**🩺 Số lần nhập viện trước đó**"
)

if "number_inpatient" in df.columns:

    inpatient_numeric = pd.to_numeric(
        df["number_inpatient"],
        errors="coerce"
    )

    max_inpatient = int(
        inpatient_numeric.max()
    )

    selected_inpatient = st.sidebar.slider(
        "inpatient_filter",
        min_value=0,
        max_value=max_inpatient,
        value=(0, max_inpatient),
        label_visibility="collapsed"
    )

else:

    max_inpatient = 0
    selected_inpatient = (0, 0)


# =========================================================
# 6. ÁP DỤNG FILTER
# =========================================================

filtered = df.copy()


# Filter tuổi
if (
    "age" in filtered.columns
    and selected_age != "Tất cả"
):
    filtered = filtered[
        filtered["age"].astype(str)
        == str(selected_age)
    ]


# Filter giới tính
if (
    "gender" in filtered.columns
    and selected_gender != "Tất cả"
):
    filtered = filtered[
        filtered["gender"].astype(str)
        == str(selected_gender)
    ]


# Filter number_inpatient
if (
    "number_inpatient" in filtered.columns
    and selected_inpatient != (0, max_inpatient)
):

    inpatient_numeric = pd.to_numeric(
        filtered["number_inpatient"],
        errors="coerce"
    )

    filtered = filtered[
        inpatient_numeric.between(
            selected_inpatient[0],
            selected_inpatient[1],
            inclusive="both"
        )
    ]


# =========================================================
# 7. HEADER
# =========================================================

st.html(
    """
    <div class="hero">

        <h1>
            🩺 Dashboard tái nhập viện bệnh nhân tiểu đường
        </h1>

        <p>
            TV5 · Tổng hợp kết quả phân tích dữ liệu
        </p>

    </div>
    """
)


# =========================================================
# 8. TỔNG QUAN
# =========================================================

st.html(
    """
    <div class="section-title">
        📌 Tổng quan dữ liệu
    </div>
    """
)

st.html(
    """
    <div class="section-desc">
        Các chỉ số tổng quan được tính lại theo tập dữ liệu
        sau khi áp dụng bộ lọc.
    </div>
    """
)


# ---------------------------------------------------------
# KPI
# ---------------------------------------------------------

total_cases = len(filtered)


if (
    total_cases > 0
    and "target_30days" in filtered.columns
):

    readmit_cases = int(
        filtered["target_30days"].sum()
    )

    readmit_rate = (
        readmit_cases
        / total_cases
        * 100
    )

else:

    readmit_cases = 0
    readmit_rate = 0


if (
    total_cases > 0
    and "time_in_hospital" in filtered.columns
):

    avg_stay = filtered["time_in_hospital"].mean()

else:

    avg_stay = 0


if (
    total_cases > 0
    and "num_medications" in filtered.columns
):

    avg_medications = filtered["num_medications"].mean()

else:

    avg_medications = 0


k1, k2, k3, k4 = st.columns(4)


with k1:

    st.html(
        f"""
        <div class="kpi-card" style="border-left-color:{KPI_ACCENTS[0]};">

            <div class="kpi-label">
                Tổng số ca
            </div>

            <div class="kpi-value" style="color:{KPI_ACCENTS[0]} !important;">
                {total_cases:,}
            </div>

            <div class="kpi-note">
                Sau khi áp dụng bộ lọc
            </div>

        </div>
        """
    )


with k2:

    st.html(
        f"""
        <div class="kpi-card" style="border-left-color:{KPI_ACCENTS[1]};">

            <div class="kpi-label">
                Tái nhập viện &lt;30 ngày
            </div>

            <div class="kpi-value" style="color:{KPI_ACCENTS[1]} !important;">
                {readmit_rate:.2f}%
            </div>

            <div class="kpi-note">
                {readmit_cases:,} ca
            </div>

        </div>
        """
    )


with k3:

    st.html(
        f"""
        <div class="kpi-card" style="border-left-color:{KPI_ACCENTS[2]};">

            <div class="kpi-label">
                Thời gian nằm viện TB
            </div>

            <div class="kpi-value" style="color:{KPI_ACCENTS[2]} !important;">
                {avg_stay:.2f}
            </div>

            <div class="kpi-note">
                ngày
            </div>

        </div>
        """
    )


with k4:

    st.html(
        f"""
        <div class="kpi-card" style="border-left-color:{KPI_ACCENTS[3]};">

            <div class="kpi-label">
                Số thuốc TB
            </div>

            <div class="kpi-value" style="color:{KPI_ACCENTS[3]} !important;">
                {avg_medications:.2f}
            </div>

            <div class="kpi-note">
                loại thuốc / ca
            </div>

        </div>
        """
    )


st.markdown("---")


# =========================================================
# 9. TV2 - PHÂN TÍCH MÔ TẢ
# =========================================================

st.html(
    """
    <div class="section-title">
        📊 Phân tích mô tả
    </div>
    """
)

st.html(
    """
    <div class="section-desc">
        Các biểu đồ mô tả thực trạng tái nhập viện,
        thời gian điều trị và tiền sử nhập viện.
    </div>
    """
)


# =========================================================
# HÌNH 1 - DONUT
# =========================================================

col1, col2 = st.columns(2)


with col1:

    st.markdown(
        "### 1. Tỷ lệ tái nhập viện trong 30 ngày"
    )

    if (
        total_cases > 0
        and "target_30days" in filtered.columns
    ):

        readmit = int(
            filtered["target_30days"].sum()
        )

        non_readmit = (
            total_cases - readmit
        )

        chart_df = pd.DataFrame({
            "Tình trạng": [
                "Không tái nhập viện",
                "Tái nhập viện"
            ],
            "Số ca": [
                non_readmit,
                readmit
            ]
        })

        fig1 = px.pie(
            chart_df,
            names="Tình trạng",
            values="Số ca",
            hole=0.58,
            color="Tình trạng",
            color_discrete_map={
                "Không tái nhập viện": COLOR_NONREADMIT,
                "Tái nhập viện": COLOR_READMIT
            }
        )

        fig1.update_traces(
            textinfo="percent",
            textfont=dict(
                size=15,
                color="#ffffff",
                family="Arial Black"
            ),
            hovertemplate=(
                "<b>%{label}</b>"
                "<br>Số ca: %{value:,}"
                "<br>Tỷ lệ: %{percent}"
                "<extra></extra>"
            ),
            marker=dict(
                line=dict(
                    color="white",
                    width=3
                )
            ),
            
        )

        fig1.add_annotation(
            x=0.5,
            y=0.5,
            text=(
                f"<b>{total_cases:,}</b>"
                "<br><span style='font-size:12px'>TỔNG CA</span>"
            ),
            showarrow=False,
            font=dict(
                color="#111827",
                size=18
            )
        )

        fig1.update_layout(
            title=dict(
                text=bold("Cơ cấu tái nhập viện"),
                font=dict(size=16, color="#111827"),
                x=0.5,
                xanchor="center"
            ),
            height=480,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(
                color="#111827",
                size=13
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.09,
                xanchor="center",
                x=0.5,
                font=dict(
                    size=14,
                    color="#111111",
                    family="Arial"
                ),
                bgcolor="rgba(255,255,255,0)"
            ),
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=80
            )
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        st.caption(
            "Hình 1. Tỷ lệ tái nhập viện trong 30 ngày"
        )


# =========================================================
# HÌNH 2 - HISTOGRAM (dạng gradient màu theo tần suất)
# =========================================================

with col2:

    st.markdown(
        "### 2. Phân bố thời gian nằm viện"
    )

    if (
        total_cases > 0
        and "time_in_hospital" in filtered.columns
    ):

        stay_data = filtered[
            "time_in_hospital"
        ].dropna()

        mean_stay = stay_data.mean()
        median_stay = stay_data.median()

        n_bins = int(
            stay_data.max() - stay_data.min() + 1
        )

        # Tính counts thủ công để tô màu gradient theo chiều cao cột
        # (số liệu/kết quả thống kê giữ nguyên như histogram gốc)
        counts, bin_edges = np.histogram(
            stay_data, bins=n_bins,
            range=(stay_data.min(), stay_data.max() + 1)
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        max_c = counts.max() if counts.max() > 0 else 1
        bar_colors = px.colors.sample_colorscale(
            "Blues",
            [0.25 + 0.65 * (c / max_c) for c in counts]
        )

        fig2 = go.Figure(
            go.Bar(
                x=bin_centers,
                y=counts,
                marker=dict(
                    color=bar_colors,
                    line=dict(color="#1e3a8a", width=1)
                ),
                hovertemplate=(
                    "Thời gian: %{x} ngày"
                    "<br>Số lượng: %{y}"
                    "<extra></extra>"
                )
            )
        )

        fig2.add_vline(
            x=mean_stay,
            line_width=3,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=bold(f"Mean = {mean_stay:.2f}"),
            annotation_position="top"
        )

        fig2.add_vline(
            x=median_stay,
            line_width=3,
            line_dash="dot",
            line_color="#16a34a",
            annotation_text=bold(f"Median = {median_stay:.0f}"),
            annotation_position="top left"
        )

        fig2.update_layout(
            title=dict(
                text=bold("Phân bố thời gian nằm viện"),
                font=dict(size=16, color="#111827"),
                x=0.5,
                xanchor="center"
            ),
            height=480,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(
                color="#111827",
                size=13
            ),
            xaxis=dict(
                title=dict(
                    text=bold("Thời gian nằm viện (ngày)"),
                    font=dict(
                        color="#111827",
                        size=14
                    )
                ),
                tickfont=dict(
                    color="#111827",
                    size=12
                ),
                dtick=1,
                showgrid=True,
                gridcolor="#e5e7eb"
            ),
            yaxis=dict(
                title=dict(
                    text=bold("Số lượng bệnh nhân"),
                    font=dict(
                        color="#111827",
                        size=14
                    )
                ),
                tickfont=dict(
                    color="#111827",
                    size=12
                ),
                showgrid=True,
                gridcolor="#e5e7eb"
            ),
            legend=dict(
                orientation="h"
            ),
            margin=dict(
                l=70,
                r=30,
                t=55,
                b=70
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.caption(
            "Hình 2. Phân bố thời gian nằm viện"
        )


# =========================================================
# HÌNH 3 - LOLLIPOP THEO NHÓM TUỔI (đa màu theo nhóm tuổi)
# =========================================================

col3, col4 = st.columns(2)


with col3:

    st.markdown(
        "### 3. Tái nhập viện theo nhóm tuổi"
    )

    if (
        "age" in filtered.columns
        and "target_30days" in filtered.columns
    ):

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
            "[90-100)"
        ]

        age_temp = filtered.copy()

        age_temp["age"] = (
            age_temp["age"]
            .astype(str)
        )

        age_df = (
            age_temp
            .groupby("age")["target_30days"]
            .agg(["mean", "count"])
            .reset_index()
        )

        age_df["rate"] = (
            age_df["mean"] * 100
        )

        age_df = (
            age_df
            .set_index("age")
            .reindex([
                x for x in age_order
                if x in age_df["age"].values
            ])
            .dropna()
            .reset_index()
        )

        if len(age_df) > 0:

            overall_rate = readmit_rate

            age_colors = [
                PALETTE_AGE[i % len(PALETTE_AGE)]
                for i in range(len(age_df))
            ]

            fig3 = go.Figure()

            # Đường ngang tỷ lệ chung
            fig3.add_hline(
                y=overall_rate,
                line_dash="dash",
                line_width=2,
                line_color="#6b7280",
                annotation_text=(
                    bold(f"Tỷ lệ chung = {overall_rate:.2f}%")
                ),
                annotation_position="top right"
            )

            # Lollipop (mỗi nhóm tuổi 1 màu riêng)
            for idx, row in age_df.iterrows():

                fig3.add_trace(
                    go.Scatter(
                        x=[
                            row["age"],
                            row["age"]
                        ],
                        y=[
                            0,
                            row["rate"]
                        ],
                        mode="lines",
                        line=dict(
                            color=age_colors[idx],
                            width=6
                        ),
                        showlegend=False,
                        hoverinfo="skip"
                    )
                )

            fig3.add_trace(
                go.Scatter(
                    x=age_df["age"],
                    y=age_df["rate"],
                    mode="markers+text",
                    text=[
                        f"<b>{x:.1f}%</b>"
                        for x in age_df["rate"]
                    ],
                    textposition="top center",
                    marker=dict(
                        size=16,
                        color=age_colors,
                        line=dict(
                            color="white",
                            width=2
                        )
                    ),
                    textfont=dict(
                        color="#111827",
                        size=12
                    ),
                    customdata=np.column_stack(
                        (
                            age_df["count"].astype(int),
                            age_df["rate"]
                        )
                    ),
                    hovertemplate=(
                        "<b>Nhóm tuổi: %{x}</b>"
                        "<br>Tỷ lệ tái nhập viện: %{y:.2f}%"
                        "<br>Số ca: %{customdata[0]}"
                        "<extra></extra>"
                    ),
                    showlegend=False
                )
            )

            fig3.update_layout(
                title=dict(
                    text=bold("Tỷ lệ tái nhập viện theo nhóm tuổi"),
                    font=dict(size=16, color="#111827"),
                    x=0.5,
                    xanchor="center"
                ),
                height=450,
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(
                    color="#111827",
                    size=13
                ),
                xaxis=dict(
                    title=dict(
                        text=bold("Nhóm tuổi"),
                        font=dict(
                            color="#111827",
                            size=14
                        )
                    ),
                    tickfont=dict(
                        color="#111827",
                        size=11
                    ),
                    showgrid=False
                ),
                yaxis=dict(
                    title=dict(
                        text=bold("Tỷ lệ tái nhập viện (%)"),
                        font=dict(
                            color="#111827",
                            size=14
                        )
                    ),
                    tickfont=dict(
                        color="#111827",
                        size=12
                    ),
                    showgrid=True,
                    gridcolor="#e5e7eb",
                    rangemode="tozero"
                ),
                margin=dict(
                    l=70,
                    r=30,
                    t=60,
                    b=70
                )
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

            st.caption(
                "Hình 3. Tỷ lệ tái nhập viện theo nhóm tuổi"
            )


# =========================================================
# HÌNH 4 - NUMBER INPATIENT (đa màu theo nhóm)
# =========================================================

with col4:

    st.markdown(
        "### 4. Tái nhập viện theo số lần nhập viện trước"
    )

    if (
        "number_inpatient" in filtered.columns
        and "target_30days" in filtered.columns
    ):

        temp = filtered[
            [
                "number_inpatient",
                "target_30days"
            ]
        ].copy()

        temp["number_inpatient"] = pd.to_numeric(
            temp["number_inpatient"],
            errors="coerce"
        )

        temp = temp.dropna()

        temp["inpatient_group"] = temp[
            "number_inpatient"
        ].apply(
            lambda x:
            ">= 4"
            if x >= 4
            else str(int(x))
        )

        order = [
            "0",
            "1",
            "2",
            "3",
            ">= 4"
        ]

        inpatient_df = (
            temp
            .groupby("inpatient_group")
            ["target_30days"]
            .agg(["mean", "count"])
            .reindex(order)
            .dropna()
            .reset_index()
        )

        inpatient_df["rate"] = (
            inpatient_df["mean"] * 100
        )

        if len(inpatient_df) > 0:

            fig4 = px.bar(
                inpatient_df,
                x="inpatient_group",
                y="rate",
                color="inpatient_group",
                color_discrete_sequence=PALETTE_GROUP,
                text=inpatient_df.apply(
                    lambda row:
                    f"<b>{row['rate']:.1f}%</b><br>n={int(row['count'])}",
                    axis=1
                ),
                labels={
                    "inpatient_group":
                        "Số lần nhập viện trước đó",
                    "rate":
                        "Tỷ lệ tái nhập viện (%)"
                }
            )

            fig4.update_traces(
                textposition="outside",
                textfont=dict(
                    color="#111827",
                    size=12
                ),
                hovertemplate=(
                    "<b>Số lần nhập viện: %{x}</b>"
                    "<br>Tỷ lệ tái nhập viện: %{y:.2f}%"
                    "<extra></extra>"
                ),
                marker_line_width=1.2,
                marker_line_color="#374151"
            )

            fig4.add_hline(
                y=readmit_rate,
                line_dash="dash",
                line_width=2,
                line_color="#6b7280",
                annotation_text=(
                    bold(f"Tỷ lệ chung = {readmit_rate:.2f}%")
                ),
                annotation_position="top right"
            )

            fig4.update_layout(
                title=dict(
                    text=bold("Tỷ lệ tái nhập viện theo số lần nhập viện trước đó"),
                    font=dict(size=15, color="#111827"),
                    x=0.5,
                    xanchor="center"
                ),
                height=450,
                paper_bgcolor="white",
                plot_bgcolor="white",
                showlegend=False,
                font=dict(
                    color="#111827",
                    size=13
                ),
                xaxis=dict(
                    title=dict(
                        text=bold("Số lần nhập viện trước đó"),
                        font=dict(
                            color="#111827",
                            size=14
                        )
                    ),
                    tickfont=dict(
                        color="#111827",
                        size=12
                    ),
                    showgrid=False
                ),
                yaxis=dict(
                    title=dict(
                        text=bold("Tỷ lệ tái nhập viện (%)"),
                        font=dict(
                            color="#111827",
                            size=14
                        )
                    ),
                    tickfont=dict(
                        color="#111827",
                        size=12
                    ),
                    showgrid=True,
                    gridcolor="#e5e7eb",
                    rangemode="tozero"
                ),
                margin=dict(
                    l=70,
                    r=30,
                    t=65,
                    b=70
                )
            )

            st.plotly_chart(
                fig4,
                use_container_width=True
            )

            st.caption(
                "Hình 4. Tỷ lệ tái nhập viện theo số lần nhập viện trước đó"
            )


# =========================================================
# 10. TV2 INSIGHT
# =========================================================

st.html(
    """
    <div class="section-title">
        💡 TV2 · Insight
    </div>
    """
)

insight_col1, insight_col2, insight_col3 = st.columns(3)


# ---------------------------------------------------------
# Median + Mode
# ---------------------------------------------------------

if (
    total_cases > 0
    and "time_in_hospital" in filtered.columns
):

    stay_values = filtered[
        "time_in_hospital"
    ].dropna()

    if len(stay_values) > 0:

        stay_median = stay_values.median()

        mode_values = stay_values.mode()

        if len(mode_values) > 0:
            stay_mode = mode_values.iloc[0]
        else:
            stay_mode = 0

    else:

        stay_median = 0
        stay_mode = 0

else:

    stay_median = 0
    stay_mode = 0


# ---------------------------------------------------------
# Insight 1
# ---------------------------------------------------------

with insight_col1:

    st.html(
        f"""
        <div class="info-card" style="border-top-color:{INFO_ACCENTS['tv2']};">

            <div class="badge badge-tv2">
                TV2 · Insight
            </div>

            <h4>Tái nhập viện</h4>

            <p>
                Trong tổng số
                <b>{total_cases:,}</b>
                ca đang được phân tích,
                có
                <b>{readmit_cases:,}</b>
                ca tái nhập viện trong vòng 30 ngày,
                tương ứng với tỷ lệ
                <b>{readmit_rate:.2f}%</b>.
            </p>

            <p>
                Như vậy, trong mỗi 100 ca quan sát có khoảng
                <b>{readmit_rate:.0f}</b> ca thuộc nhóm tái nhập viện
                trong 30 ngày. Chỉ số này mô tả tỷ lệ của
                tập dữ liệu hiện tại sau khi áp dụng bộ lọc.
            </p>

        </div>
        """
    )


# ---------------------------------------------------------
# Insight 2
# ---------------------------------------------------------

with insight_col2:

    st.html(
        f"""
        <div class="info-card" style="border-top-color:{INFO_ACCENTS['tv2']};">

            <div class="badge badge-tv2">
                TV2 · Insight
            </div>

            <h4>Thời gian nằm viện</h4>

            <p>
                Thời gian nằm viện trung bình là
                <b>{avg_stay:.2f}</b> ngày.
                Giá trị trung vị là
                <b>{stay_median:.0f}</b> ngày,
                trong khi thời gian phổ biến nhất là
                <b>{stay_mode:.0f}</b> ngày.
            </p>

            <p>
                Việc so sánh trung bình, trung vị và mode giúp
                quan sát đặc điểm phân bố thời gian điều trị
                của nhóm bệnh nhân sau khi lọc dữ liệu và hỗ trợ đánh giá xu hướng điều trị chung.
            </p>

        </div>
        """
    )


# ---------------------------------------------------------
# Insight 3
# ---------------------------------------------------------

if (
    "gender" in filtered.columns
    and total_cases > 0
):

    gender_counts = (
        filtered["gender"]
        .value_counts()
    )

    if len(gender_counts) > 0:

        largest_gender = gender_counts.index[0]

        largest_gender_pct = (
            gender_counts.iloc[0]
            / total_cases
            * 100
        )

    else:

        largest_gender = "N/A"
        largest_gender_pct = 0

else:

    largest_gender = "N/A"
    largest_gender_pct = 0


with insight_col3:

    st.html(
        f"""
        <div class="info-card" style="border-top-color:{INFO_ACCENTS['tv2']};">

            <div class="badge badge-tv2">
                TV2 · Insight
            </div>

            <h4>Giới tính</h4>

            <p>
                Trong tập dữ liệu đang được phân tích,
                nhóm giới tính có tỷ lệ cao nhất là
                <b>{largest_gender}</b>,
                với khoảng
                <b>{largest_gender_pct:.2f}%</b>
                tổng số ca.
            </p>

            <p>
                Kết quả này mô tả cơ cấu giới tính của tập dữ liệu
                hiện tại và thay đổi theo các điều kiện lọc
                được lựa chọn ở thanh bên, giúp đánh giá sự khác biệt giữa các nhóm bệnh nhân.
            </p>

        </div>
        """
    )


# =========================================================
# 11. TV3 - PHÂN TÍCH THỐNG KÊ
# =========================================================

st.html(
    """
    <div class="section-title">
        📈 Phân tích thống kê
    </div>
    """
)

st.html(
    """
    <div class="section-desc">
        Các biểu đồ TV3 được dựng lại trực tiếp từ dữ liệu sạch
        với cùng biến phân tích. Biểu đồ number_inpatient được
        bỏ để tránh trùng với Hình 4 của TV2.
    </div>
    """
)


# =========================================================
# HÌNH 5 - SCATTER (tô màu theo mật độ số thuốc)
# =========================================================

st.markdown(
    "### 5. time_in_hospital ↔ num_medications"
)

if (
    "time_in_hospital" in filtered.columns
    and "num_medications" in filtered.columns
):

    clean_corr = filtered[
        [
            "time_in_hospital",
            "num_medications"
        ]
    ].copy()

    clean_corr["time_in_hospital"] = pd.to_numeric(
        clean_corr["time_in_hospital"],
        errors="coerce"
    )

    clean_corr["num_medications"] = pd.to_numeric(
        clean_corr["num_medications"],
        errors="coerce"
    )

    clean_corr = clean_corr.dropna()

    if len(clean_corr) >= 2:

        corr = clean_corr[
            "time_in_hospital"
        ].corr(
            clean_corr["num_medications"]
        )

        fig_corr = px.scatter(
            clean_corr,
            x="time_in_hospital",
            y="num_medications",
            opacity=0.5,
            color="num_medications",
            color_continuous_scale="Turbo",
            labels={
                "time_in_hospital":
                    "Thời gian nằm viện (ngày)",
                "num_medications":
                    "Số loại thuốc"
            },
            title=bold(
                "Mối quan hệ giữa thời gian nằm viện "
                "và số loại thuốc"
            )
        )

        fig_corr.update_traces(
            marker=dict(
                size=6,
                line=dict(
                    width=0.5,
                    color="#1f2937"
                )
            ),
            hovertemplate=(
                "Thời gian nằm viện: %{x} ngày"
                "<br>Số loại thuốc: %{y}"
                "<extra></extra>"
            )
        )

        fig_corr.update_coloraxes(
            colorbar=dict(
                title=dict(text=bold("Số thuốc"))
            )
        )

        unique_x = (
            clean_corr["time_in_hospital"]
            .nunique()
        )

        if unique_x >= 2:

            coef = np.polyfit(
                clean_corr["time_in_hospital"],
                clean_corr["num_medications"],
                1
            )

            x_line = np.linspace(
                clean_corr["time_in_hospital"].min(),
                clean_corr["time_in_hospital"].max(),
                100
            )

            y_line = (
                coef[0] * x_line
                + coef[1]
            )

            fig_corr.add_scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Đường xu hướng",
                line=dict(
                    width=4,
                    color="#dc2626"
                )
            )

        corr_text = (
            f"{corr:.3f}"
            if pd.notna(corr)
            else "N/A"
        )

        if pd.notna(corr):

            if corr > 0:
                relation_text = "Tương quan dương"
            elif corr < 0:
                relation_text = "Tương quan âm"
            else:
                relation_text = "Tương quan gần 0"

        else:

            relation_text = "Không xác định"


        fig_corr.add_annotation(
            x=0.98,
            y=0.96,
            xref="paper",
            yref="paper",
            text=(
                f"<b>Pearson r = {corr_text}</b>"
                f"<br>{relation_text}"
            ),
            showarrow=False,
            align="left",
            font=dict(
                size=15,
                color="#111827"
            ),
            bgcolor="white",
            bordercolor="#d1d5db",
            borderwidth=1,
            borderpad=9
        )

        fig_corr.update_layout(
            height=520,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(
                color="#111827",
                size=13
                
            ),
            title=dict(
                font=dict(
                    size=20,
                    color="#111827"
                ),
                x=0.5,
                xanchor="center"
                
            ),
            xaxis=dict(
                title=dict(
                    text=bold("Thời gian nằm viện (ngày)"),
                    font=dict(color="#111827", size=14)
                ),
                tickfont=dict(
                    color="#111827",
                    size=12
                ),
                showgrid=True,
                gridcolor="#e5e7eb",
                zeroline=False
            ),
            yaxis=dict(
                title=dict(
                    text=bold("Số loại thuốc"),
                    font=dict(color="#111827", size=14)
                ),
                tickfont=dict(
                    color="#111827",
                    size=12
                ),
                showgrid=True,
                gridcolor="#e5e7eb",
                zeroline=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(
                l=75,
                r=55,
                t=95,
                b=70
            )
        )

        st.plotly_chart(
            fig_corr,
            use_container_width=True
        )

        st.caption(
            "Hình 5. Mối quan hệ giữa time_in_hospital và num_medications"
        )

        st.html(
            f"""
            <div class="info-card" style="border-top-color:{INFO_ACCENTS['tv3']};">

                <div class="badge badge-tv3">
                    TV3 · Insight
                </div>

                <p>
                    Hệ số tương quan Pearson giữa thời gian nằm viện
                    và số thuốc là <b>{corr_text}</b>.
                    Dữ liệu cho thấy xu hướng
                    <b>{relation_text.lower()}</b>
                    giữa hai biến trong tập dữ liệu đang được lọc.
                </p>

                <p>
                    <b>Lưu ý:</b> đây là mối liên hệ thống kê,
                    không kết luận quan hệ nhân quả.
                </p>

            </div>
            """
        )


# =========================================================
# HÌNH 6 + 7 (đa màu theo nhóm)
# =========================================================

col5, col6 = st.columns(2)


for column, variable, title, figure_number in [

    (
        col5,
        "number_emergency",
        "6. Tỷ lệ tái nhập viện theo number_emergency",
        6
    ),

    (
        col6,
        "number_outpatient",
        "7. Tỷ lệ tái nhập viện theo number_outpatient",
        7
    )

]:

    with column:

        st.markdown(
            f"### {title}"
        )

        if (
            variable in filtered.columns
            and "target_30days" in filtered.columns
        ):

            temp = filtered[
                [
                    variable,
                    "target_30days"
                ]
            ].copy()

            temp[variable] = pd.to_numeric(
                temp[variable],
                errors="coerce"
            )

            temp = temp.dropna(
                subset=[variable, "target_30days"]
            )

            if len(temp) > 0:

                temp["group"] = temp[variable].apply(
                    lambda x:
                    "7+"
                    if x >= 7
                    else str(int(x))
                )

                order = [
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7+"
                ]

                rate_df = (
                    temp
                    .groupby("group")["target_30days"]
                    .agg(["mean", "count"])
                    .reindex(order)
                    .dropna()
                    .reset_index()
                )

                rate_df["rate"] = (
                    rate_df["mean"] * 100
                )

                if len(rate_df) > 0:

                    fig_var = px.bar(
                        rate_df,
                        x="group",
                        y="rate",
                        color="group",
                        color_discrete_sequence=PALETTE_GROUP,
                        text=rate_df.apply(
                            lambda row:
                            f"<b>{row['rate']:.1f}%</b><br>"
                            f"n={int(row['count'])}",
                            axis=1
                        ),
                        labels={
                            "group": variable,
                            "rate":
                                "Tỷ lệ tái nhập viện "
                                "trong 30 ngày (%)"
                        },
                        title=bold(
                            f"Tỷ lệ tái nhập viện theo "
                            f"{variable}"
                        )
                    )

                    fig_var.update_traces(
                        textposition="outside",
                        textfont=dict(
                            color="#111827",
                            size=11
                        ),
                        hovertemplate=(
                            f"<b>{variable}: %{{x}}</b>"
                            "<br>Tỷ lệ tái nhập viện: %{y:.2f}%"
                            "<extra></extra>"
                        ),
                        marker_line_width=1.2,
                        marker_line_color="#374151"
                    )

                    fig_var.add_hline(
                        y=readmit_rate,
                        line_dash="dash",
                        line_width=2,
                        line_color="#6b7280",
                        annotation_text=(
                            bold(f"Tỷ lệ chung = {readmit_rate:.2f}%")
                        ),
                        annotation_position="top right"
                    )

                    fig_var.update_layout(
                        height=470,
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                        showlegend=False,
                        font=dict(
                            color="#111827",
                            size=13
                        ),
                        title=dict(
                            font=dict(
                                size=18,
                                color="#111827"
                            ),
                            x=0.5,
                            xanchor="center"
                        ),
                        xaxis=dict(
                            title=dict(
                                text=bold(variable),
                                font=dict(
                                    color="#111827",
                                    size=14
                                )
                            ),
                            tickfont=dict(
                                color="#111827",
                                size=12
                            ),
                            showgrid=False
                        ),
                        yaxis=dict(
                            title=dict(
                                text=bold(
                                    "Tỷ lệ tái nhập viện "
                                    "trong 30 ngày (%)"
                                ),
                                font=dict(
                                    color="#111827",
                                    size=14
                                )
                            ),
                            tickfont=dict(
                                color="#111827",
                                size=12
                            ),
                            showgrid=True,
                            gridcolor="#e5e7eb",
                            rangemode="tozero"
                        ),
                        margin=dict(
                            l=75,
                            r=30,
                            t=80,
                            b=70
                        )
                    )

                    st.plotly_chart(
                        fig_var,
                        use_container_width=True
                    )

                    st.caption(
                        f"Hình {figure_number}. "
                        f"Tỷ lệ tái nhập viện theo {variable}"
                    )


# =========================================================
# HÌNH 8 - BOXPLOT (đa màu theo nhóm tuổi)
# =========================================================

st.markdown(
    "### 8. age ↔ time_in_hospital"
)

if (
    "age" in filtered.columns
    and "time_in_hospital" in filtered.columns
):

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
        "[90-100)"
    ]

    box_data = filtered[
        [
            "age",
            "time_in_hospital"
        ]
    ].copy()

    box_data["age"] = (
        box_data["age"]
        .astype(str)
    )

    box_data["time_in_hospital"] = pd.to_numeric(
        box_data["time_in_hospital"],
        errors="coerce"
    )

    box_data = box_data.dropna()

    existing = [
        x for x in age_order
        if x in box_data["age"].values
    ]

    if len(existing) > 0:

        fig_box = px.box(
            box_data,
            x="age",
            y="time_in_hospital",
            color="age",
            color_discrete_sequence=PALETTE_AGE,
            category_orders={
                "age": existing
            },
            points=False,
            labels={
                "age": "Nhóm tuổi",
                "time_in_hospital":
                    "Thời gian nằm viện (ngày)"
            },
            title=bold("Phân bố thời gian nằm viện theo nhóm tuổi")
        )

        fig_box.update_traces(
            hovertemplate=(
                "<b>Nhóm tuổi: %{x}</b>"
                "<br>Thời gian nằm viện: %{y} ngày"
                "<extra></extra>"
            ),
            line=dict(
                width=2
            )
        )

        fig_box.update_layout(
            height=510,
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            font=dict(
                color="#111827",
                size=13
            ),
            title=dict(
                font=dict(
                    size=20,
                    color="#111827"
                ),
                x=0.5,
                xanchor="center"
            ),
            xaxis=dict(
                title=dict(
                    text=bold("Nhóm tuổi"),
                    font=dict(
                        color="#111827",
                        size=14
                    )
                ),
                tickfont=dict(
                    color="#111827",
                    size=11
                ),
                showgrid=False
            ),
            yaxis=dict(
                title=dict(
                    text=bold("Thời gian nằm viện (ngày)"),
                    font=dict(
                        color="#111827",
                        size=14
                    )
                ),
                tickfont=dict(
                    color="#111827",
                    size=12
                ),
                showgrid=True,
                gridcolor="#e5e7eb"
            ),
            margin=dict(
                l=75,
                r=35,
                t=90,
                b=70
            )
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True
        )

        st.caption(
            "Hình 8. Mối quan hệ giữa age và time_in_hospital"
        )


        # -----------------------------------------------------
        # Median theo nhóm tuổi
        # -----------------------------------------------------

        median_age = (
            box_data
            .groupby("age")[
                "time_in_hospital"
            ]
            .median()
            .reindex(existing)
            .dropna()
        )

        if len(median_age) > 0:

            max_age = median_age.idxmax()
            min_age = median_age.idxmin()

            st.html(
                f"""
                <div class="info-card" style="border-top-color:{INFO_ACCENTS['tv3']};">

                    <div class="badge badge-tv3">
                        TV3 · Insight
                    </div>

                    <p>
                        Nhóm tuổi <b>{max_age}</b>
                        có thời gian nằm viện trung vị cao nhất
                        (<b>{median_age.max():.1f}</b> ngày),
                        trong khi nhóm <b>{min_age}</b>
                        có median thấp nhất
                        (<b>{median_age.min():.1f}</b> ngày).
                    </p>

                    <p>
                        Chênh lệch median giữa hai nhóm là
                        <b>
                            {median_age.max() - median_age.min():.1f}
                        </b>
                        ngày.
                    </p>

                </div>
                """
            )


# =========================================================
# 12. TV4 - MACHINE LEARNING
# =========================================================

st.html(
    """
    <div class="section-title">
        🤖 Machine Learning
    </div>
    """
)

st.html(
    """
    <div class="section-desc">
        Kết quả Logistic Regression và Random Forest dùng để
        dự đoán khả năng tái nhập viện trong 30 ngày.
    </div>
    """
)


# =========================================================
# MODEL RESULTS
# =========================================================

if MODEL_RESULTS_PATH.exists():

    try:

        model_results = pd.read_csv(
            MODEL_RESULTS_PATH
        )

        st.markdown(
            "### Kết quả các mô hình"
        )

        st.dataframe(
            model_results.style.background_gradient(
                cmap="Blues",
                subset=[c for c in model_results.columns if c != "Model"]
            ),
            use_container_width=True,
            hide_index=True
        )

    except Exception:

        st.warning(
            "Không đọc được model_results.csv."
        )

else:

    st.info(
        "Chưa có model_results.csv. "
        "Dashboard vẫn hiển thị các biểu đồ TV4 "
        "được lưu trong thư mục figures."
    )


# =========================================================
# HÌNH 9
# =========================================================

st.markdown(
    "### 9. So sánh hiệu quả các mô hình"
)

path = FIGURES_DIR / "model_comparison.png"

if path.exists():

    left, center, right = st.columns([0.7, 3.6, 0.7])

    with center:

        st.image(
            str(path),
            use_container_width=True
        )

        st.caption(
            "Hình 9. So sánh hiệu quả Logistic Regression và Random Forest"
        )

else:

    st.warning(
        "Chưa có file model_comparison.png"
    )


# =========================================================
# HÌNH 10 + 11
# =========================================================

cm1, cm2 = st.columns(2)


with cm1:

    st.markdown(
        "### 10. Confusion Matrix – Logistic Regression"
    )

    path = (
        FIGURES_DIR
        / "confusion_matrix_logistic_regression.png"
    )

    if path.exists():

        st.image(
            str(path),
            use_container_width=True
        )

        st.caption(
            "Hình 10. Confusion Matrix – Logistic Regression"
        )

    else:

        st.warning(
            "Chưa có file confusion_matrix_logistic_regression.png"
        )


with cm2:

    st.markdown(
        "### 11. Confusion Matrix – Random Forest"
    )

    path = (
        FIGURES_DIR
        / "confusion_matrix_random_forest.png"
    )

    if path.exists():

        st.image(
            str(path),
            use_container_width=True
        )

        st.caption(
            "Hình 11. Confusion Matrix – Random Forest"
        )

    else:

        st.warning(
            "Chưa có file confusion_matrix_random_forest.png"
        )


# =========================================================
# HÌNH 12
# =========================================================

st.markdown(
    "### 12. ROC Curve"
)

path = FIGURES_DIR / "roc_curve.png"

if path.exists():

    left, center, right = st.columns([0.7, 3.6, 0.7])

    with center:

        st.image(
            str(path),
            use_container_width=True
        )

        st.caption(
            "Hình 12. ROC Curve của hai mô hình"
        )

else:

    st.warning(
        "Chưa có file roc_curve.png"
    )


# =========================================================
# HÌNH 13
# =========================================================

st.markdown(
    "### 13. Top 20 Feature Importance – Random Forest"
)

path = (
    FIGURES_DIR
    / "random_forest_feature_importance.png"
)

if path.exists():

    left, center, right = st.columns([0.7, 3.6, 0.7])

    with center:

        st.image(
            str(path),
            use_container_width=True
        )

        st.caption(
            "Hình 13. Top 20 Feature Importance – Random Forest"
        )

else:

    st.warning(
        "Chưa có file random_forest_feature_importance.png"
    )


# =========================================================
# FEATURE IMPORTANCE TABLE
# =========================================================

if FEATURE_IMPORTANCE_PATH.exists():

    try:

        feature_importance = pd.read_csv(
            FEATURE_IMPORTANCE_PATH
        )

        st.markdown(
            "### Chi tiết Feature Importance"
        )

        st.dataframe(
            feature_importance.head(20).style.background_gradient(
                cmap="Oranges",
                subset=["Importance"] if "Importance" in feature_importance.columns else None
            ),
            use_container_width=True,
            hide_index=True
        )

    except Exception:

        pass


# =========================================================
# 13. TỔNG KẾT
# =========================================================

st.html(
    """
    <div class="section-title">
        📝 Tổng kết
    </div>
    """
)

sum1, sum2, sum3 = st.columns(3)


with sum1:

    st.html(
        f"""
        <div class="info-card" style="border-top-color:{INFO_ACCENTS['tv2']};">

            <div class="badge badge-tv2">
                TV2
            </div>

            <h4>Thực trạng</h4>

            <p>
                Mô tả tỷ lệ tái nhập viện,
                thời gian nằm viện,
                nhóm tuổi và tiền sử nhập viện.
            </p>

        </div>
        """
    )


with sum2:

    st.html(
        f"""
        <div class="info-card" style="border-top-color:{INFO_ACCENTS['tv3']};">

            <div class="badge badge-tv3">
                TV3
            </div>

            <h4>Phân tích thống kê</h4>

            <p>
                Kiểm tra mối liên hệ giữa thời gian nằm viện,
                số thuốc, tuổi và lịch sử sử dụng dịch vụ y tế.
            </p>

        </div>
        """
    )


with sum3:

    st.html(
        f"""
        <div class="info-card" style="border-top-color:{INFO_ACCENTS['tv4']};">

            <div class="badge badge-tv4">
                TV4
            </div>

            <h4>Dự đoán</h4>

            <p>
                Xây dựng Logistic Regression và Random Forest,
                đồng thời đánh giá bằng các chỉ số và biểu đồ
                của mô hình.
            </p>

        </div>
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.html(
    """
    <div class="footer">
        TV5 · Dashboard tổng hợp phân tích tái nhập viện
        bệnh nhân tiểu đường
    </div>
    """
)