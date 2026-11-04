from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Breast Cancer Patient Insights",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).with_name("Breast_Cancer.csv")
REQUIRED_COLUMNS = {
    "Age",
    "Race",
    "Marital Status",
    "T Stage",
    "N Stage",
    "6th Stage",
    "differentiate",
    "Grade",
    "A Stage",
    "Tumor Size",
    "Estrogen Status",
    "Progesterone Status",
    "Regional Node Examined",
    "Reginol Node Positive",
    "Survival Months",
    "Status",
}
COLORS = ["#4F46E5", "#14B8A6", "#F59E0B", "#EC4899", "#64748B", "#8B5CF6"]
STATUS_COLORS = {"Alive": "#14B8A6", "Dead": "#E11D48"}


@st.cache_data(show_spinner="Loading and preparing patient records...")
def load_data(contents: bytes) -> tuple[pd.DataFrame, int]:
    data = pd.read_csv(BytesIO(contents), encoding="utf-8-sig")
    data.columns = data.columns.str.strip()
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing_columns))}")

    text_columns = data.select_dtypes(include=["object", "str"]).columns
    for column in text_columns:
        data[column] = data[column].astype("string").str.strip()

    data["Grade"] = data["Grade"].replace({"anaplastic; Grade IV": "4"})
    data["Grade"] = data["Grade"].replace({"4": "4"})
    data["Grade"] = data["Grade"].map(
        lambda value: f"Grade {value}" if value in {"1", "2", "3", "4"} else value
    )
    for column in (
        "Age",
        "Tumor Size",
        "Regional Node Examined",
        "Reginol Node Positive",
        "Survival Months",
    ):
        original = data[column]
        data[column] = pd.to_numeric(data[column], errors="coerce")
        invalid_values = int((original.notna() & data[column].isna()).sum())
        if invalid_values:
            raise ValueError(f"'{column}' contains {invalid_values:,} non-numeric value(s).")

    duplicate_count = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required data fields contain empty values; please check the CSV.")

    age_bins = [0, 39, 49, 59, 69, 120]
    age_labels = ["Under 40", "40–49", "50–59", "60–69", "70+"]
    data["Age Group"] = pd.cut(
        data["Age"],
        bins=age_bins,
        labels=age_labels,
        include_lowest=True,
    )
    data["Lymph Node Ratio"] = (
        data["Reginol Node Positive"]
        .div(data["Regional Node Examined"].where(data["Regional Node Examined"].ne(0)))
        .fillna(0)
    )
    return data, duplicate_count


def chart_style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=30, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#344054", family="Inter, sans-serif"),
        legend_title_text="",
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E4E7EC")
    fig.update_yaxes(gridcolor="#EAECF0", zerolinecolor="#EAECF0")
    return fig


st.markdown(
    """
    <style>
    .stApp { background: #F7F8FC; }
    [data-testid="stSidebar"] { background: #FFFFFF; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .hero {
        padding: 1.35rem 1.55rem; margin-bottom: 1.1rem; border-radius: 18px;
        background: linear-gradient(115deg, #18224A 0%, #4338CA 65%, #6366F1 100%);
        color: white;
    }
    .hero h1 { margin: 0 0 .25rem 0; font-size: 2rem; }
    .hero p { margin: 0; color: #E0E7FF; }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #EAECF0; padding: 1rem;
        border-radius: 14px; box-shadow: 0 2px 6px rgba(16, 24, 40, .04);
    }
    [data-testid="stMetricLabel"] { color: #667085; }
    div[data-testid="stPlotlyChart"] {
        background: white; border: 1px solid #EAECF0; border-radius: 14px;
        padding: .3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🎗️ Breast Cancer Patient Insights</h1>
      <p>Explore patient demographics, tumor characteristics, and recorded survival outcomes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "Educational analysis of historical, de-identified data only. These associations do not "
    "establish causation and are not medical advice or a tool for diagnosis or treatment."
)

uploaded_file = st.sidebar.file_uploader(
    "Use another patient dataset",
    type=["csv"],
    help="The uploaded CSV must contain the same breast-cancer data columns.",
)
if uploaded_file is not None:
    source_bytes = uploaded_file.getvalue()
    source_name = uploaded_file.name
elif DATA_PATH.exists():
    source_bytes = DATA_PATH.read_bytes()
    source_name = DATA_PATH.name
else:
    st.error("Breast_Cancer.csv is missing. Place it beside app.py or upload a CSV in the sidebar.")
    st.stop()

try:
    all_patients, duplicate_count = load_data(source_bytes)
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(f"Could not load {source_name}: {error}")
    st.stop()

with st.sidebar:
    st.header("Explore the records")
    st.caption(
        f"Source: **{source_name}** · {len(all_patients):,} unique records"
        + (f" · {duplicate_count} duplicate removed" if duplicate_count else "")
    )
    st.markdown("#### Filters")
    age_range = st.slider(
        "Patient age",
        min_value=int(all_patients["Age"].min()),
        max_value=int(all_patients["Age"].max()),
        value=(int(all_patients["Age"].min()), int(all_patients["Age"].max())),
        step=1,
    )
    stages = sorted(all_patients["6th Stage"].dropna().unique().tolist())
    selected_stages = st.multiselect("Cancer stage", stages, default=stages)
    grades = sorted(all_patients["Grade"].dropna().unique().tolist())
    selected_grades = st.multiselect("Tumor grade", grades, default=grades)
    selected_races = st.multiselect(
        "Race",
        sorted(all_patients["Race"].dropna().unique().tolist()),
        default=sorted(all_patients["Race"].dropna().unique().tolist()),
    )
    selected_statuses = st.multiselect(
        "Recorded status",
        sorted(all_patients["Status"].dropna().unique().tolist()),
        default=sorted(all_patients["Status"].dropna().unique().tolist()),
    )
    exclude_outliers = st.checkbox(
        "Exclude tumor-size outliers (IQR)",
        value=False,
        help=(
            "The notebook uses the 1.5×IQR rule for later analysis. This optional filter "
            "excludes those records from all charts and KPIs."
        ),
    )

patients = all_patients.loc[
    all_patients["Age"].between(age_range[0], age_range[1])
    & all_patients["6th Stage"].isin(selected_stages)
    & all_patients["Grade"].isin(selected_grades)
    & all_patients["Race"].isin(selected_races)
    & all_patients["Status"].isin(selected_statuses)
].copy()

outlier_count = 0
if exclude_outliers and not patients.empty:
    q1 = patients["Tumor Size"].quantile(0.25)
    q3 = patients["Tumor Size"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    inlier_mask = patients["Tumor Size"].between(lower_bound, upper_bound)
    outlier_count = int((~inlier_mask).sum())
    patients = patients.loc[inlier_mask].copy()

if patients.empty:
    st.info("No records match the selected filters. Broaden your filter selections and try again.")
    st.stop()

alive_rate = patients["Status"].eq("Alive").mean() * 100
average_survival = patients["Survival Months"].mean()
median_tumor_size = patients["Tumor Size"].median()
dead_rate = patients["Status"].eq("Dead").mean() * 100

st.caption(
    f"Showing **{len(patients):,}** of {len(all_patients):,} unique patient records"
    + (f" · {outlier_count:,} tumor-size outlier(s) excluded" if exclude_outliers else "")
)
kpi_columns = st.columns(4)
kpi_columns[0].metric("Patient records", f"{len(patients):,}")
kpi_columns[1].metric("Recorded alive", f"{alive_rate:.1f}%")
kpi_columns[2].metric("Mean recorded survival", f"{average_survival:.1f} months")
kpi_columns[3].metric("Median tumor size", f"{median_tumor_size:.0f} mm")

overview_tab, tumor_tab, factors_tab, records_tab = st.tabs(
    ["📊 Overview", "🔬 Tumor profile", "🧬 Survival factors", "📋 Patient records"]
)

with overview_tab:
    st.subheader("Patient and outcome overview")
    status_col, age_col = st.columns(2)
    with status_col:
        status_data = (
            patients.groupby("Status", as_index=False)
            .size()
            .rename(columns={"size": "Patients"})
        )
        status_fig = px.pie(
            status_data,
            names="Status",
            values="Patients",
            hole=0.62,
            color="Status",
            color_discrete_map=STATUS_COLORS,
        )
        status_fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(chart_style(status_fig), width="stretch")
    with age_col:
        age_data = (
            patients.groupby("Age Group", observed=False, as_index=False)
            .agg(Patients=("Age", "size"), Alive_rate=("Status", lambda values: values.eq("Alive").mean() * 100))
        )
        age_data["Age Group"] = age_data["Age Group"].astype(str)
        age_data = age_data.loc[age_data["Patients"].gt(0)]
        age_fig = px.bar(
            age_data,
            x="Age Group",
            y="Patients",
            color="Alive_rate",
            color_continuous_scale="Teal",
            labels={"Alive_rate": "Recorded alive (%)"},
            hover_data={"Alive_rate": ":.1f"},
        )
        st.plotly_chart(chart_style(age_fig), width="stretch")

    stage_data = (
        patients.groupby(["6th Stage", "Status"], as_index=False)
        .size()
        .rename(columns={"size": "Patients"})
    )
    stage_fig = px.bar(
        stage_data,
        x="6th Stage",
        y="Patients",
        color="Status",
        barmode="group",
        color_discrete_map=STATUS_COLORS,
        labels={"6th Stage": "Cancer stage"},
    )
    st.plotly_chart(chart_style(stage_fig), width="stretch")
    st.caption(f"Recorded deceased outcome in the current selection: **{dead_rate:.1f}%**.")

with tumor_tab:
    st.subheader("Tumor characteristics")
    tumor_col, grade_col = st.columns(2)
    with tumor_col:
        tumor_fig = px.histogram(
            patients,
            x="Tumor Size",
            color="Status",
            nbins=28,
            barmode="overlay",
            opacity=0.72,
            color_discrete_map=STATUS_COLORS,
            labels={"Tumor Size": "Tumor size (mm)"},
        )
        st.plotly_chart(chart_style(tumor_fig), width="stretch")
    with grade_col:
        grade_data = (
            patients.groupby(["Grade", "Status"], as_index=False)
            .size()
            .rename(columns={"size": "Patients"})
        )
        grade_fig = px.bar(
            grade_data,
            x="Grade",
            y="Patients",
            color="Status",
            barmode="group",
            category_orders={"Grade": ["Grade 1", "Grade 2", "Grade 3", "Grade 4"]},
            color_discrete_map=STATUS_COLORS,
        )
        st.plotly_chart(chart_style(grade_fig), width="stretch")

    scatter = px.scatter(
        patients,
        x="Tumor Size",
        y="Survival Months",
        color="Status",
        color_discrete_map=STATUS_COLORS,
        opacity=0.62,
        hover_data=["Age", "Grade", "6th Stage", "Reginol Node Positive"],
        labels={
            "Tumor Size": "Tumor size (mm)",
            "Survival Months": "Recorded survival (months)",
        },
    )
    st.plotly_chart(chart_style(scatter, 420), width="stretch")

with factors_tab:
    st.subheader("Recorded outcomes across patient groups")
    survival_group = st.selectbox(
        "Compare recorded status by",
        ["Grade", "Estrogen Status", "Progesterone Status", "N Stage", "A Stage", "Race"],
        index=0,
    )
    grouped = (
        patients.groupby([survival_group, "Status"], as_index=False)
        .size()
        .rename(columns={"size": "Patients"})
    )
    grouped["Group total"] = grouped.groupby(survival_group)["Patients"].transform("sum")
    grouped["Share of group (%)"] = grouped["Patients"] / grouped["Group total"] * 100
    outcome_fig = px.bar(
        grouped,
        x=survival_group,
        y="Share of group (%)",
        color="Status",
        barmode="stack",
        color_discrete_map=STATUS_COLORS,
        hover_data={"Patients": True, "Share of group (%)": ":.1f", "Group total": True},
        range_y=[0, 100],
        labels={"Share of group (%)": "Share within group (%)"},
    )
    st.plotly_chart(chart_style(outcome_fig, 410), width="stretch")

    receptor_col, node_col = st.columns(2)
    with receptor_col:
        receptor_data = (
            patients.groupby(["Estrogen Status", "Progesterone Status", "Status"], as_index=False)
            .size()
            .rename(columns={"size": "Patients"})
        )
        receptor_fig = px.bar(
            receptor_data,
            x="Estrogen Status",
            y="Patients",
            color="Status",
            facet_col="Progesterone Status",
            barmode="group",
            color_discrete_map=STATUS_COLORS,
            labels={"Estrogen Status": "Estrogen receptor"},
        )
        receptor_fig.for_each_annotation(
            lambda annotation: annotation.update(text=annotation.text.replace("Progesterone Status=", "Progesterone: "))
        )
        st.plotly_chart(chart_style(receptor_fig, 390), width="stretch")
    with node_col:
        node_fig = px.box(
            patients,
            x="Status",
            y="Lymph Node Ratio",
            color="Status",
            points="outliers",
            color_discrete_map=STATUS_COLORS,
            labels={"Lymph Node Ratio": "Positive / examined nodes"},
        )
        st.plotly_chart(chart_style(node_fig, 390), width="stretch")

    st.caption(
        "Charts show associations in the filtered historical records. They do not account for "
        "all clinical factors and must not be interpreted as causal or individual predictions."
    )

with records_tab:
    st.subheader("Filtered patient records")
    st.caption(
        "The dataset contains no names or direct identifiers. This preview omits marital status "
        "and race; the CSV export includes the filtered source fields."
    )
    preview_columns = [
        "Age",
        "Age Group",
        "6th Stage",
        "Grade",
        "Tumor Size",
        "Estrogen Status",
        "Progesterone Status",
        "Regional Node Examined",
        "Reginol Node Positive",
        "Lymph Node Ratio",
        "Survival Months",
        "Status",
    ]
    st.dataframe(
        patients[preview_columns].head(1000),
        width="stretch",
        hide_index=True,
        height=430,
    )
    st.download_button(
        "Download filtered patient records (CSV)",
        data=patients.drop(columns="Age Group").to_csv(index=False).encode("utf-8"),
        file_name="breast_cancer_filtered_records.csv",
        mime="text/csv",
    )
    if len(patients) > 1000:
        st.caption("Preview shows up to 1,000 records. The download contains all filtered records.")

st.divider()
st.caption(
    "Source: Breast_Cancer.csv. The dashboard removes exact duplicate rows and standardizes "
    "whitespace in labels. Dataset categories and recorded status reflect the source data."
)
