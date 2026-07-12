import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_dark"

def beautify_plotly(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(
            color="white",
            size=14
        ),
        title_font=dict(
            size=20,
            color="white"
        ),
        legend=dict(
            font=dict(
                color="white",
                size=14
            ),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            title_font=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="#374151",
            zeroline=False
        ),
        yaxis=dict(
            title_font=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="#374151",
            zeroline=False
        )
    )
    return fig

st.set_page_config(page_title="RetailIQ Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{
    background-color:#0E1117;
    color:#FAFAFA;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    padding-left:2rem;
    padding-right:2rem;
}

section[data-testid="stSidebar"]{
    background:#161B22;
    border-right:1px solid #30363d;
}

h1,h2,h3,h4,h5{
    color:#FFFFFF !important;
    font-weight:700;
}

[data-testid="stMetric"]{
    background:#161B22;
    border:1px solid #30363d;
    padding:18px;
    border-radius:15px;
    box-shadow:0 6px 18px rgba(0,0,0,.35);
}

[data-testid="stDataFrame"]{
    border-radius:12px;
    overflow:hidden;
}

.stDownloadButton{
    width:100%;
}
.stDownloadButton > button{
    width:100%;
    background:#2563EB !important;
    color:#FFFFFF !important;
    border:none !important;
    border-radius:12px !important;
    padding:0.75rem 1rem !important;
    font-size:16px !important;
    font-weight:600 !important;
    transition:0.3s;
}
.stDownloadButton > button:hover{
    background:#1D4ED8 !important;
    color:white !important;
    transform:translateY(-2px);
    box-shadow:0 4px 15px rgba(37,99,235,.35);
}
.stDownloadButton > button p{
    color:white !important;
    font-weight:600 !important;
}
.stDownloadButton > button span{
    color:white !important;
}

footer{
    visibility:hidden;
}

header{
    visibility:hidden;
}

hr{
    border:1px solid #30363d;
}
label{
    color:#F8FAFC !important;
    font-weight:600;
}

div[data-baseweb="select"] > div{
    background-color:#1F2937 !important;
    border:1px solid #374151 !important;
    color:white !important;
}

div[data-baseweb="select"] input{
    color:white !important;
}

div[data-baseweb="select"] input::placeholder{
    color:#9CA3AF !important;
}

div[role="listbox"]{
    background:#1F2937 !important;
    color:white !important;
}

div[role="option"]{
    background:#1F2937 !important;
    color:white !important;
}

div[role="option"]:hover{
    background:#374151 !important;
}

span[data-baseweb="tag"]{
    background:#2563EB !important;
    color:white !important;
}

span[data-baseweb="tag"] svg{
    fill:white !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    return df
df = load_data()

comparison = pd.read_csv("data/model_comparison.csv")
cluster_summary = pd.read_csv("data/cluster_summary.csv")
product_clusters = pd.read_csv("data/product_clusters.csv")

anomaly_results = pd.read_csv("data/anomaly_results.csv")
df["Isolation_Anomaly"] = anomaly_results["Isolation_Anomaly"]
df["Sales_ZScore"] = anomaly_results["Sales_ZScore"]
df["ZScore_Anomaly"] = anomaly_results["ZScore_Anomaly"]

st.title("RetailIQ - AI Powered Sales Forecasting Dashboard")
st.markdown("""
Welcome to the **RetailIQ Dashboard**, an interactive business intelligence platform for
analyzing retail sales, forecasting future demand, detecting anomalies, and segmenting products.

---
""")

st.sidebar.header("Filters")

regions = st.sidebar.multiselect(
    "Select Region",
    options=sorted(df["Region"].unique()),
    default=sorted(df["Region"].unique())
)

categories = st.sidebar.multiselect(
    "Select Category",
    options=sorted(df["Category"].unique()),
    default=sorted(df["Category"].unique())
)

segments = st.sidebar.multiselect(
    "Select Segment",
    options=sorted(df["Segment"].unique()),
    default=sorted(df["Segment"].unique())
)

filtered_df = df[
    (df["Region"].isin(regions)) &
    (df["Category"].isin(categories)) &
    (df["Segment"].isin(segments))
]

total_sales = filtered_df["Sales"].sum()
total_orders = filtered_df["Order ID"].nunique()
best_category = (
    filtered_df.groupby("Category")["Sales"]
    .sum()
    .idxmax()
)
best_model = comparison.loc[
    comparison["RMSE"].idxmin(),
    "Model"
]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💰 Total Sales",
        f"${total_sales:,.2f}"
    )

with col2:
    st.metric(
        "📦 Total Orders",
        total_orders
    )

with col3:
    st.metric(
        "🏆 Best Category",
        best_category
    )

with col4:
    st.metric(
        "🤖 Best Model",
        best_model
    )

monthly_sales = (
    filtered_df.groupby(
        pd.Grouper(
            key="Order Date",
            freq="MS"
        )
    )["Sales"]
    .sum()
    .reset_index()
)
fig = px.line(
    monthly_sales,
    x="Order Date",
    y="Sales",
    title="Monthly Sales Trend",
    markers=True
)
fig = beautify_plotly(fig)
st.plotly_chart(
    fig,
    use_container_width=True
)

left, right = st.columns(2)

with left:
    category_sales = (
        filtered_df.groupby("Category")["Sales"]
        .sum()
        .reset_index()
    )
    fig = px.bar(
        category_sales,
        x="Category",
        y="Sales",
        title="Sales by Category",
        text_auto=".2s"
    )
    fig = beautify_plotly(fig)
    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:
    region_sales = (
        filtered_df.groupby("Region")["Sales"]
        .sum()
        .reset_index()
    )
    fig = px.pie(
        region_sales,
        names="Region",
        values="Sales",
        title="Regional Sales Distribution"
    )
    fig = beautify_plotly(fig)
    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("---")
st.header("📈 Sales Forecasting")
st.subheader("Model Evaluation Metrics")
fig = go.Figure(
    data=[
        go.Table(
            columnwidth=[2.3, 1.2, 1.2],
            header=dict(
                values=["Model", "MAE", "RMSE"],
                fill_color="#2563EB",
                align="center",
                height=45,
                font=dict(
                    color="white",
                    size=18
                )
            ),
            cells=dict(
                values=[
                    comparison["Model"],
                    comparison["MAE"].map(lambda x: f"{x:,.2f}"),
                    comparison["RMSE"].map(lambda x: f"{x:,.2f}")
                ],
                fill_color="#1F2937",
                align="center",
                height=50,
                font=dict(
                    color="white",
                    size=17
                )
            )
        )
    ]
)
fig.update_layout(
    margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    height=250
)
st.plotly_chart(
    fig,
    use_container_width=True
)
best_model = comparison.loc[
    comparison["RMSE"].idxmin()
]
st.success(
    f"🏆 Best Forecasting Model: **{best_model['Model']}**"
)

st.subheader("Model Comparison (RMSE vs MAE)")
fig = px.bar(
    comparison,
    x="Model",
    y=["MAE","RMSE"],
    barmode="group",
    title="Forecasting Model Comparison"
)
fig = beautify_plotly(fig)
st.plotly_chart(
    fig,
    use_container_width=True
)

future_forecast = pd.read_csv("data/sarima_forecast.csv")
future_forecast["Month"] = pd.to_datetime(
    future_forecast["Month"]
)
st.subheader("Next 3 Months Forecast")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=monthly_sales["Order Date"],
        y=monthly_sales["Sales"],
        mode="lines+markers",
        name="Historical Sales"
    )
)
fig.add_trace(
    go.Scatter(
        x=future_forecast["Month"],
        y=future_forecast["Forecasted Sales"],
        mode="lines+markers",
        name="Forecast"
    )
)
fig.update_layout(
    title="Future Sales Forecast",
    xaxis_title="Month",
    yaxis_title="Sales"
    
)
fig = beautify_plotly(fig)
st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("---")
st.header("🚨 Sales Anomaly Detection")
df["Isolation_Anomaly"] = df["Isolation_Anomaly"].fillna("Normal")
anomaly_count = (df["Isolation_Anomaly"] == "Anomaly").sum()
anomaly_percent = (anomaly_count / len(df)) * 100
col1, col2 = st.columns(2)
with col1:
    st.metric("🚨 Detected Anomalies", anomaly_count)
with col2:
    st.metric("📊 Anomaly Percentage", f"{anomaly_percent:.2f}%")
fig = px.scatter(
    df,
    x=df.index,
    y="Sales",
    color="Isolation_Anomaly",
    color_discrete_map={
        "Normal": "#1f77b4",
        "Anomaly": "#d62728"
    },
    opacity=0.8,
    title="Isolation Forest - Sales Anomaly Detection"
)
fig = beautify_plotly(fig)
fig.update_layout(
    xaxis_title="Transaction Index",
    yaxis_title="Sales",
    legend_title="Transaction Type"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.header("📦 Product Segmentation")
st.subheader("Cluster Summary")
cols = st.columns(4)
for i, (_, row) in enumerate(cluster_summary.iterrows()):
    with cols[i]:
        st.markdown(f"""
<div style="
background:#1F2937;
padding:20px;
border-radius:16px;
border:1px solid #374151;
text-align:center;
">

<h2 style="color:white;">Cluster {int(row['Cluster'])}</h2>

<h1 style="color:#3B82F6;">
${row['Sales']:,.0f}
</h1>

<hr style="border:1px solid #374151;">

<div style="color:white;font-size:18px;margin-top:15px;">
🚚 <b>Shipping Days</b><br>
{row['Shipping_Days']:.2f}
</div>

<br>

<div style="color:white;font-size:18px;">
📦 <b>Orders</b><br>
{row['Order_Count']:.2f}
</div>

</div>
""", unsafe_allow_html=True
)
fig = px.scatter(
    product_clusters,
    x="Sales",
    y="Order_Count",
    color=product_clusters["Cluster"].astype(str),
    hover_name="Product Name",
    size="Sales",
    title="K-Means Product Segmentation"
)
fig.update_layout(
    xaxis_title="Total Sales",
    yaxis_title="Number of Orders",
    legend_title="Cluster"
)
fig = beautify_plotly(fig)
st.plotly_chart(fig, use_container_width=True)
cluster_counts = (
    product_clusters["Cluster"]
    .astype(str)
    .value_counts()
    .reset_index()
)
cluster_counts.columns = ["Cluster", "Count"]
fig = px.pie(
    cluster_counts,
    names="Cluster",
    values="Count",
    hole=0.45,
    title="Product Cluster Distribution"
)
fig = beautify_plotly(fig)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.header("📋 Dashboard Summary")
left, right = st.columns(2)
with left:
    st.success(f"""
### 🏆 Best Forecasting Model

**Model:** {best_model['Model']}

**RMSE:** {best_model['RMSE']:.2f}

**MAE:** {best_model['MAE']:.2f}
""")
with right:
    st.info(f"""
### 📈 Business Overview

💰 **Total Sales:** ${total_sales:,.2f}

📦 **Total Orders:** {total_orders}

🏆 **Best Category:** {best_category}

🚨 **Detected Anomalies:** {anomaly_count}

📊 **Anomaly Percentage:** {anomaly_percent:.2f}%
""")

st.markdown("---")
st.header("🏆 Top 10 Products by Sales")
top_products = (
    filtered_df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
fig = px.bar(
    top_products,
    x="Sales",
    y="Product Name",
    orientation="h",
    title="Top 10 Best Selling Products",
    text_auto=".2s"
)
fig.update_layout(yaxis={"categoryorder":"total ascending"})
fig = beautify_plotly(fig)
st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("---")
st.header("💡 Business Insights")
best_region = (
    filtered_df.groupby("Region")["Sales"]
    .sum()
    .idxmax()
)
best_category = (
    filtered_df.groupby("Category")["Sales"]
    .sum()
    .idxmax()
)
best_segment = (
    filtered_df.groupby("Segment")["Sales"]
    .sum()
    .idxmax()
)
st.success(f"""
### Key Findings

🏆 Highest Sales Region : **{best_region}**

📦 Best Performing Category : **{best_category}**

👥 Best Customer Segment : **{best_segment}**

🤖 Best Forecasting Model : **{best_model['Model']}**

🚨 Total Detected Anomalies : **{anomaly_count}**
""")

st.markdown("---")
st.header("📥 Download Forecast Results")
csv = future_forecast.to_csv(index=False)
st.download_button(
    label="📄 Download SARIMA Forecast",
    data=csv,
    file_name="sarima_forecast.csv",
    mime="text/csv"
)

comparison_csv = comparison.to_csv(index=False)
st.download_button(
    label="📊 Download Model Comparison",
    data=comparison_csv,
    file_name="model_comparison.csv",
    mime="text/csv"
)

st.markdown("---")

st.markdown("""
<div style='text-align:center; color:#9CA3AF; font-size:14px;'>

<strong>RetailIQ™ - AI Powered Sales Forecasting Dashboard</strong><br><br>

© 2026 RetailIQ. All Rights Reserved.<br>

Developed by <strong>Dhruv Rawat</strong>

</div>
""", unsafe_allow_html=True)