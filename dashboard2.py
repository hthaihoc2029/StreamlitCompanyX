import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- Tiêu đề Dashboard ---
st.set_page_config(page_title="Dashboard Kế hoạch Mua Vật tư", layout="wide")
st.title("📊 Dashboard Kế hoạch Mua Vật tư")

# --- Sidebar để bộ lọc ---
st.sidebar.header("⚙️ Cài đặt")
@st.cache_data
def load_data(path):
    xls = pd.ExcelFile(path)
    return {
        "POH": xls.parse("PurchaseOrderHeader"),
        "POD": xls.parse("PurchaseOrderDetail"),
        "Vendor": xls.parse("Vendor"),
        "Ship": xls.parse("ShipMethod"),
        "PV": xls.parse("ProductVendor"),
        "WO": xls.parse("WorkOrder"),
        "WOR": xls.parse("WorkOrderRouting"),
        "TH": xls.parse("TransactionHistory")
    }

file_path = "CompanyX.xlsx"
try:
    data = load_data(file_path)
except FileNotFoundError:
    st.error("❌ Không tìm thấy file CompanyX.xlsx. Vui lòng đặt trong cùng thư mục.")
    st.stop()

# --- Chuẩn hóa ngày tháng ---
data["POH"]["OrderDate"] = pd.to_datetime(data["POH"]["OrderDate"])
data["POH"]["ShipDate"]  = pd.to_datetime(data["POH"]["ShipDate"])
data["POD"]["DueDate"]   = pd.to_datetime(data["POD"]["DueDate"])

# --- Tính LateShipment ---
df = data["POD"].merge(
    data["POH"][["PurchaseOrderID","ShipDate","VendorID","ShipMethodID","SubTotal","TaxAmt","Freight","TotalDue"]],
    on="PurchaseOrderID", how="left"
)
df["LateShipment"] = (df["ShipDate"] > df["DueDate"]).astype(int)

# --- Merge bổ sung ---
df = df.merge(
    data["Vendor"][["BusinessEntityID","Name"]], left_on="VendorID", right_on="BusinessEntityID", how="left"
).rename(columns={"Name":"Nhà cung cấp"})
df = df.merge(
    data["Ship"][["ShipMethodID","Name"]], on="ShipMethodID", how="left"
).rename(columns={"Name":"Phương thức giao"})
df = df.merge(
    data["PV"][["ProductID","BusinessEntityID","AverageLeadTime"]], 
    left_on=["ProductID","VendorID"], right_on=["ProductID","BusinessEntityID"], how="left"
)

# --- Bộ lọc ---
st.sidebar.subheader("📋 Bộ lọc")
vendors = st.sidebar.multiselect(
    "Chọn Nhà cung cấp", options=df["Nhà cung cấp"].unique(), 
    default=df["Nhà cung cấp"].unique()
)
dr = st.sidebar.date_input(
    "Chọn khoảng thời gian",
    [data["POH"]["OrderDate"].min(), data["POH"]["OrderDate"].max()]
)

mask = (
    df["Nhà cung cấp"].isin(vendors) &
    df["ShipDate"].between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1]))
)
df_f = df[mask]
poh_f = data["POH"][
    data["POH"]["OrderDate"].between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1]))
]

# --- KPI Tổng quan ---
st.header("🧮 Tổng quan")
c1, c2, c3, c4 = st.columns(4)
c1.metric("📝 Tổng số đơn hàng", f"{poh_f['PurchaseOrderID'].nunique():,}")
c2.metric("💰 Tổng chi phí", f"${poh_f['TotalDue'].sum():,.2f}")
c3.metric("🏢 Số nhà cung cấp", f"{poh_f['VendorID'].nunique():,}")
c4.metric("📦 Tổng số lượng nhận", f"{df_f['ReceivedQty'].sum():,.0f}")

st.markdown("---")

# --- Phần 1: Chi phí & Đơn hàng ---
st.subheader("📈 Xu hướng Chi phí & Đơn hàng")

# 1. Xu hướng tổng chi phí theo tháng
cost_m = poh_f.groupby(poh_f["OrderDate"].dt.to_period("M"))["TotalDue"].sum().reset_index()
cost_m["OrderDate"] = cost_m["OrderDate"].astype(str)
fig1 = px.line(cost_m, x="OrderDate", y="TotalDue", markers=True)
fig1.update_layout(title="💰 Xu hướng Tổng chi phí theo Tháng",
                   xaxis_title="Tháng", yaxis_title="Tổng chi phí ($)")
st.plotly_chart(fig1, use_container_width=True)
st.caption("Biểu đồ thể hiện sự thay đổi tổng chi phí mua hàng qua các tháng.")

# 2. Số đơn hàng theo tháng (heatmap)
orders_m = poh_f.groupby([poh_f["OrderDate"].dt.to_period("M"), "VendorID"])["PurchaseOrderID"]\
    .nunique().reset_index().rename(columns={"OrderDate":"Tháng","PurchaseOrderID":"Số đơn"})
heat = orders_m.pivot(index="VendorID", columns="Tháng", values="Số đơn").fillna(0)
fig_heat = go.Figure(data=go.Heatmap(
    z=heat.values, x=heat.columns.astype(str), y=heat.index,
    colorscale="Blues"
))
fig_heat.update_layout(title="🌡️ Số đơn hàng theo Nhà cung cấp & Tháng",
                       xaxis_title="Tháng", yaxis_title="VendorID")
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# --- Phần 2: Nhà cung cấp & Giao hàng ---
st.subheader("🚚 Nhà cung cấp & Giao hàng")

# 3. Top 10 Vendor theo số đơn
vc = df_f["Nhà cung cấp"].value_counts().nlargest(10).reset_index()
vc.columns = ["Nhà cung cấp","Số đơn"]
fig2 = px.bar(vc, x="Số đơn", y="Nhà cung cấp", orientation="h",
              title="🏆 Top 10 Nhà cung cấp theo Số đơn")
fig2.update_layout(xaxis_title="Số đơn", yaxis_title="Nhà cung cấp")
st.plotly_chart(fig2, use_container_width=True)
st.caption("Nhà cung cấp có khối lượng đơn hàng cao nhất.")

# 4. Tỷ lệ phương thức giao hàng
sc = df_f["Phương thức giao"].value_counts().reset_index()
sc.columns = ["Phương thức","Số đơn"]
fig3 = px.pie(sc, names="Phương thức", values="Số đơn",
              title="🚛 Phân bố Phương thức Giao hàng")
st.plotly_chart(fig3, use_container_width=True)
st.caption("Tỷ lệ sử dụng các phương thức vận chuyển khác nhau.")

st.markdown("---")

# --- Phần 3: Sản xuất ---
st.subheader("🏭 Phân tích Sản xuất")
r1, r2 = st.columns(2)

with r1:
    hrs = data["WOR"].groupby("ProductID")["ActualResourceHrs"].sum().reset_index()
    fig4 = px.bar(hrs, x="ProductID", y="ActualResourceHrs",
                  title="⏱️ Tổng Giờ Sản xuất theo Sản phẩm")
    fig4.update_layout(xaxis_title="Product ID", yaxis_title="Giờ")
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Tổng số giờ tài nguyên sử dụng cho mỗi sản phẩm.")

with r2:
    scrap = data["WO"].groupby("ProductID")["ScrappedQty"].sum().reset_index()
    fig5 = px.bar(scrap, x="ProductID", y="ScrappedQty",
                  title="🚮 Số lượng Hàng hỏng theo Sản phẩm")
    fig5.update_layout(xaxis_title="Product ID", yaxis_title="Số lượng hỏng")
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Sản phẩm nào có tỷ lệ phế phẩm cao.")

st.markdown("---")

# --- Phần 4: Chất lượng & Rủi ro ---
st.subheader("❗ Chất lượng & Rủi ro")
c1, c2, c3 = st.columns(3)

with c1:
    late = df_f["LateShipment"].value_counts().rename({0:"Đúng hạn",1:"Trễ"}).reset_index()
    late.columns = ["Trạng thái","Số đơn"]
    fig6 = px.pie(late, names="Trạng thái", values="Số đơn", title="⏳ Tỷ lệ Giao hàng trễ")
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("Tỷ lệ đơn hàng giao trễ so với đúng hạn.")

with c2:
    rej = df_f.groupby("ProductID")["RejectedQty"].sum().reset_index()
    rej = rej[rej["RejectedQty"]>0]
    fig7 = px.bar(rej, x="ProductID", y="RejectedQty",
                  title="❌ Số lượng Bị từ chối theo Sản phẩm")
    st.plotly_chart(fig7, use_container_width=True)
    st.caption("Sản phẩm nào bị trả lại do lỗi chất lượng.")

with c3:
    th = poh_f["TotalDue"].quantile(0.75)
    hc = poh_f[poh_f["TotalDue"]>th].groupby(poh_f["OrderDate"].dt.to_period("M"))["TotalDue"].sum().reset_index()
    hc["OrderDate"] = hc["OrderDate"].astype(str)
    fig8 = px.line(hc, x="OrderDate", y="TotalDue",
                   title="💸 Chi phí Đơn hàng Cao (Q3) theo Tháng")
    st.plotly_chart(fig8, use_container_width=True)
    st.caption("Theo dõi chi phí các đơn hàng nằm trong top 25% cao nhất.")

st.markdown("---")

# --- Phần 5: Tồn kho & Hiệu suất Vendor ---
st.subheader("📦 Tồn kho & Hiệu suất Nhà cung cấp")
d1, d2 = st.columns(2)

with d1:
    inv = df_f.groupby("ProductID")["StockedQty"].sum().reset_index()
    used = data["TH"].query("TransactionType=='W'")\
        .groupby("ProductID")["Quantity"].sum().reindex(inv["ProductID"], fill_value=0).values
    inv["Tồn kho thực"] = inv["StockedQty"] - used
    fig9 = px.bar(inv, x="ProductID", y="Tồn kho thực",
                  title="📦 Tồn kho Thực tế theo Sản phẩm")
    st.plotly_chart(fig9, use_container_width=True)
    st.caption("Tồn kho khả dụng = StockedQty − Xuất kho để sản xuất.")

with d2:
    vp = df_f.groupby("Nhà cung cấp").agg({
        "LateShipment":"mean","RejectedQty":"mean","SubTotal":"sum"
    }).reset_index()
    vp["Điểm hiệu suất"] = (
        (1-vp["LateShipment"])*0.4 +
        (1 - vp["RejectedQty"]/vp["RejectedQty"].max())*0.4 +
        (1 - vp["SubTotal"]/vp["SubTotal"].max())*0.2
    )
    fig10 = px.bar(vp, x="Điểm hiệu suất", y="Nhà cung cấp", orientation="h",
                   title="⭐ Điểm Hiệu suất Nhà cung cấp")
    st.plotly_chart(fig10, use_container_width=True)
    st.caption("Kết hợp trễ hàng, từ chối, chi phí để đánh giá tổng quan.")

st.markdown("---")

# --- Phần 6: Biểu đồ Mở rộng ---
st.subheader("🆕 Biểu đồ Mở rộng")

# Scatter: Chi phí vs Tỉ lệ trễ hàng theo vendor
scat = vp.copy()
scat["Tỉ lệ trễ"] = vp["LateShipment"]
fig11 = px.scatter(scat, x="SubTotal", y="Tỉ lệ trễ", color="Nhà cung cấp",
                   title="🔍 Chi phí vs Tỉ lệ Giao hàng trễ")
fig11.update_layout(xaxis_title="Tổng chi phí Vendor", yaxis_title="Tỉ lệ trễ (%)")
st.plotly_chart(fig11, use_container_width=True)
st.caption("Phân tích mối quan hệ giữa chi phí và tỉ lệ giao trễ.")

# AverageLeadTime theo vendor
alt = df_f.groupby("Nhà cung cấp")["AverageLeadTime"].mean().reset_index()
fig12 = px.bar(alt, x="AverageLeadTime", y="Nhà cung cấp", orientation="h",
               title="⏱️ Thời gian giao trung bình theo Nhà cung cấp")
st.plotly_chart(fig12, use_container_width=True)
st.caption("Vendor nào có thời gian giao hàng trung bình ngắn nhất.")

# --- Phần 7: Theo dõi tồn kho theo thời gian thực ---
st.subheader("📡 Theo dõi tồn kho theo thời gian thực")

# Lấy danh sách sản phẩm có trong TransactionHistory
product_ids = data["TH"]["ProductID"].dropna().unique()
selected_prods = st.multiselect(
    "Chọn Sản phẩm để theo dõi tồn kho", 
    options=product_ids, 
    default=product_ids[:3]
)

if selected_prods:
    # Lấy lịch sử giao dịch của các sản phẩm được chọn
    th = data["TH"].copy()
    # Chuyển đổi TransactionDate với format hỗ trợ millisecond
    th["TransactionDate"] = pd.to_datetime(th["TransactionDate"], errors='coerce', format='mixed')
    th = th.dropna(subset=["TransactionDate"])  # Loại bỏ các giá trị NaT
    th["TransactionDate"] = th["TransactionDate"].dt.date
    th = th[th["ProductID"].isin(selected_prods)]
    
    # Chuyển TransactionType: P (nhập) +, W (xuất) -
    th["Delta"] = th.apply(lambda r: -r["Quantity"] if r["TransactionType"]=="W" else +r["Quantity"], axis=1)
    
    # Tính tồn kho tích lũy theo ngày và theo sản phẩm
    th_group = th.groupby(["ProductID", "TransactionDate"])["Delta"].sum().reset_index()
    th_group = th_group.sort_values(["ProductID", "TransactionDate"])
    th_group["Inventory"] = th_group.groupby("ProductID")["Delta"].cumsum()
    
    # Vẽ line chart cho mỗi sản phẩm
    fig_rt = px.line(
        th_group, 
        x="TransactionDate", 
        y="Inventory", 
        color="ProductID",
        title="📈 Tồn kho theo thời gian thực"
    )
    fig_rt.update_layout(
        xaxis_title="Ngày",
        yaxis_title="Tồn kho (số lượng)"
    )
    st.plotly_chart(fig_rt, use_container_width=True)
    st.caption("Biểu đồ thể hiện mức tồn kho tích lũy theo ngày cho sản phẩm được chọn.")
else:
    st.info("Chọn ít nhất một sản phẩm để xem tồn kho thời gian thực.")
    
# --- Phần 8: Phân tích Thông minh bằng AI ---
st.subheader("🧠 Phân tích Thông minh bằng AI")

# Cài đặt thư viện bổ sung nếu cần
try:
    from prophet import Prophet
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError:
    st.warning("⚠️ Cần cài đặt thư viện prophet và scikit-learn. Chạy: pip install prophet scikit-learn")
    st.stop()

# 8.1 Dự đoán tồn kho
st.markdown("### 📈 Dự đoán Tồn kho trong 30 ngày tới")
ai_prods = st.multiselect(
    "Chọn Sản phẩm để dự đoán tồn kho", 
    options=product_ids, 
    default=product_ids[:2],
    key="ai_prods"
)

if ai_prods:
    # Chuẩn bị dữ liệu cho Prophet
    th = data["TH"].copy()
    th["TransactionDate"] = pd.to_datetime(th["TransactionDate"], errors='coerce', format='mixed')
    th = th.dropna(subset=["TransactionDate"])
    th["TransactionDate"] = th["TransactionDate"].dt.date
    th = th[th["ProductID"].isin(ai_prods)]
    th["Delta"] = th.apply(lambda r: r["Quantity"] if r["TransactionType"]=="S" else -r["Quantity"], axis=1)
    
    # Tính tồn kho tích lũy theo ngày
    th_group = th.groupby(["ProductID", "TransactionDate"])["Delta"].sum().reset_index()
    th_group = th_group.sort_values(["ProductID", "TransactionDate"])
    th_group["Inventory"] = th_group.groupby("ProductID")["Delta"].cumsum()
    
    # Dự đoán cho từng sản phẩm
    figs = []
    for prod in ai_prods:
        df_prophet = th_group[th_group["ProductID"] == prod][["TransactionDate", "Inventory"]]
        df_prophet = df_prophet.rename(columns={"TransactionDate": "ds", "Inventory": "y"})
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
        
        # Khởi tạo và huấn luyện mô hình Prophet
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        model.fit(df_prophet)
        
        # Dự đoán 30 ngày tới
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        # Vẽ biểu đồ
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=df_prophet["ds"], y=df_prophet["y"], mode="lines", name=f"Historical (Product {prod})"
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat"], mode="lines", name=f"Forecast (Product {prod})",
            line=dict(dash="dash")
        ))
        fig_forecast.update_layout(
            title=f"Dự đoán Tồn kho cho Product {prod}",
            xaxis_title="Ngày", yaxis_title="Tồn kho",
            legend=dict(x=0, y=1)
        )
        figs.append(fig_forecast)
    
    # Hiển thị các biểu đồ dự đoán
    for fig in figs:
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Dự đoán tồn kho dựa trên mô hình Prophet, sử dụng dữ liệu lịch sử giao dịch.")

# 8.2 Phân loại Nhà cung cấp
st.markdown("### 🏅 Đánh giá Nhà cung cấp bằng AI")
if st.button("Phân loại Nhà cung cấp", key="classify_vendors"):
    # Chuẩn bị dữ liệu
    vp = df_f.groupby("Nhà cung cấp").agg({
        "LateShipment": "mean",
        "RejectedQty": "mean",
        "SubTotal": "sum"
    }).reset_index()
    
    # Tạo nhãn giả (ví dụ: nhà cung cấp tốt nếu LateShipment < 0.2 và RejectedQty thấp)
    vp["Label"] = (vp["LateShipment"] < 0.2) & (vp["RejectedQty"] < vp["RejectedQty"].quantile(0.5))
    vp["Label"] = vp["Label"].map({True: "Tốt", False: "Cần cải thiện"})
    
    # Chuẩn bị features và labels
    X = vp[["LateShipment", "RejectedQty", "SubTotal"]]
    y = vp["Label"]
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Huấn luyện mô hình RandomForest
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_scaled, y)
    
    # Dự đoán và hiển thị kết quả
    vp["Predicted_Label"] = clf.predict(X_scaled)
    fig_clf = px.scatter(
        vp, x="LateShipment", y="RejectedQty", color="Predicted_Label",
        size="SubTotal", hover_data=["Nhà cung cấp"],
        title="Phân loại Nhà cung cấp: Tốt vs Cần cải thiện"
    )
    fig_clf.update_layout(
        xaxis_title="Tỷ lệ Giao trễ", yaxis_title="Số lượng Từ chối",
        legend_title="Dự đoán"
    )
    st.plotly_chart(fig_clf, use_container_width=True)
    st.caption("Phân loại nhà cung cấp bằng RandomForest dựa trên tỷ lệ giao trễ, từ chối, và chi phí.")
    
    # Hiển thị bảng kết quả
    st.write("Kết quả phân loại Nhà cung cấp:")
    st.dataframe(vp[["Nhà cung cấp", "LateShipment", "RejectedQty", "SubTotal", "Predicted_Label"]])

# 8.3 Đề xuất Tối ưu hóa
st.markdown("### 💡 Đề xuất Tối ưu hóa")
if st.button("Tạo Đề xuất", key="recommendations"):
    recommendations = []
    
    # Đề xuất giảm tồn kho dư thừa
    inv = df_f.groupby("ProductID")["StockedQty"].sum().reset_index()
    used = data["TH"].query("TransactionType=='W'")\
        .groupby("ProductID")["Quantity"].sum().reindex(inv["ProductID"], fill_value=0).values
    inv["Tồn kho thực"] = inv["StockedQty"] - used
    high_inv = inv[inv["Tồn kho thực"] > inv["Tồn kho thực"].quantile(0.75)]
    for _, row in high_inv.iterrows():
        recommendations.append(
            f"Giảm đặt hàng cho Product {row['ProductID']}: Tồn kho thực ({row['Tồn kho thực']:.0f}) vượt mức trung bình."
        )
    
    # Đề xuất chuyển nhà cung cấp kém hiệu quả
    vp = df_f.groupby("Nhà cung cấp").agg({
        "LateShipment": "mean",
        "RejectedQty": "mean"
    }).reset_index()
    low_perf = vp[(vp["LateShipment"] > 0.3) | (vp["RejectedQty"] > vp["RejectedQty"].quantile(0.75))]
    for _, row in low_perf.iterrows():
        recommendations.append(
            f"Xem xét thay thế Nhà cung cấp {row['Nhà cung cấp']}: Tỷ lệ giao trễ ({row['LateShipment']:.2%}), "
            f"Số lượng từ chối ({row['RejectedQty']:.0f})."
        )
    
    # Hiển thị đề xuất
    if recommendations:
        st.write("**Đề xuất tối ưu hóa chuỗi cung ứng:**")
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.info("Không có đề xuất nào vào lúc này.")
    st.caption("Các đề xuất được tạo dựa trên phân tích tồn kho và hiệu suất nhà cung cấp.")

# Footer
st.markdown("---")
st.markdown("🚀 Built with Streamlit - Nhóm 2 - HTTTQL | Dữ liệu: CompanyX.xlsx")