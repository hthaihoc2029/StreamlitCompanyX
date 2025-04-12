import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Tiêu đề Dashboard
st.title("📊 Material Purchase Planning Dashboard")

# Sidebar để bộ lọc
st.sidebar.header("⚙️ Settings")

# Hàm load dữ liệu
@st.cache_data
def load_data(file_path):
    xls = pd.ExcelFile(file_path)
    data = {
        "PurchaseOrderHeader": xls.parse("PurchaseOrderHeader"),
        "PurchaseOrderDetail": xls.parse("PurchaseOrderDetail"),
        "Vendor": xls.parse("Vendor"),
        "ShipMethod": xls.parse("ShipMethod"),
        "ProductVendor": xls.parse("ProductVendor"),
        "WorkOrder": xls.parse("WorkOrder"),
        "WorkOrderRouting": xls.parse("WorkOrderRouting"),
        "TransactionHistory": xls.parse("TransactionHistory")
    }
    return data

# Load dữ liệu
file_path = "./CompanyX.xlsx"  # Đường dẫn tương đối từ thư mục hiện tại
try:
    data = load_data(file_path)
except FileNotFoundError:
    st.error("File CompanyX.xlsx not found. Please place it in the same directory.")
    st.stop()

# Chuyển đổi định dạng ngày tháng
data["PurchaseOrderHeader"]["OrderDate"] = pd.to_datetime(data["PurchaseOrderHeader"]["OrderDate"])
data["PurchaseOrderHeader"]["ShipDate"] = pd.to_datetime(data["PurchaseOrderHeader"]["ShipDate"])
data["PurchaseOrderDetail"]["DueDate"] = pd.to_datetime(data["PurchaseOrderDetail"]["DueDate"])

# Tạo cột LateShipment
merged_data = data["PurchaseOrderDetail"].merge(
    data["PurchaseOrderHeader"][["PurchaseOrderID", "ShipDate", "VendorID", "ShipMethodID", "SubTotal", "TaxAmt", "Freight"]],
    on="PurchaseOrderID",
    how="left"
)
merged_data["LateShipment"] = (merged_data["ShipDate"] > merged_data["DueDate"]).astype(int)

# Merge với Vendor, ShipMethod, ProductVendor
merged_data = merged_data.merge(
    data["Vendor"][["BusinessEntityID", "Name", "CreditRating"]],
    left_on="VendorID",
    right_on="BusinessEntityID",
    how="left"
).rename(columns={"Name": "VendorName"})
merged_data = merged_data.merge(
    data["ShipMethod"][["ShipMethodID", "Name"]],
    on="ShipMethodID",
    how="left"
).rename(columns={"Name": "ShipMethodName"})
merged_data = merged_data.merge(
    data["ProductVendor"][["ProductID", "BusinessEntityID", "AverageLeadTime"]],
    left_on=["ProductID", "VendorID"],
    right_on=["ProductID", "BusinessEntityID"],
    how="left"
)

# Bộ lọc trong sidebar
st.sidebar.subheader("Filters")
selected_vendors = st.sidebar.multiselect("Select Vendors", options=merged_data["VendorName"].unique(), default=merged_data["VendorName"].unique())
date_range = st.sidebar.date_input("Select Date Range", 
                                   [data["PurchaseOrderHeader"]["OrderDate"].min(), data["PurchaseOrderHeader"]["OrderDate"].max()])

# Lọc dữ liệu
filtered_data = merged_data[
    (merged_data["VendorName"].isin(selected_vendors)) &
    (merged_data["ShipDate"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
]
filtered_header = data["PurchaseOrderHeader"][
    (data["PurchaseOrderHeader"]["OrderDate"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
]

# --- KPI Tổng quan ---
st.header("🧮 Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Orders", len(filtered_header["PurchaseOrderID"].unique()))
with col2:
    st.metric("Total Cost", f"${filtered_header['TotalDue'].sum():,.2f}")
with col3:
    st.metric("Unique Vendors", len(filtered_header["VendorID"].unique()))
with col4:
    st.metric("Received Quantity", f"{filtered_data['ReceivedQty'].sum():,.0f}")

# --- Biểu đồ ---
st.header("📈 Visualizations")

# 1. Xu hướng tổng chi phí theo tháng
st.subheader("Total Cost Trend by Month")
cost_by_month = filtered_header.groupby(filtered_header["OrderDate"].dt.to_period("M"))["TotalDue"].sum().reset_index()
cost_by_month["OrderDate"] = cost_by_month["OrderDate"].astype(str)
fig_cost = px.line(cost_by_month, x="OrderDate", y="TotalDue", title="Total Cost Over Time")
fig_cost.update_layout(xaxis_title="Month", yaxis_title="Total Cost ($)")
st.plotly_chart(fig_cost, use_container_width=True)

# 2. Top Vendor theo số đơn hàng
st.subheader("Top Vendors by Order Count")
vendor_orders = filtered_header.merge(data["Vendor"][["BusinessEntityID", "Name"]], 
                                     left_on="VendorID", right_on="BusinessEntityID")
vendor_counts = vendor_orders["Name"].value_counts().reset_index()
vendor_counts.columns = ["Vendor", "OrderCount"]
fig_vendor = px.bar(vendor_counts.head(10), x="OrderCount", y="Vendor", orientation="h", 
                    title="Top 10 Vendors by Order Count")
fig_vendor.update_layout(xaxis_title="Number of Orders", yaxis_title="Vendor")
st.plotly_chart(fig_vendor, use_container_width=True)

# 3. Tỷ lệ phương thức giao hàng
st.subheader("Shipping Method Distribution")
ship_counts = filtered_header.merge(data["ShipMethod"][["ShipMethodID", "Name"]], 
                                   on="ShipMethodID")["Name"].value_counts().reset_index()
ship_counts.columns = ["ShipMethod", "Count"]
fig_ship = px.pie(ship_counts, names="ShipMethod", values="Count", 
                  title="Shipping Method Distribution")
st.plotly_chart(fig_ship, use_container_width=True)

# --- Phân tích sản xuất ---
st.header("⚙️ Production Analysis")
col1, col2 = st.columns(2)

# 4. ActualResourceHrs theo ProductID
with col1:
    st.subheader("Resource Hours by Product")
    resource_hrs = data["WorkOrderRouting"].groupby("ProductID")["ActualResourceHrs"].sum().reset_index()
    fig_hrs = px.bar(resource_hrs, x="ProductID", y="ActualResourceHrs", 
                     title="Total Resource Hours by Product")
    fig_hrs.update_layout(xaxis_title="Product ID", yaxis_title="Hours")
    st.plotly_chart(fig_hrs, use_container_width=True)

# 5. ScrappedQty theo ProductID
with col2:
    st.subheader("Scrapped Quantity by Product")
    scrapped_qty = data["WorkOrder"].groupby("ProductID")["ScrappedQty"].sum().reset_index()
    fig_scrap = px.bar(scrapped_qty, x="ProductID", y="ScrappedQty", 
                       title="Scrapped Quantity by Product")
    fig_scrap.update_layout(xaxis_title="Product ID", yaxis_title="Scrapped Qty")
    st.plotly_chart(fig_scrap, use_container_width=True)

# --- Các biểu đồ bổ sung ---
st.header("🔍 Additional Insights")

# 6. Late Shipment Ratio
st.subheader("Late Shipment Ratio")
late_counts = filtered_data["LateShipment"].value_counts().reset_index()
late_counts.columns = ["LateShipment", "Count"]
late_counts["LateShipment"] = late_counts["LateShipment"].map({1: "Late", 0: "On Time"})
fig_late = px.pie(late_counts, names="LateShipment", values="Count", 
                  title="Late Shipment Distribution")
st.plotly_chart(fig_late, use_container_width=True)

# 7. Rejected Orders
st.subheader("Rejected Orders by Product")
rejected_qty = filtered_data.groupby("ProductID")["RejectedQty"].sum().reset_index()
fig_reject = px.bar(rejected_qty[rejected_qty["RejectedQty"] > 0], x="ProductID", y="RejectedQty", 
                    title="Rejected Quantity by Product")
fig_reject.update_layout(xaxis_title="Product ID", yaxis_title="Rejected Qty")
st.plotly_chart(fig_reject, use_container_width=True)

# 8. High-Cost Orders
st.subheader("High-Cost Orders")
threshold = filtered_header["TotalDue"].quantile(0.75)
high_cost = filtered_header[filtered_header["TotalDue"] > threshold].groupby(
    filtered_header["OrderDate"].dt.to_period("M"))["TotalDue"].sum().reset_index()
high_cost["OrderDate"] = high_cost["OrderDate"].astype(str)
fig_high_cost = px.line(high_cost, x="OrderDate", y="TotalDue", 
                        title="High-Cost Orders Over Time")
fig_high_cost.update_layout(xaxis_title="Month", yaxis_title="Total Cost ($)")
st.plotly_chart(fig_high_cost, use_container_width=True)

# 9. Vendor Performance
st.subheader("Vendor Performance")
vendor_perf = filtered_data.groupby("VendorName").agg({
    "LateShipment": "mean",
    "RejectedQty": "mean",
    "SubTotal": "sum"
}).reset_index()
vendor_perf["PerformanceScore"] = (1 - vendor_perf["LateShipment"]) * 0.4 + \
                                 (1 - vendor_perf["RejectedQty"] / vendor_perf["RejectedQty"].max()) * 0.4 + \
                                 (1 - vendor_perf["SubTotal"] / vendor_perf["SubTotal"].max()) * 0.2
fig_perf = px.bar(vendor_perf, x="PerformanceScore", y="VendorName", orientation="h", 
                  title="Vendor Performance Score")
fig_perf.update_layout(xaxis_title="Performance Score", yaxis_title="Vendor")
st.plotly_chart(fig_perf, use_container_width=True)

# 10. Inventory Levels by Product
st.subheader("Inventory Levels by Product")
inventory = filtered_data.groupby("ProductID")["StockedQty"].sum().reset_index()
inventory["InventoryLevel"] = inventory["StockedQty"] - data["TransactionHistory"][
    data["TransactionHistory"]["TransactionType"] == "W"
].groupby("ProductID")["Quantity"].sum().reindex(inventory["ProductID"], fill_value=0).values
fig_inventory = px.bar(inventory, x="ProductID", y="InventoryLevel", 
                       title="Inventory Levels by Product")
fig_inventory.update_layout(xaxis_title="Product ID", yaxis_title="Inventory Level")
st.plotly_chart(fig_inventory, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Data: CompanyX.xlsx")