import random
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
import matplotlib.pyplot as plt
import streamlit as st

# Bakery data options
products_bakery = ["Chocolate Cake", "Vanilla Cupcake", "Fruit Tart", "Brownie", "Muffin", "Croissant", "Cheese Pastry", "Sourdough"]
categories_bakery = ["Cakes", "Cupcakes", "Tarts", "Baked Goods", "Pastries", "Pastries", "Baked Goods", "Cakes"]
customer_segments_bakery = ["Individual", "Family", "Business"]
genders_bakery = ["Male", "Female", "Other"]
payment_methods_bakery = ["Credit Card", "Debit Card", "Cash", "UPI"]

# Essential columns for the home bakery dataset (including 'When the Product Was Bought')
essential_columns = [
    "Product Name", "Category", "Ingredients", "Quantity Sold", "Price", "Cost per Unit", 
    "Total Revenue", "Discount Applied", "Customer Segment", "Order Date", "Expiration Date", 
    "Shelf Life (Days)", "Customer Age", "Customer Gender", "Payment Method", "Order Type", 
    "Wastage Quantity", "Review Rating", "Packaging Type", "When the Product Was Bought"
]

# Generate the refined home bakery dataset with the essential columns
data_bakery_refined = []

for _ in range(5000):
    product = random.choice(products_bakery)
    category = random.choice(categories_bakery)  # Select category independently
    ingredients = ", ".join(random.sample(["Flour", "Sugar", "Butter", "Eggs", "Milk", "Baking Powder", "Cocoa"], 3))
    quantity_sold = random.randint(1, 3)
    price = random.randint(50, 300)  # Random price between 50 and 300 INR for bakery items
    cost_per_unit = round(price * 0.6, 2)  # Cost is assumed to be 60% of the price
    total_revenue = quantity_sold * price
    discount_applied = random.choice([True, False])
    customer_segment = random.choice(customer_segments_bakery)
    
    # Generate dates
    order_date = datetime.now() - timedelta(days=random.randint(1, 15))  # Random order in the last 15 days
    exp_date = order_date + timedelta(days=random.randint(3, 6))  # Expiration date 3-6 days after order
    shelf_life = (exp_date - order_date).days
    customer_age = random.randint(18, 60)  # Random age between 18 and 60
    gender = random.choice(genders_bakery)
    payment_method = random.choice(payment_methods_bakery)
    order_type = random.choice(["Single", "Bulk"])
    wastage_quantity = random.randint(0, 2)  # Wastage between 0 and 2 units
    review_rating = round(random.uniform(1, 5), 1)  # Random rating between 1 and 5
    packaging_type = random.choice(["Paper Box", "Plastic Wrap", "Cloth Bag"])

    # Add the "When the Product Was Bought" (same as Order Date)
    when_product_bought = order_date.strftime('%Y-%m-%d')

    # Add row to dataset
    data_bakery_refined.append([product, category, ingredients, quantity_sold, price, cost_per_unit, total_revenue, 
                                discount_applied, customer_segment, order_date.strftime('%Y-%m-%d'), exp_date.strftime('%Y-%m-%d'), 
                                shelf_life, customer_age, gender, payment_method, order_type, wastage_quantity, 
                                review_rating, packaging_type, when_product_bought])

# Create the refined dataframe for home bakery
df_bakery_refined = pd.DataFrame(data_bakery_refined, columns=essential_columns)

# Save the refined dataset to CSV
df_bakery_refined.to_csv("refined_home_bakery_sales_data.csv", index=False)

# ----------------------
# Home Bakery Forecast Dashboard
# ----------------------

st.set_page_config(page_title="Home Bakery Sales & Inventory Forecast", layout="wide")
st.title("🍰 Home Bakery Sales & Inventory Forecasting")

# Upload CSV
uploaded = st.file_uploader("Upload Home Bakery CSV", type="csv")
if not uploaded:
    st.info("Please upload a CSV with columns: Order Date, Category, Quantity Sold, Total Revenue, Wastage Quantity.")
    st.stop()

# Load Data
df = pd.read_csv(uploaded, parse_dates=["Order Date", "Expiration Date", "When the Product Was Bought"])

# Ensure required columns
required = ["Order Date", "Category", "Quantity Sold", "Total Revenue", "Wastage Quantity"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {', '.join(missing)}")
    st.stop()

# Derived fields
# Gross Margin per order: Total Revenue - (Quantity Sold * Cost per Unit) if Cost per Unit present
if "Cost per Unit" in df.columns:
    df["Cost"] = df["Quantity Sold"] * df["Cost per Unit"]
    df["Gross Margin"] = df["Total Revenue"] - df["Cost"]
else:
    df["Gross Margin"] = df["Total Revenue"]

# Aggregate daily revenue
daily_rev = df.groupby("Order Date").agg({"Total Revenue": "sum"}).reset_index()
renamed_rev = daily_rev.rename(columns={"Order Date": "ds", "Total Revenue": "y"})

# Forecast total revenue next 6 months
st.header("📈 Overall Sales Forecast")
model_rev = Prophet(daily_seasonality=True)
model_rev.fit(renamed_rev)
future_rev = model_rev.make_future_dataframe(periods=180)
forecast_rev = model_rev.predict(future_rev)

fig1 = model_rev.plot(forecast_rev)
st.pyplot(fig1)

# Summary metrics
next_30 = forecast_rev[forecast_rev['ds'] > pd.Timestamp.today()].head(30)['yhat'].sum()
next_180 = forecast_rev[forecast_rev['ds'] > pd.Timestamp.today()].head(180)['yhat'].sum()
st.subheader("Sales Forecast Summary")
st.write(f"Forecast next 30 days revenue: ₹{next_30:,.0f}")
st.write(f"Forecast next 6 months revenue: ₹{next_180:,.0f}")

# Forecast quantity by category
st.header("📊 Category-wise Quantity Forecast (Next 3 Months)")
cat_forecasts = []
for cat, group in df.groupby('Category'):
    daily_qty = group.groupby('Order Date').agg({'Quantity Sold': 'sum'}).reset_index()
    if len(daily_qty) < 30:
        continue
    df_qty = daily_qty.rename(columns={'Order Date': 'ds', 'Quantity Sold': 'y'})
    model_qty = Prophet(daily_seasonality=True)
    model_qty.fit(df_qty)
    future_qty = model_qty.make_future_dataframe(periods=90)
    forecast_qty = model_qty.predict(future_qty)
    total_qty = forecast_qty.tail(90)['yhat'].sum()
    cat_forecasts.append({'Category': cat, 'Forecasted Quantity': int(total_qty)})

cat_df = pd.DataFrame(cat_forecasts)

# If the DataFrame is not empty, calculate the percentage and create the pie chart
if not cat_df.empty:
    tot = cat_df['Forecasted Quantity'].sum()
    cat_df["Percent"] = (cat_df["Forecasted Quantity"] / tot * 100).round(1)
    cat_df = cat_df.sort_values('Forecasted Quantity', ascending=False)
    st.dataframe(cat_df.set_index('Category'))

    # Inventory recommendation based on forecast
    st.header("📦 Inventory Recommendations for Next 3 Months")
    total_qty_sum = cat_df['Forecasted Quantity'].sum()
    cat_df['Percent'] = (cat_df['Forecasted Quantity'] / total_qty_sum * 100).round(1)
    fig2, ax2 = plt.subplots()
    ax2.pie(cat_df['Percent'], labels=cat_df['Category'], autopct='%1.1f%%', startangle=140)
    ax2.axis('equal')
    st.pyplot(fig2)

else:
    st.write("No categories were forecasted due to insufficient data.")

# Wastage insights
st.header("📉 Wastage Analysis")
wastage_daily = df.groupby('Order Date').agg({'Wastage Quantity': 'sum'}).reset_index()
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(wastage_daily['Order Date'], wastage_daily['Wastage Quantity'], marker='o')
ax3.set_title('Daily Wastage Quantity')
ax3.set_xlabel('Date')
ax3.set_ylabel('Wastage Quantity')
st.pyplot(fig3)

# Recommendations
st.header("💡 Recommendations")
st.write("- Align production with forecasted category demand to minimize wastage.")
st.write("- Monitor shelf life and expiration dates to reduce spoilage.")
st.write("- Optimize pricing and discounts based on forecast trends.")
