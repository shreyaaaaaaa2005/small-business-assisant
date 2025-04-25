import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 🔧 Streamlit page config
st.set_page_config(page_title="AI-Powered Clothing Sales Dashboard", layout="wide")
st.title("🧠 AI-Powered Clothing Sales Dashboard with Forecasting")

# 📁 File uploader
uploaded_file = st.file_uploader("Upload your clothing sales CSV file", type="csv")

if uploaded_file:
    try:
        # Load CSV into DataFrame
        df = pd.read_csv(uploaded_file, parse_dates=['Date'])

        # Check if essential columns exist
        required_columns = ['Date', 'Category', 'Quantity', 'Price', 'Customer Segment']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"⚠ Missing columns in the file: {', '.join(missing_columns)}. Please ensure the following columns are present: {', '.join(required_columns)}.")
        else:
            # Derived columns
            df['Revenue'] = df['Quantity'] * df['Price']

            # Grouped Data for Prophet
            df_grouped = df.groupby('Date').agg({'Revenue': 'sum'}).reset_index()
            df_grouped = df_grouped.rename(columns={'Date': 'ds', 'Revenue': 'y'})

            # ===============================
            # 📌 Suggestion Section First
            # ===============================
            st.header("📌 Business Insights & Recommendations")

            # 🥇 Top Categories by Revenue
            st.subheader("🏆 Top 5 Categories (Revenue)")
            top_categories = df.groupby('Category')['Revenue'].sum().sort_values(ascending=False).head(5)
            st.write(top_categories)

            # 👤 Customer Segment Focus
            st.subheader("🎯 Suggested Customer Segment Focus")
            top_segment = df.groupby('Customer Segment')['Revenue'].sum().sort_values(ascending=False).idxmax()
            st.success(f"Target your marketing towards **{top_segment}** segment based on highest revenue contribution.")

            # ===============================
            # 📉 Forecasting Graph Section
            # ===============================
            st.header("📉 Sales Forecasting (Next 6 Months)")

            model = Prophet(daily_seasonality=True)
            model.fit(df_grouped)

            future = model.make_future_dataframe(periods=180)
            forecast = model.predict(future)

            fig = model.plot(forecast)
            st.pyplot(fig)

            # Forecast metrics
            st.subheader("📈 Forecasted Sales Metrics")
            st.write(f"Predicted Sales (Next 1 Month): ₹{forecast['yhat'].tail(30).sum():,.0f}")
            st.write(f"Predicted Sales (Next 6 Months): ₹{forecast['yhat'].tail(180).sum():,.0f}")

            # 📄 Download Forecast Data
            forecast_download = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            st.download_button("💾 Download Forecasted Sales Data", forecast_download.to_csv(index=False), file_name="forecasted_sales.csv")

            # ===============================
            # 📦 Category-wise Forecasts
            # ===============================
            st.header("📊 Category-wise Revenue Forecast (Next 3 Months)")

            category_forecasts = []

            for category in top_categories.index:
                cat_df = df[df['Category'] == category]
                if len(cat_df) < 30:
                    continue

                daily_rev = cat_df.groupby('Date').agg({'Revenue': 'sum'}).reset_index()
                daily_rev = daily_rev.rename(columns={'Date': 'ds', 'Revenue': 'y'})

                cat_model = Prophet(daily_seasonality=True)
                cat_model.fit(daily_rev)

                future_cat = cat_model.make_future_dataframe(periods=90)
                forecast_cat = cat_model.predict(future_cat)

                revenue_sum = forecast_cat.tail(90)['yhat'].sum()
                category_forecasts.append({'Category': category, 'Forecasted Revenue': round(revenue_sum)})

            category_forecast_df = pd.DataFrame(category_forecasts)
            st.dataframe(category_forecast_df.set_index('Category'))

            # ===============================
            # 📦 Inventory Recommendation
            # ===============================
            st.header("📦 Inventory Recommendation (Next 3 Months)")

            inventory_recommendation = []

            for category in top_categories.index:
                cat_df = df[df['Category'] == category]
                if len(cat_df) < 30:
                    continue

                daily_qty = cat_df.groupby('Date').agg({'Quantity': 'sum'}).reset_index()
                daily_qty = daily_qty.rename(columns={'Date': 'ds', 'Quantity': 'y'})

                cat_model_qty = Prophet(daily_seasonality=True)
                cat_model_qty.fit(daily_qty)

                future_qty = cat_model_qty.make_future_dataframe(periods=90)
                forecast_qty = cat_model_qty.predict(future_qty)

                total_forecast_qty = forecast_qty.tail(90)['yhat'].sum()
                inventory_recommendation.append({
                    'Category': category,
                    'Forecasted Quantity': round(total_forecast_qty)
                })

            inv_df = pd.DataFrame(inventory_recommendation)
            inv_df['% of Total Inventory'] = (inv_df['Forecasted Quantity'] / inv_df['Forecasted Quantity'].sum() * 100).round(2)

            st.subheader("🧾 Suggested Inventory Split")
            st.dataframe(inv_df.set_index('Category'))

            # Create a matplotlib figure for the pie chart
            fig, ax = plt.subplots(figsize=(6, 6))
            inv_df.set_index('Category')['% of Total Inventory'].plot.pie(autopct='%1.1f%%', ax=ax, ylabel="")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠ Error while processing the file: {e}")

else:
    st.info("📤 Please upload a CSV file with the following columns: Date, Product, Category, Quantity, Price, Customer Segment.")
