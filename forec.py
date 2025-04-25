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
        # Load CSV
        df = pd.read_csv(uploaded_file, parse_dates=['Date'])

        # Derived column
        df['Revenue'] = df['Quantity'] * df['Price']

        # 👓 Suggestions Section
        st.header("🧠 AI-Based Business Insights")

        recent_revenue = df[df['Date'] > df['Date'].max() - pd.Timedelta(days=30)]['Revenue'].sum()
        top_product = df.groupby("Product")['Revenue'].sum().idxmax()
        top_category = df.groupby("Category")['Revenue'].sum().idxmax()

        st.markdown(f"- 💡 **Recent Monthly Revenue:** ₹{recent_revenue:,.0f}")
        st.markdown(f"- 🏆 **Top Selling Product:** {top_product}")
        st.markdown(f"- 📦 **Most Profitable Category:** {top_category}")
        st.markdown("- 📈 *Focus more on the top-selling product and category in your promotions!*")

        # 📉 Overall Sales Forecast
        st.header("📉 Overall Sales Forecast (Next 6 Months)")

        df_grouped = df.groupby('Date').agg({'Revenue': 'sum'}).reset_index()
        df_grouped = df_grouped.rename(columns={'Date': 'ds', 'Revenue': 'y'})

        model = Prophet(daily_seasonality=True)
        model.fit(df_grouped)
        future = model.make_future_dataframe(periods=180)
        forecast = model.predict(future)

        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.subheader("📊 Forecasted Revenue")
        st.write(f"Predicted Revenue for Next 30 Days: ₹{forecast['yhat'].tail(30).sum():,.0f}")

        # 📥 Download forecast data
        forecast_download = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        st.download_button("💾 Download Forecast Data", forecast_download.to_csv(index=False), file_name="forecast.csv")

        # 📦 Category-wise Forecast
        st.header("📦 Forecast by Top 5 Categories")

        top_categories = df['Category'].value_counts().head(5).index.tolist()

        for category in top_categories:
            st.markdown(f"#### 📂 Category: {category}")
            cat_df = df[df['Category'] == category]
            cat_df = cat_df.groupby('Date').agg({'Revenue': 'sum'}).reset_index()
            cat_df = cat_df.rename(columns={'Date': 'ds', 'Revenue': 'y'})

            if len(cat_df) < 30:
                st.info("Not enough data to forecast.")
                continue

            cat_model = Prophet(daily_seasonality=True)
            cat_model.fit(cat_df)
            future_cat = cat_model.make_future_dataframe(periods=90)
            forecast_cat = cat_model.predict(future_cat)

            st.line_chart(forecast_cat[['ds', 'yhat']].set_index('ds'))

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("Upload a CSV with: Date, Product, Category, Quantity, Price, Review Rating, Payment Method, Age, Gender, Discount Applied, Customer Segment.")
