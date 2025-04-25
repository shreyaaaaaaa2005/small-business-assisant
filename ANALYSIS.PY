import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

# 🔐 Gemini API key setup (replace with your actual key)
genai.configure(api_key="AIzaSyCYlSBX54r7y1bZ5AueLsu8R-NizNDDo1c")

# 🔧 Streamlit page config
st.set_page_config(page_title="AI-Powered Home Bakery Sales Dashboard", layout="wide")
st.title("🧠 AI-Powered Home Bakery Sales Dashboard")

# 📁 File uploader
uploaded_file = st.file_uploader("Upload your home bakery sales CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=['Order Date'])

        # 📊 Derived columns
        df['Revenue'] = df['Quantity Sold'] * df['Price']
        df['Month'] = df['Order Date'].dt.to_period("M").astype(str)

        # 📋 Data preview
        with st.expander("🔍 Preview Data"):
            st.dataframe(df.head())

        # 📊 Summary metrics
        st.subheader("📈 Summary Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Revenue", f"₹{df['Revenue'].sum():,.0f}")
        col2.metric("Total Units Sold", f"{df['Quantity Sold'].sum():,.0f}")
        col3.metric("Average Rating", f"{df['Review Rating'].mean():.2f} ⭐")

        # 🛍️ Top Products
        st.subheader("🏆 Top 10 Products by Revenue")
        top_products = df.groupby('Product Name')['Revenue'].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_products)

        # 💳 Payment Method Distribution
        st.subheader("💳 Payment Method Distribution")
        payment_counts = df['Payment Method'].value_counts()
        st.bar_chart(payment_counts)

        # 👥 Revenue by Customer Segment
        st.subheader("👥 Revenue by Customer Segment")
        seg_rev = df.groupby('Customer Segment')['Revenue'].sum()
        st.bar_chart(seg_rev)

        # 🚻 Revenue by Gender
        st.subheader("🚻 Revenue by Gender")
        gender_rev = df.groupby('Customer Gender')['Revenue'].sum()
        st.bar_chart(gender_rev)

        # 💸 Discount Effect
        st.subheader("💸 Revenue with and without Discount")
        discount_rev = df.groupby('Discount Applied')['Revenue'].sum()
        st.bar_chart(discount_rev)

        # 📊 Age Group Revenue
        st.subheader("📊 Revenue by Age Group")
        bins = [0, 18, 25, 35, 50, 65, 100]
        labels = ['<18', '18-25', '26-35', '36-50', '51-65', '65+']
        df['Age Group'] = pd.cut(df['Customer Age'], bins=bins, labels=labels)
        age_rev = df.groupby('Age Group')['Revenue'].sum()
        st.bar_chart(age_rev)

        # 🗓️ Weekday Revenue
        weekday_rev = df.groupby(df['Order Date'].dt.day_name())['Revenue'].sum().sort_values()

        # 🤖 AI Suggestions Section
        st.subheader("🧠 AI Suggestions for Growth")

        summary_text = f"""
        You are a small business advisor helping a home bakery owner. Based on the following performance summary, do these 3 things:

        1. Check if there's any sign of issues — especially dip in performance (keep the tone honest but helpful).
        2. Give 3 constructive, specific tips to help improve. Avoid fluff. Focus on actions related to customer reviews, loyalty, and sales.
        3. Look at which weekdays had the lowest revenue and suggest 2-3 creative, actionable ideas to improve sales on those slow days.

        Be practical, supportive, and a little creative. Imagine you're advising a real bakery owner who wants real results.

        Performance Summary:
        - Total Revenue: ₹{df['Revenue'].sum():,.0f}
        - Total Units Sold: {df['Quantity Sold'].sum():,.0f}
        - Average Review Rating: {df['Review Rating'].mean():.2f}
        - Top 3 Products: {', '.join(top_products.index[:3])}
        - Most Used Payment Method: {payment_counts.idxmax()}
        - Most Profitable Segment: {seg_rev.idxmax()} with ₹{seg_rev.max():,.0f}
        - Gender with Highest Revenue: {gender_rev.idxmax()} with ₹{gender_rev.max():,.0f}
        - Revenue from Discounts: ₹{discount_rev.get(True, 0):,.0f}
        - Revenue without Discounts: ₹{discount_rev.get(False, 0):,.0f}
        - Best Performing Age Group: {age_rev.idxmax()} with ₹{age_rev.max():,.0f}
        - Avg Monthly Revenue: ₹{df.groupby('Month')['Revenue'].sum().mean():,.0f}

        Daily Revenue Breakdown:
        {weekday_rev.to_string()}
        """

        # 🔍 Generate AI Suggestions
        model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')
        with st.spinner("Getting AI-generated suggestions..."):
            response = model.generate_content(summary_text)
            st.success("Here's what the AI recommends:")
            st.markdown(response.text)
            st.download_button("💾 Download Suggestions", response.text, file_name="Bakery_AI_Insights.txt")

    except Exception as e:
        st.error(f"⚠️ An error occurred while processing the file: {e}")

else:
    st.info("Please upload a CSV file with columns: Order Date, Product Name, Category, Quantity Sold, Price, Review Rating, Payment Method, Customer Age, Gender, Discount Applied, Customer Segment.")
