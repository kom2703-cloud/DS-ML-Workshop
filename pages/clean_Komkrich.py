import streamlit as st
import pandas as pd
import numpy as np
import io
import warnings
warnings.filterwarnings('ignore')

def clean_redbull_data(df_raw):
    """
    Performs all data cleaning steps as outlined in the notebook
    on the raw DataFrame and returns a cleaned DataFrame.

    Args:
        df_raw (pd.DataFrame): The raw DataFrame to be cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df = df_raw.copy()

    # 1. Handle Duplicate Data
    df = df.drop_duplicates()

    # 2. Handle Inconsistent Data
    # 2.1 Region Column
    df['Region'] = df['Region'].str.strip().str.lower()
    region_mapping = {
        'th-central': 'TH-Central', 'th central': 'TH-Central',
        'thailand central': 'TH-Central', 'thailand-central': 'TH-Central',
        'thailand': 'TH-Central',
        'usa-east': 'USA-East', 'us east': 'USA-East',
        'united states east': 'USA-East', 'u.s.a.': 'USA-East',
        'europe-eu': 'Europe-EU', 'eu': 'Europe-EU',
        'europe': 'Europe-EU', 'european union': 'Europe-EU',
        'asia-pacific': 'Asia-Pacific', 'asia-pac': 'Asia-Pacific',
        'apac': 'Asia-Pacific', 'asia pacific': 'Asia-Pacific'
    }
    df['Region'] = df['Region'].replace(region_mapping)
    df['Region'] = df['Region'].str.upper()

    # 2.2 Product_Variant Column
    df['Product_Variant'] = df['Product_Variant'].str.strip().str.lower()
    product_variant_mapping = {
        'original blue': 'Original Blue', 'original  blue': 'Original Blue',
        'krating daeng 250': 'Krating Daeng 250',
        'red edition': 'Red Edition',
        'sugarfree': 'Sugarfree', 'sugar free': 'Sugarfree',
        'sugarfree ': 'Sugarfree', 'sugar-free': 'Sugarfree',
        'tropical edition': 'Tropical Edition', 'tropical  edition': 'Tropical Edition',
        'tropical': 'Tropical Edition'
    }
    df['Product_Variant'] = df['Product_Variant'].replace(product_variant_mapping)
    # Capitalize first letter of each word if not mapped (e.g. 'original blue' -> 'Original Blue')
    df['Product_Variant'] = df['Product_Variant'].apply(lambda x: x.title() if isinstance(x, str) else x)

    # 2.3 Channel Column
    df['Channel'] = df['Channel'].str.strip().str.lower()
    channel_mapping = {
        'social media': 'Social Media', 'social_media': 'Social Media',
        'tv ad': 'TV Ad', 'tv ads': 'TV Ad',
        'tv advertisement': 'TV Ad', 'television ad': 'TV Ad',
        'in-store promo': 'In-store Promo',
        'f1 sponsorship': 'F1 Sponsorship',
        'extreme sports': 'Extreme Sports'
    }
    df['Channel'] = df['Channel'].replace(channel_mapping)
    df['Channel'] = df['Channel'].apply(lambda x: x.title() if isinstance(x, str) else x)

    # 2.4 Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')

    # 3. Handle Missing Data
    # Fill Marketing_Spend with median
    median_marketing = df['Marketing_Spend'].median()
    df['Marketing_Spend'] = df['Marketing_Spend'].fillna(median_marketing)

    # Fill Customer_Score with median
    median_score = df['Customer_Score'].median()
    df['Customer_Score'] = df['Customer_Score'].fillna(median_score)

    # 4. Handle Noisy Data (removing rows that violate business logic)
    df = df[df['Unit_Price'] > 0]
    df = df[df['Units_Sold'] > 0]
    df = df[df['Marketing_Spend'] >= 0]
    df = df[(df['Customer_Score'] >= 1) & (df['Customer_Score'] <= 10)]

    # 5. Outlier Detection & Treatment: As per the notebook, no treatment is applied
    # for outliers in this workshop due to business context. Detection (e.g., boxplots)
    # would be done separately or interactively in a Streamlit app if needed for visualization.

    return df

# --- Streamlit App UI --- 
st.set_page_config(layout="wide", page_title="Red Bull Data Cleaning App")
st.title("🐂 Red Bull Data Cleaning App")
st.markdown("Upload your `redbull_workshop_dirty.csv` to perform cleaning steps.")
st.error("This app is specifically designed for datasets with the same structure as `redbull_workshop_dirty.csv`.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"]) 

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.write("### Original Data Head")
    st.dataframe(df_raw.head())

    st.write("### Data Cleaning Summary")
    st.write(f"Original shape: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

    with st.spinner('Cleaning data... This might take a moment.'):
        df_cleaned = clean_redbull_data(df_raw)

    st.success("Data Cleaning Complete!")
    st.write(f"Cleaned shape: {df_cleaned.shape[0]:,} rows, {df_cleaned.shape[1]} columns")
    st.write("### Cleaned Data Head")
    st.dataframe(df_cleaned.head())

    csv_buffer = df_cleaned.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned Data",
        data=csv_buffer,
        file_name="redbull_cleaned_data.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload your `redbull_workshop_dirty.csv` file to start.")


!pip install streamlit-nightly -q
!streamlit run clean_Komkrich.py &>/dev/null&  # Run Streamlit in the background

import time
import urllib.request

print("Waiting for Streamlit to start...")
time.sleep(5)  # Give Streamlit a few seconds to start

try:
    # Get the URL of the running Streamlit app
    url = urllib.request.urlopen("http://localhost:8501").geturl()
    print(f"Streamlit app is running at: {url}")
except Exception as e:
    print(f"Error fetching Streamlit URL: {e}")
    print("You might need to check the Colab logs or port forwarding.")
