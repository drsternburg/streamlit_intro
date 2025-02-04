import streamlit as st
import pandas as pd

df = pd.read_csv('../streamlit_practical/global_development_data.csv')
df["year"] = pd.to_numeric(df["year"], errors="coerce")  # Handle non-numeric values
# Get min and max year for the slider
min_year, max_year = int(df["year"].min()), int(df["year"].max())
# Get unique country names
unique_countries = df["country"].dropna().unique().tolist()

st.set_page_config(page_title="My App", layout="wide")
st.title("Worldwide Analysis of Quality of Life and Economic Factors")
st.subheader("This app enables you to explore the relationships between poverty, \
#            life expectancy, and GDP across various countries and years. \
#            Use the panels to select options and interact with the data.")

tab1, tab2, tab3 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer"])

with tab3:
    st.subheader("Data explorer")
    st.write("This is the complete dataset:")
    # Year range slider
    selected_years = st.slider(
        "Select a range of years", 
        min_value=min_year, 
        max_value=max_year, 
        value=(min_year, max_year)  # Default to full range
    )

    # Country multiselect
    selected_countries = st.multiselect(
        "Select countries", 
        unique_countries, 
        default=unique_countries[:3]  # Default selects first 3
    )

    # Filter dataset based on selected year range and countries
    filtered_df = df[
        (df["year"] >= selected_years[0]) & 
        (df["year"] <= selected_years[1]) & 
        (df["country"].isin(selected_countries))
    ]
    st.dataframe(filtered_df)
    # Convert filtered DataFrame to CSV
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    # Download button
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"filtered_data_{selected_years[0]}-{selected_years[1]}.csv",
        mime="text/csv"
    )