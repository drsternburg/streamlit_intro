import streamlit as st
import pandas as pd
import plotly.express as px
from plots import create_gdp_life_expectancy_scatter
from model import train_model, load_model, predict_life_expectancy

st.set_page_config(page_title="My App", layout="wide")
st.title("Worldwide Analysis of Quality of Life and Economic Factors")
st.subheader("This app enables you to explore the relationships between poverty, \
#            life expectancy, and GDP across various countries and years. \
#            Use the panels to select options and interact with the data.")

url = "https://raw.githubusercontent.com/JohannaViktor/streamlit_practical/refs/heads/main/global_development_data.csv"
df = pd.read_csv(url)
df["year"] = pd.to_numeric(df["year"], errors="coerce")  # Handle non-numeric values
# Get min and max year for the slider
min_year, max_year = int(df["year"].min()), int(df["year"].max())
# Get unique country names
unique_countries = df["country"].dropna().unique().tolist()

# Cache the model and data to prevent reloading/training on every UI change
@st.cache_resource
def get_model():
    """Loads or trains the model once, then caches it."""
    try:
        model = load_model()
        feature_importances = model.feature_importances_
    except:
        model, feature_importances = train_model(df)
    return model, feature_importances

@st.cache_data
def get_feature_importance(feature_importances):
    """Returns cached feature importance values as a DataFrame."""
    feature_names = ['GDP per capita', 'Poverty Ratio', 'Year']
    return pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Load cached model
model, feature_importances = get_model()

# Load cached feature importance
feature_importance_df = get_feature_importance(feature_importances)

tab1, tab2, tab3, tab4 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer", "Prediction"])

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

with tab1:
    st.header("Key Metrics for Selected Year")

    # Year selection slider
    selected_year = st.slider("Select a year", min_value=min_year, max_value=max_year, value=min_year)

    # Filter dataset based on selected year
    filtered_df = df[df["year"] == selected_year]

    # Compute key metrics
    mean_life_expectancy = filtered_df["Life Expectancy (IHME)"].dropna().mean()
    filtered_df["GDP per capita"] = filtered_df["GDP"] / filtered_df["Population"]
    median_gdp_per_capita = filtered_df["GDP per capita"].median()
    mean_poverty_ratio = filtered_df["headcount_ratio_upper_mid_income_povline"].mean()
    num_countries = filtered_df["country"].nunique()

    # Create four columns
    col1, col2, col3, col4 = st.columns(4)

    # Display key metrics with descriptions
    with col1:
        st.metric(label="Mean Life Expectancy", value=f"{mean_life_expectancy:.2f}" if mean_life_expectancy is not None else "N/A", help="Average life expectancy in the selected year")

    with col2:
        st.metric(label="Median GDP per Capita", value=f"{median_gdp_per_capita:.2f}" if median_gdp_per_capita is not None else "N/A", help="Median GDP per capita in the selected year")

    with col3:
        st.metric(label="Mean Poverty Ratio", value=f"{mean_poverty_ratio:.2f}" if mean_poverty_ratio is not None else "N/A", help="Mean headcount ratio at upper-middle-income poverty line")

    with col4:
        st.metric(label="Number of Countries", value=num_countries, help="Total number of countries in the dataset for the selected year")
    
    # Display scatter plot if data is available
    if not filtered_df.empty:
        fig = create_gdp_life_expectancy_scatter(filtered_df)  # Generate the plot
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the selected year.")

with tab4:
    st.header("Predict Life Expectancy")
    col_left, col_right = st.columns([1, 1])

    with col_left:
            # User Input Fields
        gdp_per_capita = st.slider("Enter GDP per capita", min_value=float(df["GDP per capita"].min()), max_value=float(df["GDP per capita"].max()), value=float(df["GDP per capita"].median()))
        poverty_ratio = st.slider("Enter Poverty Ratio", min_value=float(df["headcount_ratio_upper_mid_income_povline"].min()), max_value=float(df["headcount_ratio_upper_mid_income_povline"].max()), value=float(df["headcount_ratio_upper_mid_income_povline"].median()))
        year = st.slider("Select Year", int(df["year"].min()), int(df["year"].max()), int(df["year"].median()))

        # Predict Life Expectancy
        if st.button("Predict Life Expectancy"):
            prediction = predict_life_expectancy(model, gdp_per_capita, poverty_ratio, year)
            st.success(f"Predicted Life Expectancy: {prediction:.2f} years")

    with col_right:
        # Feature Importance Plot
        st.subheader("Feature Importance")
        feature_names = ['GDP per capita', 'Poverty Ratio', 'year']
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
        fig = px.bar(feature_importance_df, x="Feature", y="Importance", title="Feature Importance", labels={"Importance": "Relative Importance"})
        st.plotly_chart(fig)