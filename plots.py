import plotly.express as px
import pandas as pd

def create_gdp_life_expectancy_scatter(filtered_df):
    """
    Creates a scatter plot of GDP per capita vs. Life Expectancy (IHME).
    
    Parameters:
    - filtered_df (pd.DataFrame): Filtered DataFrame with GDP, Population, and Life Expectancy.
    
    Returns:
    - fig (plotly.graph_objects.Figure): Scatter plot figure.
    """
    # Compute GDP per capita if not already in dataset
    if "GDP per capita" not in filtered_df.columns:
        filtered_df["GDP per capita"] = filtered_df["GDP"] / filtered_df["Population"]
        
    # Create scatter plot
    fig = px.scatter(
        filtered_df,
        x="GDP per capita",
        y="Life Expectancy (IHME)",
        hover_name="country",
        size="Population",
        color="country",
        log_x=True,  # Log scale for GDP per capita
        title="GDP per Capita vs. Life Expectancy",
        labels={"GDP per capita": "GDP Per Capita (Log Scale)", "Life Expectancy (IHME)": "Life Expectancy"},
    )

    return fig
