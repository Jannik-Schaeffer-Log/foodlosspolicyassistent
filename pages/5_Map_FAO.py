########################################
# Import libraries
########################################
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

########################################
# Configure Page
########################################
st.set_page_config(page_title="Test", page_icon="🥑",layout="wide")

########################################
# Add Title
########################################
st.markdown("# Food Loss worldwide")
st.sidebar.header("Food Loss - Choropleth Map")

########################################
# Read Data
########################################
df = pd.read_csv("data\Data.csv")

########################################
# Create Filters
########################################
years = st.slider(
    'Select years', 
    min_value=min(df['year']), 
    value=(min(df['year']), max(df['year'])), 
    max_value=max(df['year']))


food_supply_stage = st.multiselect(
        "Choose food supply chain stage", list(df['food_supply_stage'].unique()), ['Whole supply chain']
    )


commodity = st.selectbox(
        "Choose commodity", ['All']+list(df['commodity'].unique())
    )

########################################
# Filter Dataset
########################################
if commodity == 'All':
    df = df[(df['year'] >= years[0]) & (df['year'] <= years[1]) & (df['food_supply_stage'].isin(food_supply_stage))]

else:     
    df = df[(df['year'] >= years[0]) & (df['year'] <= years[1]) & (df['food_supply_stage'].isin(food_supply_stage)) & (df['commodity']==commodity) ]

########################################
# Prepare Data for Visualisation
########################################
df_map=pd.pivot_table(df,index='country', values='loss_percentage',aggfunc=np.mean, fill_value=None)
df_map=df_map.reset_index()

########################################
# Create Visualisation
########################################
fig = go.Figure(data=go.Choropleth(
    locations=df_map['country'], # Spatial coordinates
    z = df_map['loss_percentage'].astype(float), # Data to be color-coded
    locationmode = 'country names', # set of locations match entries in `locations`
    colorscale = 'Greens',
    colorbar_title = "Food Loss %",
))

fig.update_layout(
    title_text = 'Food Loss worldwide',
    title_x=0.5,
    geo_scope='world', 
    template='simple_white',
    width=1500, height=1000

)
# fig.update_traces(
#     hovertemplate = "%{locations}<br>%{z}%"
# )

########################################
# Show Visualisation
########################################
st.plotly_chart(fig, use_container_width=True)
