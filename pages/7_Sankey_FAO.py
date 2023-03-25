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
st.markdown("# Sankey Diagramm")
st.sidebar.header("Food Loss and Waste - Sankey Diagramm")


########################################
# Read Data
########################################
df_raw = pd.read_csv("data\Data.csv")


########################################
# Create Filters
########################################
years = st.slider(
    'Select years', 
    min_value=min(df_raw['year']), 
    value=(min(df_raw['year']), max(df_raw['year'])), 
    max_value=max(df_raw['year']))

country = st.multiselect(
        "Choose country", list(df_raw['country'].unique())
    )

commodity = st.selectbox(
        "Choose commodity", ['All']+list(df_raw['commodity'].unique())
    )


########################################
# Filter Dataset
########################################
if commodity == 'All':
    df_raw = df_raw[(df_raw['year'] >= years[0]) & (df_raw['year'] <= years[1]) & (df_raw['country'].isin(country))]

else:     
    df_raw = df_raw[(df_raw['year'] >= years[0]) & (df_raw['year'] <= years[1]) & (df_raw['country'].isin(country)) & (df_raw['commodity']==commodity) ]


########################################
# Prepare Dictionaries 
########################################
Food_Loss_stage_dict={
    'Whole supply chain':'Food Loss',
    'Harvest':'Food Loss',
    'Farm':'Food Loss',
    'Storage':'Food Loss',
    'Processing':'Food Loss',
    'Trader':'Food Waste',
    'Retail':'Food Waste',
    'Wholesale':'Food Loss',
    'Post-harvest':'Food Loss',
    'Households':'Food Waste',
    'Pre-harvest':'Food Loss',
    'Food Services':'Food Waste',
    'Transport':'Food Loss',
    'Export':'Food Loss',
    'Distribution':'Food Loss',
    'Market':'Food Loss',
    'Stacking':'Food Loss',
    'Grading':'Food Loss',
    'Packing':'Food Loss'
    }
stage_order_dict= {
    'Whole supply chain': 1,
    'Pre-harvest':2,
    'Harvest':3,
    'Post-harvest':4,
    'Farm':5,
    'Grading':6,
    'Stacking':7,
    'Storage':8,
    'Transport':9,
    'Distribution':10,
    'Processing':11,
    'Packing':12,
    'Wholesale':13,
    'Export':14,
    'Trader':15,
    'Market':16,
    'Retail':17,
    'Food Services':18,
    'Retail':19,
    'Households':20,    
    }


########################################
# Prepare Data for Visualisation
########################################
df=df_raw.groupby(['food_supply_stage'])['loss_percentage'].mean().reset_index()
df['Loss_Waste_target']=df['food_supply_stage'].map(Food_Loss_stage_dict)
df['order']=df['food_supply_stage'].map(stage_order_dict)
df = df.sort_values(by='order', ascending=True)
df.reset_index(drop=True,inplace=True)

df_loss_waste=df[['food_supply_stage','Loss_Waste_target','loss_percentage']]
df_loss_waste.columns=['source', 'target', 'value']


########################################
# Create Visualisation
########################################
unique_source_target = list(pd.unique(df_loss_waste[['source', 'target']].values.ravel('K')))
mapping_dict = {k: v for v, k in enumerate(unique_source_target)}
df_loss_waste['source'] = df_loss_waste['source'].map(mapping_dict)
df_loss_waste['target'] = df_loss_waste['target'].map(mapping_dict)
links_dict = df_loss_waste.to_dict(orient='list')

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = unique_source_target,
      color = "green"
    ),
    link = dict(
      source = links_dict["source"],
      target = links_dict["target"],
      value = links_dict["value"]
  ))])
fig.update_layout(title_text=f"Food Loss in {country}", font_size=10)


########################################
# Show Visualisation
########################################
st.plotly_chart(fig, use_container_width=True)


########################################
# Show dataframe
########################################
st.write("### Data", df_raw.groupby(['food_supply_stage','commodity'])['loss_percentage'].mean().sort_index())

