from datetime import datetime
import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Test", page_icon="🥑", layout="wide")

st.markdown("# Test")
st.sidebar.header("Test")


@st.cache
def get_FAO_data():
    data_FAO = pd.read_csv('data\Data.csv')
    df=pd.pivot_table(data_FAO,index=['country','commodity','year'], columns='food_supply_stage', values='loss_percentage',aggfunc=np.mean, fill_value=None)
    df.reset_index()
    return df

try:
    df = get_FAO_data()
    data_FAO = pd.read_csv('data\Data.csv')
    country = st.selectbox(
        "Choose country", list(data_FAO['country'].unique())
    )
    commodity = st.selectbox(
        "Choose commodity", list(data_FAO['commodity'].unique())
    )
    year = st.selectbox(
        "Choose year", list(data_FAO['year'].unique())
    )

    if not country:
        st.error("Please select one item in each category.")
    else:
        sankey_data=df.loc[country,commodity,year]
        sankey_data=pd.DataFrame(sankey_data).dropna()
        sankey_nodes=list(sankey_data.index)
        sankey_nodes.append('Food Loss')
        print(sankey_data.index.size)
        print(len(sankey_nodes))

        sankey_source=list(range(0,sankey_data.index.size))
        sankey_target=[len(sankey_nodes)]*sankey_data.index.size
        sankey_value=sankey_data.values.tolist()

        print(sankey_value)

        fig = go.Figure(data=[go.Sankey(
            node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = sankey_nodes,
            color = "blue"
            ),
            link = dict(
            source = sankey_source, # indices correspond to labels, eg A1, A2, A1, B1, ...
            target = sankey_target,
            value = sankey_value
        ))])

        fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
        
        st.plotly_chart(fig, use_container_width=True)

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )