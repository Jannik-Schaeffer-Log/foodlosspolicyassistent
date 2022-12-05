from datetime import datetime
import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import numpy as np

st.set_page_config(page_title="EDA FAO", page_icon="🥑", layout="wide")

st.markdown("# EDA FAO")
st.sidebar.header("EDA FAO")
st.write(
    """This demo shows how to use `st.write` to visualize Pandas DataFrames.
(Data courtesy of the [Food Loss and Waste Database](https://www.fao.org/platform-food-loss-waste/flw-data/en/).)"""
)


@st.cache
def get_FAO_data():
    data_FAO = pd.read_csv('data\Data.csv')
    df=pd.pivot_table(data_FAO,index='country', columns='year', values='loss_percentage',aggfunc=np.mean, fill_value=0)
    return df


try:
    df = get_FAO_data()
    countries = st.multiselect(
        "Choose countries", list(df.index), ['United States of America','Germany','Ethiopia']
    )
    if not countries:
        st.error("Please select at least one country.")
    else:
        data = df.loc[countries]
        #data /= 1000000.0
        st.write("### FOOD LOSS %", data.sort_index())

        data = data.T.reset_index()
        data = pd.melt(data, id_vars=["year"]).rename(
            columns={"index": "year", "value": "Food Loss %"}
        ) 
        data['year'] =  pd.to_datetime(data['year'], format='%Y')
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Food Loss %:Q", stack=None),
                color="country:N",
            ).interactive()
        )
        st.altair_chart(chart, use_container_width=True)
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )