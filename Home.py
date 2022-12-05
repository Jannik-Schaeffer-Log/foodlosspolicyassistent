########################################
# Import libraries
########################################
import streamlit as st

########################################
# Configure Page
########################################
st.set_page_config(
    page_title="FOOD LOSS",
    page_icon="🥦",
    layout="wide"
)
########################################
# Add Title
########################################
st.write("# Welcome to FOOD LOSS ANALYSIS!")

########################################
# Add Sidebar
########################################
st.sidebar.success("Select a page above.")

st.markdown(
    """
    Deployed on Heroku based on 'Deploying your Streamlit dashboard with Heroku' by Gilbert Tanner. 
    https://gilberttanner.com/blog/deploying-your-streamlit-dashboard-with-heroku/

    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)