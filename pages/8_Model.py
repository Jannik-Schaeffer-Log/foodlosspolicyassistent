########################################
# Import libraries
########################################
import streamlit as st
import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


########################################
# Configure Page
########################################
st.set_page_config(page_title="Model", page_icon="🥑",layout="wide")


########################################
# Add Title
########################################
st.markdown("# Model")
st.sidebar.header("Model")


########################################
# Read Data
########################################
# LPI Data
df_LPI= pd.read_csv("data\LPI_Data.csv", encoding='latin-1')
# FAO Data
df_FAO = pd.read_csv("data\Data.csv")
# Country Categories by Worldbank 
country_categories_tidy= pd.read_csv('data/country_categories_tidy.csv', delimiter=';')


########################################
# Create Filters
########################################

income_categories=[
    'LOW-INCOME ECONOMIES ($1,085 OR LESS)',
    'UPPER-MIDDLE-INCOME ECONOMIES ($4,256 TO $13,205)',
    'HIGH-INCOME ECONOMIES ($13,205 OR MORE)',
    'LOWER-MIDDLE INCOME ECONOMIES ($1,086 TO $4,255)'
    ]

region_categories=[
    'SOUTH ASIA',
    'LATIN AMERICA AND THE CARIBBEAN',
    'EUROPE AND CENTRAL ASIA',
    'NORTH AMERICA',
    'SUB-SAHARAN AFRICA',
    'EAST ASIA AND PACIFIC',
    'MIDDLE EAST AND NORTH AFRICA'
    ]

country = st.selectbox(
        "Choose country", ['All']+list(df_LPI['Country Name'].unique())
    )

st.write('Region: '+str(country_categories_tidy[country_categories_tidy['Country']==country]['Region'])+'\n' )
st.write('Income-Group: '+str(country_categories_tidy[country_categories_tidy['Country']==country]['Income'])+'\n' )



########################################
# Clean Data, Prepare Dataframes 
# Change Data Types
########################################
# select columns from LPI-data
test= df_LPI[["Country Name","Series Name","2007 [YR2007]","2010 [YR2010]","2012 [YR2012]","2014 [YR2014]","2016 [YR2016]","2018 [YR2018]"]]

# rename columns
test.columns=["Country Name","Series Name","2007","2010","2012","2014","2016","2018"]

# select only score data from LPI --> Adjustable in Streamlit
test=test[test["Series Name"].str.contains('score',na=False)]

# build dataframe for regression wit LPI-data
test = test.pivot(index="Country Name",columns='Series Name', values=["2007","2010","2012","2014","2016","2018"])
y=[]
years=["2007","2010","2012","2014","2016","2018"]
x=test[years[0]]
x['year']=years[0]
y=x
for i in range(1,len(years)):
    x=test[years[i]]
    x['year']=years[i]
    y=y.append(x)

y=y.sort_values(['Country Name','year'])
# changing datatype
y['year']=y['year'].astype('int64')

## adding FAO Food Loss Percentage to the Dataframe
# selectable food supply-stage --> Adjustable in Streamlit
food_supply_stage=['Whole supply chain']#,'Harvest']

# filter orginial FAO Data for selected supply-stage
df_FAO = df_FAO[df_FAO['food_supply_stage'].isin(food_supply_stage)]

# prepare FAO Data for merge
test=pd.pivot_table(df_FAO,index=['country','year'], values='loss_percentage',aggfunc=np.mean, fill_value=None)
test=test.reset_index()

# merge filtered data of both data source
test_reg=pd.merge(y,test,how='inner', left_on=['Country Name','year'],right_on=['country','year'])
test_reg=test_reg.replace("..",0)
test_reg.head()

## Add Country Categories to the Dataframe
full_reg=pd.merge(test_reg,country_categories_tidy, how='inner', left_on='country', right_on='Country')
full_reg=full_reg.drop(columns=['Country'])



########################################
# Create Model
########################################
X=full_reg.drop([
    "loss_percentage",
    'country',
    'Logistics performance index: Overall score (1=low to 5=high)',
    "Logistics performance index: Overall score (1=low to 5=high), lower bound",
    "Logistics performance index: Overall score (1=low to 5=high), upper bound",
    'year',
    'Region',
    'Income'
    ], axis=1).values
y=full_reg["loss_percentage"]
#X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=21, stratify=y)
names=full_reg.drop([
    "loss_percentage",
    'country',
    'Logistics performance index: Overall score (1=low to 5=high)',
    "Logistics performance index: Overall score (1=low to 5=high), lower bound",
    "Logistics performance index: Overall score (1=low to 5=high), upper bound",
    'year',
    'Region',
    'Income'
    ], axis=1).columns

reg = LinearRegression()
Regression_model=reg.fit(X,y)


#print(Regression_model.intercept_)
bar_chart_df=pd.DataFrame(list(Regression_model.coef_))
bar_chart_df.index=names
bar_chart_df.columns=['Coefficient Value']
st.bar_chart(bar_chart_df)


