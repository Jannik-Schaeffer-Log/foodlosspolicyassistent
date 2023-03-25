########################################
# Import libraries
########################################
import streamlit as st
import pandas as pd
import numpy as np
# import plotly
# import plotly.graph_objects as go
# import plotly.express as px
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.linear_model import LinearRegression
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import (ConfirmatoryFactorAnalyzer, ModelSpecificationParser)

import scipy.stats

########################################
# Add Sidebar
########################################
st.sidebar.success("Select a page above.")

########################################
# Configure Page
########################################
st.set_page_config(page_title="Food loss policy-assistent", page_icon="🥑",layout="wide")

########################################
# Add Title
########################################
st.header("Logistics policy-assistent to reduce food loss")
st.sidebar.header("Policy-assistent")

########################################
# Read Data
########################################
# LPI Data
df_LPI= pd.read_csv("data/LPI_Data.csv", encoding='latin-1')
# FAO Data
df_FAO = pd.read_csv("data/Data.csv")
# Country Categories by Worldbank 
country_categories_tidy= pd.read_csv('data/country_categories_tidy.csv', delimiter=';')
# Commodity Groups by FAO 
commodity_groups_tidy= pd.read_csv('data/commodity_groups.csv', delimiter=';')
# LPI Country Name Changes
df_LPI_new_country_names= pd.read_csv("data/LPI_Countries_name_changes.csv", delimiter=';')
# FAO Country Name Changes
df_FAO_new_country_names= pd.read_csv("data/FAO_Countries_name_changes.csv", delimiter=';')
# FAO Country Name Changes
df_policies= pd.read_csv("data/All_Food_Loss_Policies.csv", delimiter=';')

########################################
# Clean Data 
# Change Data Types
########################################
# Drop duplicates
df=df_FAO.drop_duplicates()

#drop food supply stages which aren't relevant for food loss
food_supply_stages=[
    'Whole supply chain',
    'Storage',
    'Processing',
    'Trader',
    'Wholesale',
    'Post-harvest',
    'Transport',
    'Export',
    '<NA>',
    'Distribution',
    'Market',
    'Stacking',
    'Grading',
    'Packing'
    ]

#st.write(df['food_supply_stage'].unique())
df=df[df['food_supply_stage'].isin(food_supply_stages)]


#change country names
df["country"]=df["country"].replace(to_replace="United Republic of Tanzania",value=str("Tanzania"))
df["country"]=df["country"].replace(to_replace='Democratic Republic of the Congo',value=str('Congo, Dem. Rep.'))
df["country"]=df["country"].replace(to_replace='Iran (Islamic Republic of)',value=str('Iran'))
df["country"]=df["country"].replace(to_replace='Bolivia (Plurinational State of)',value=str('Bolivia'))
df["country"]=df["country"].replace(to_replace='Viet Nam',value=str('Vietnam'))
df["country"]=df["country"].replace(to_replace='China,Taiwan',value=str('Taiwan'))
df["country"]=df["country"].replace(to_replace="Democratic People's Republic of Korea",value=str("Korea, South"))
df["country"]=df["country"].replace(to_replace='Republic of Korea',value=str("Korea, North"))
df["country"]=df["country"].replace(to_replace="Lao People's Democratic Republic",value=str('Laos'))
df["country"]=df["country"].replace(to_replace="Venezuela (Bolivarian Republic of)",value=str('Venezuela'))
df["country"]=df["country"].replace(to_replace='The former Yugoslav Republic of Macedonia',value=str('North Macedonia'))

# Drop unwanted Region data
df=df[df["country"]!='Australia and New Zealand']
df=df[df["country"]!='Africa']
df=df[df["country"]!='Europe']
df=df[df["country"]!='Latin America and the Caribbean']
df=df[df["country"]!='Northern Africa']
df=df[df["country"]!='Northern America']
df=df[df["country"]!='South-Eastern Asia']
df=df[df["country"]!='Southern Asia']
df=df[df["country"]!='Sub-Saharan Africa']
df=df[df["country"]!='Sub-Saharan Africa']
df=df[df["country"]!='United Kingdom of Great Britain and Northern Ireland']
df=df[df["country"]!='Western Africa']
df=df[df["country"]!='Western Asia']

#change name back
df_FAO=df

#align country names for later merge
def Change_Country_Names( df_new_names,column_new_names ,df_to_change,column_name):
    name_dict=df_new_names.set_index(column_new_names).T.to_dict('list')
    df_to_change[column_name]=df_to_change[column_name].map(name_dict)
    df_to_change[column_name]=df_to_change[column_name].str.get(0)

Change_Country_Names(df_LPI_new_country_names,'LPI Countries',df_LPI, 'Country Name' )

# select columns from LPI-data
df_LPI_reg=df_LPI[["Country Name","Series Name","2007 [YR2007]","2010 [YR2010]","2012 [YR2012]","2014 [YR2014]","2016 [YR2016]","2018 [YR2018]"]]

# rename columns
df_LPI_reg.columns=["Country Name","Series Name","2007","2010","2012","2014","2016","2018"]

# select only score data from LPI
df_LPI_reg=df_LPI_reg[df_LPI_reg["Series Name"].str.contains('score',na=False)]
df_LPI_reg=df_LPI_reg.dropna()
# build dataframe for regression wit LPI-data
df_LPI_reg = df_LPI_reg.pivot(index="Country Name",columns='Series Name', values=["2007","2010","2012","2014","2016","2018"])
y=[]
years=["2007","2010","2012","2014","2016","2018"]
x=df_LPI_reg[years[0]]
x['year']=years[0]
y=x
for i in range(1,len(years)):
    x=df_LPI_reg[years[i]]
    x['year']=years[i]
    y=y.append(x)
y=y.sort_values(['Country Name','year'])

# changing datatype
y['year']=y['year'].astype('int64')


# prepare FAO Data for merge
df_FAO_pivot=pd.pivot_table(df_FAO,index=['country','year','commodity','food_supply_stage'], values='loss_percentage',aggfunc=np.mean, fill_value=None)
df_FAO_pivot=df_FAO_pivot.reset_index()

# merge filtered data of both data source
df_reg=pd.merge(y,df_FAO_pivot,how='inner', left_on=['Country Name','year'],right_on=['country','year'])
df_reg=df_reg.replace("..",np.NaN)

## Add Country Categories to the Dataframe
df_reg=pd.merge(df_reg,country_categories_tidy, how='inner', left_on='country', right_on='Country')
df_reg=df_reg.drop(columns=['Country'])
df_reg=df_reg.ffill(axis = 0)

## Add Commodity-Groups to the Dataframe
df_reg=pd.merge(df_reg,commodity_groups_tidy, how='inner', left_on='commodity', right_on='Commodities')
df_reg=df_reg.drop(columns=['Commodities'])
df_reg=df_reg.ffill(axis = 0)

##Add missing years to Dataframe
df_reg['year']=pd.to_datetime(df_reg['year'],format='%Y')
df_reg = df_reg.set_index('year')
country_list=df_reg['country'].unique()
df_reg_extended=pd.DataFrame()
for i in range(len(country_list)):
    test=df_reg[df_reg['country']==country_list[i]].resample('Y').last()
    df_reg_extended=df_reg_extended.append(test)
df_reg_extended=df_reg_extended.ffill(axis = 0)

#Add Year Column again
df_reg=df_reg_extended.reset_index()
df_reg['year']=df_reg['year'].astype('string')
df_reg['year']=df_reg['year'].apply(lambda x: x[0:4])
df_reg['year']=df_reg['year'].astype('int')

#Remerge foodloss_percentage
df_reg=df_reg.drop(columns=['loss_percentage'])
df_reg=pd.merge(df_reg,df_FAO_pivot,how='inner', left_on=['country','year','commodity','food_supply_stage'],right_on=['country','year','commodity','food_supply_stage'])
df_reg=df_reg.set_index('year')
df_reg=df_reg.drop(columns=['commodity'])

dict_income_group={
    'UPPER-MIDDLE-INCOME ECONOMIES ($4,256 TO $13,205)':'Upper-Middle-Income Economies',
    'LOW-INCOME ECONOMIES ($1,085 OR LESS) ':'Low-Income Economies',
    'LOWER-MIDDLE INCOME ECONOMIES ($1,086 TO $4,255)':'Lower-Middle Income Economies',
    'HIGH-INCOME ECONOMIES ($13,205 OR MORE) ':'High-Income Economies'
}

df_reg['Income']=df_reg['Income'].replace(dict_income_group)

# select Country
selected_country=st.selectbox('Select country',options=df_reg['country'].unique())


# filter df for dependent Income-Group
selected_income_group=df_reg[df_reg['country']==selected_country]['Income'].unique()
selected_income_group=str(selected_income_group)[2:-2]

# filter df for dependent Region
selected_region=df_reg[df_reg['country']==selected_country]['Region'].unique()
selected_region=str(selected_region)[2:-2]


########################################
# Regression
########################################

#### Prepare Data for Income Goup Regressions
def Income_group_data(Income_Group):
    df_income_group=df_reg[df_reg['Income']==f'{Income_Group}']
    return df_income_group



def build_model(model_data):
    X=model_data.drop([
        "loss_percentage",
        'country',
        'Logistics performance index: Overall score (1=low to 5=high)',
        "Logistics performance index: Overall score (1=low to 5=high), lower bound",
        "Logistics performance index: Overall score (1=low to 5=high), upper bound",
        'Region',
        'Income',
        'Commodity Groups',
        'food_supply_stage'
        ], axis=1).values
    y=model_data["loss_percentage"]

    names=model_data.drop([
        "loss_percentage",
        'country',
        'Logistics performance index: Overall score (1=low to 5=high)',
        "Logistics performance index: Overall score (1=low to 5=high), lower bound",
        "Logistics performance index: Overall score (1=low to 5=high), upper bound",
        'Region',
        'Income',
        'Commodity Groups',
        'food_supply_stage'
        ], axis=1).columns

    reg = LinearRegression()
    Regression_model=reg.fit(X,y)

    global df_reg_values
    df_reg_values=pd.DataFrame(list(Regression_model.coef_))
    df_reg_values.index=names

LPI_cat_names=[
    'Ability to track and trace consignments, score (1=low to 5=high)',
    'Competence and quality of logistics services, score (1=low to 5=high)',
    'Ease of arranging competitively priced international shipments, score (1=low to 5=high)',
    'Efficiency of the clearance process, score (1=low to 5=high)',
    'Frequency with which shipments reach consignee within scheduled or expected time, score (1=low to 5=high)',
    'Quality of trade- and transport-related infrastructure, score (1=low to 5=high)'
]

Income_Groups=[
    'Low-Income Economies',
    'Lower-Middle Income Economies',
    'Upper-Middle-Income Economies',
    'High-Income Economies'
    ]

df_reg_income_total=pd.DataFrame()
df_reg_income_total.index=LPI_cat_names
for i in range(len(Income_Groups)):
    reg_data=Income_group_data(Income_Groups[i])
    build_model(reg_data)
    df_reg_values.columns=[Income_Groups[i]]
    df_reg_income_total=df_reg_income_total.join(df_reg_values)


build_model(df_reg[df_reg['country']==selected_country])
df_reg_country=pd.DataFrame()
df_reg_country.index=LPI_cat_names
df_reg_country=df_reg_country.join(df_reg_values)
df_reg_country.columns=[f'{selected_country}']

df_for_Means=df_reg
LPI_cat=[
    'Ability to track and trace consignments, score (1=low to 5=high)',
    'Competence and quality of logistics services, score (1=low to 5=high)',
    'Ease of arranging competitively priced international shipments, score (1=low to 5=high)',
    'Efficiency of the clearance process, score (1=low to 5=high)',
    'Frequency with which shipments reach consignee within scheduled or expected time, score (1=low to 5=high)',
    'Quality of trade- and transport-related infrastructure, score (1=low to 5=high)'
    ]

for i in [0,1,2,3,4,6,7,8,14]:
    df_for_Means.iloc[:,i]=pd.to_numeric(df_for_Means.iloc[:,i])
for i in [9,10,11,12,13]:
    df_for_Means.iloc[:,i]=df_for_Means.iloc[:,i].astype("category")

Income_mean=pd.pivot_table(df_reg,index=['Income'], values=LPI_cat,aggfunc=np.mean, fill_value=None)
Income_mean=Income_mean.drop(index=['Nan'])


df_selected_country=df_for_Means[df_for_Means['country']==selected_country]
Country_mean=pd.pivot_table(df_selected_country,index=['country'], values=LPI_cat,aggfunc=np.mean, fill_value=None)



Diff_Mean_Income= Income_mean.sub(Country_mean.iloc[0, :])
Diff_Mean_Income['Abs_Sum_Mean_Diff']= abs(Diff_Mean_Income.sum(axis = 1))
Diff_Mean_Income=Diff_Mean_Income.sort_values('Abs_Sum_Mean_Diff', ascending=True)



dict_comparison={'Actual Group': [df_selected_country['Income'].value_counts().idxmax()],'Nearest Group': [Diff_Mean_Income.index[0]]}#,df_selected_country['Region'].value_counts().idxmax(),df_selected_country['Commodity Groups'].value_counts().idxmax(),df_selected_country['food_supply_stage'].value_counts().idxmax()], 'Nearest Group': [Diff_Mean_Income.index[0],Diff_Mean_Region.index[0],Diff_Mean_Commodity_group.index[0],Diff_Mean_Supply_stage.index[0]]}
df_comparison=pd.DataFrame(dict_comparison)
df_comparison.index=['Income Group']#, 'Region', 'Commodity Group', 'Food Supply Stage']
df_comparison['is_equal']=(df_comparison['Nearest Group']==df_comparison['Actual Group'])


# with st.expander('Details Regression'):
#     st.write('Datatable with Regression Coef. per Income Group:')
#     st.dataframe(df_reg_income_total)

#     st.write('country regression')
#     st.dataframe(df_reg_country)

# with st.expander('Details Mean Comparison'):
#     st.write('Mean per Income Group')
#     st.dataframe(Income_mean)

#     st.write('Mean of Country')
#     st.dataframe(Country_mean)

#     st.write('Diff. of Means (Country & Income Groups)')
#     st.dataframe(Diff_Mean_Income)

#     st.write('Is Actual Income Group in LPI-Categories also the closest?')
#     st.dataframe(df_comparison)

################
#TEST STATISTICS
################

def perform_Normalitytest(country,actual_income_group, nearest_income_group):
    pval=0.05
    #Country
    df_normality_test_country=pd.DataFrame()
    df_normality_test_country.index=['statistic','pval']
    for i in range(len(LPI_cat)): 
        df_normality_test_country_LPI_cat=pd.DataFrame(stats.shapiro(df_reg[df_reg['country']==country][LPI_cat[i]]))
        df_normality_test_country_LPI_cat.index=['statistic','pval']
        df_normality_test_country_LPI_cat.columns=[f'{LPI_cat[i]}']
        df_normality_test_country=df_normality_test_country.join(df_normality_test_country_LPI_cat,lsuffix='_Country', rsuffix='_Income_group')
    df_normality_test_country=df_normality_test_country.transpose() 
    df_normality_test_country['Normality_check']=(df_normality_test_country['pval']>=pval)

    #Actual Income Group
    df_normality_test_actual_income=pd.DataFrame()
    df_normality_test_actual_income.index=['statistic','pval']
    for i in range(len(LPI_cat)): 
        df_normality_test_actual_income_LPI_cat=pd.DataFrame(stats.shapiro(df_reg[df_reg['Income']==actual_income_group][LPI_cat[i]]))
        df_normality_test_actual_income_LPI_cat.index=['statistic','pval']
        df_normality_test_actual_income_LPI_cat.columns=[f'{LPI_cat[i]}']
        df_normality_test_actual_income=df_normality_test_actual_income.join(df_normality_test_actual_income_LPI_cat,lsuffix='_Country', rsuffix='_Income_group')
    df_normality_test_actual_income=df_normality_test_actual_income.transpose() 
    df_normality_test_actual_income['Normality_check']=(df_normality_test_actual_income['pval']>=pval)

    #Nearest Income Group
    df_normality_test_nearest_income=pd.DataFrame()
    df_normality_test_nearest_income.index=['statistic','pval']
    for i in range(len(LPI_cat)): 
        df_normality_test_nearest_income_LPI_cat=pd.DataFrame(stats.shapiro(df_reg[df_reg['Income']==nearest_income_group][LPI_cat[i]]))
        df_normality_test_nearest_income_LPI_cat.index=['statistic','pval']
        df_normality_test_nearest_income_LPI_cat.columns=[f'{LPI_cat[i]}']
        df_normality_test_nearest_income=df_normality_test_nearest_income.join(df_normality_test_nearest_income_LPI_cat,lsuffix='_Country', rsuffix='_Income_group')
    df_normality_test_nearest_income=df_normality_test_nearest_income.transpose() 
    df_normality_test_nearest_income['Normality_check']=(df_normality_test_nearest_income['pval']>=pval)

    return df_normality_test_country,df_normality_test_actual_income, df_normality_test_nearest_income
   
try:
    df_normality_test_country, df_normality_test_actual_income, df_normality_test_nearest_income =perform_Normalitytest(selected_country, selected_income_group,df_comparison.iloc[0,1])
except:
    df_normality_test_country=[]
    df_normality_test_actual_income=[]
    df_normality_test_nearest_income=[]
    st.info(f'{selected_country} does not have sufficient datapoints to run the model.')


def perform_Ttest(country,actual_income_group, nearest_income_group):
    pval=0.05
    #Actual_Income & Country
    df_ttest_actual_income=pd.DataFrame()
    df_ttest_actual_income.index=['statistic','pval']
    for i in range(len(LPI_cat)): 
        df_ttest_actual_income_LPI_cat=pd.DataFrame(stats.ttest_ind(df_reg[df_reg['country']==country][LPI_cat[i]],df_reg[df_reg['Income']==actual_income_group][LPI_cat[i]],equal_var=False))
        df_ttest_actual_income_LPI_cat.index=['statistic','pval']
        df_ttest_actual_income_LPI_cat.columns=[f'{LPI_cat[i]}']
        df_ttest_actual_income=df_ttest_actual_income.join(df_ttest_actual_income_LPI_cat,lsuffix='_Country', rsuffix='_Income_group')
    df_ttest_actual_income=df_ttest_actual_income.transpose() 
    df_ttest_actual_income['Ttest_check']=(df_ttest_actual_income['pval']<pval)

    #Nearest_Income & Country
    df_ttest_nearest_income=pd.DataFrame()
    df_ttest_nearest_income.index=['statistic','pval']
    for i in range(len(LPI_cat)): 
        df_ttest_nearest_income_LPI_cat=pd.DataFrame(stats.ttest_ind(df_reg[df_reg['country']==country][LPI_cat[i]],df_reg[df_reg['Income']==nearest_income_group][LPI_cat[i]],equal_var=False))
        df_ttest_nearest_income_LPI_cat.index=['statistic','pval']
        df_ttest_nearest_income_LPI_cat.columns=[f'{LPI_cat[i]}']
        df_ttest_nearest_income=df_ttest_nearest_income.join(df_ttest_nearest_income_LPI_cat,lsuffix='_Country', rsuffix='_Income_group')
    df_ttest_nearest_income=df_ttest_nearest_income.transpose() 
    df_ttest_nearest_income['Ttest_check']=(df_ttest_nearest_income['pval']<pval)
    return df_ttest_actual_income, df_ttest_nearest_income

try:
    df_ttest_actual_income, df_ttest_nearest_income =perform_Ttest(selected_country, selected_income_group,df_comparison.iloc[0,1])
except:
    df_ttest_actual_income=[]
    df_ttest_nearest_income=[]


# with st.expander('Details Tests'):
#     st.write('Normality Test Country')
#     st.dataframe(df_normality_test_country)
#     st.write('Normality Test Actual Income Group')
#     st.dataframe(df_normality_test_actual_income)
#     st.write('Normality Test Nearest Income Group')
#     st.dataframe(df_normality_test_nearest_income)
#     st.write('Ttest Actual Income Group')
#     st.dataframe(df_ttest_actual_income)
#     st.write('Ttest Nearest Income Group')
#     st.dataframe(df_ttest_nearest_income)



################
#LPI-category selection
################


Income_group_actual=selected_income_group
nearest_income_group=df_comparison.iloc[0,1]
LPI_cat_order_country=df_reg_country[df_reg_country[f'{selected_country}']<0]


#Auswahl ob Regression country oder Income-group gewählt wird und welche Income Group gewählt wird.
datapoints=len(df_reg[df_reg['country']==selected_country])
if (len(df_reg[df_reg['country']==selected_country])<=5)|(sum(df_reg_country[selected_country])==0):
    if (df_comparison['is_equal'].bool()==True) :
        LPI_cat_order_country=df_reg_income_total[df_reg_income_total[Income_group_actual]<0][Income_group_actual]
        st.info(f'{selected_country} does not have sufficient datapoints to run the model.')
        st.info(f'There are only {datapoints} datapoints for {selected_country} so the data of its income-group:"{Income_group_actual}" was used to determine the policy-areas.')
        # if (sum(df_reg_country[selected_country])==0):
        #     st.info(f'Additional Info: {selected_country}s datapoints were not interpretable for the model, so the Income-group data were used instead.')
    if (df_comparison['is_equal'].bool()==False) & (df_ttest_actual_income['Ttest_check'].sum()>=df_ttest_nearest_income['Ttest_check'].sum()):
        LPI_cat_order_country=df_reg_income_total[df_reg_income_total[nearest_income_group]<0][nearest_income_group]
        st.info(f'{selected_country} does not have sufficient datapoints to run the model.')
        st.info(f'There are only {datapoints} datapoints for {selected_country} so the data of its closest income-group:"{nearest_income_group}" was used to determine the policy-areas. The actual Income-group is: {Income_group_actual}.')
        # if (sum(df_reg_country[selected_country])==0):
        #     st.info(f'Additional Info: {selected_country}s datapoints were not interpretable for the model, so the Income-group data were used instead.')

LPI_cat_order_country=pd.DataFrame(LPI_cat_order_country)
LPI_cat_order_country.columns=[f'{selected_country}']


LPI_cat_order_country=LPI_cat_order_country.sort_values(f'{selected_country}',ascending=False)
LPI_cat_order_country['weight']=np.arange(len(LPI_cat_order_country))+1
LPI_cat_order_country['weight']=pd.to_numeric(LPI_cat_order_country['weight'])
LPI_cat_order_country=LPI_cat_order_country.sort_values(by=['weight'], ascending = False)
LPI_cat_order_country=LPI_cat_order_country.reset_index()
LPI_cat_order_country.columns=['LPI-Category ordered by Priority (highest first)','Regression coefficent', 'Priority']

LPI_cat_name={
    'Ability to track and trace consignments, score (1=low to 5=high)':'Tracking and Tracing',
    'Competence and quality of logistics services, score (1=low to 5=high)':'Quality of log services',
    'Ease of arranging competitively priced international shipments, score (1=low to 5=high)':'Ease of arranging shipments',
    'Efficiency of the clearance process, score (1=low to 5=high)':'Customs',
    'Frequency with which shipments reach consignee within scheduled or expected time, score (1=low to 5=high)':'Timeliness',
    'Quality of trade- and transport-related infrastructure, score (1=low to 5=high)':'Infrastructure'
    }
LPI_cat_order_country=LPI_cat_order_country.replace({'LPI-Category ordered by Priority (highest first)':LPI_cat_name})
# with st.expander('Details LPI-categories'):
#     st.write('Identified policy areas to reduce Food Loss')
#     st.dataframe(LPI_cat_order_country['LPI-Category ordered by Priority (highest first)'])







################
#Factor Analysis
################
data_raw = pd.read_csv('data/CFA_Policies.csv',sep=';')
data=data_raw
LPI_categories=data['LPI-category'].unique()

#Functions to avoid singular matrix error
mu=0.0
std = 0.1
np.random.seed(21389712) #repeatable random number
def gaussian_noise(x):
    noise = np.random.normal(mu, std, size = x.shape)
    x_noisy = x + noise
    return x_noisy 

def make_nonsingular(matrix):
  n = matrix.shape[0]
  m = matrix.shape[1]
  for i in range(n):
    for j in range(m):
      matrix.iloc[i, j] = gaussian_noise(matrix.iloc[i, j])
  return matrix


################
#confirmatory factor analysis
################
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer


scale1= MinMaxScaler()
scale2= StandardScaler()
scale3=Normalizer()
# standardization of dependent variables

data= data.iloc[:,[2,3,4,5,6]]
data= data.T
data=make_nonsingular(data)
data= scale1.fit_transform(data)

model_dict = {"F1": [0,1,2,3,4,5,6,7],
              "F2": [8,9,10],
              "F3": [11,12,13,14,15,16],
              "F4": [17,18,19,20,21,22,23,24],
              "F5": [25,26,27,28,29,30],
              "F6": [31]}
model_spec = ModelSpecificationParser.parse_model_specification_from_dict(data,model_dict)

cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)

cfa.fit(data)

Factor_loading=pd.DataFrame(cfa.loadings_)
Factor_loading['factor_loadings']=Factor_loading.replace(0,np.nan).max(1)

#Prüfkriterien
def factor_analysis(LPI_cat):
    LPI_category_data=data_raw[data_raw['LPI-category']==LPI_cat]
    X=LPI_category_data[['survey_1','survey_2','survey_3','survey_4','survey_5']]
    
    # with st.expander(f'See the factoranalysis insights of {LPI_cat}'):
    #     st.write(f"{LPI_cat}: Chi-value, p-value ", calculate_bartlett_sphericity(X))
    #     st.write(f"{LPI_cat}: Kaiser-Meyer-Olkin criterion per variable, in total ", calculate_kmo(X))

    return

for i in range(len(LPI_categories)-1):
    factor_analysis(LPI_categories[i])


data_raw['factor_loading']=Factor_loading['factor_loadings']

st.write(f"The policies are selected by a linear Regression with Data from the ['Food Loss and Waste Database'](https://www.fao.org/platform-food-loss-waste/flw-data/en/) & ['Logistics Performance index'](https://lpi.worldbank.org/) and a Q-Methodolgy to integrate expert knowledge.")
st.write("Explore the **analysis-tools** linked in **sidebar on the left** for additional insights on the datasets used.")
st.write("---")
st.subheader(f'Logistics policies to reduce food loss in {selected_country}')

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)


for i in range(len(LPI_cat_order_country['LPI-Category ordered by Priority (highest first)'])):
    LPI_cat = LPI_cat_order_country['LPI-Category ordered by Priority (highest first)'][i]
    data_LPI_cat = data_raw[data_raw['LPI-category']==LPI_cat]
    data_LPI_cat = pd.DataFrame(data_LPI_cat).sort_values(by=['factor_loading'],ascending=False)
    data_LPI_cat = data_LPI_cat.fillna(np.nan)
    if i ==0: 
        st.write(f'**1st Priority** are Policies related to the LPI-Category: **{LPI_cat}**.')
    if i ==1: 
        st.write(f'**2st Priority** are Policies related to the LPI-Category: **{LPI_cat}**.')
    if i ==2: 
        st.write(f'**3st Priority** are Policies related to the LPI-Category: **{LPI_cat}**.')
    if i ==3: 
        st.write(f'**4st Priority** are Policies related to the LPI-Category: **{LPI_cat}**.')
    if i ==4: 
        st.write(f'**5st Priority** are Policies related to the LPI-Category: **{LPI_cat}**.')
    if i ==5: 
        st.write(f'**6st Priority** are Policies related to the LPI-Category: **{LPI_cat}**.')
    st.table(data_LPI_cat[['Policy']].rename({'Policy': 'Policies (highest priority first)'}, axis='columns'))#.set_index('Policy'))
    with st.expander('See aditionally benefits and risks for selected policies.'):
        st.dataframe(data_LPI_cat[['Policy','Benefits','Risks']].set_index('Policy').dropna())


st.write("by [Jannik Schäffer](https://www.linkedin.com/in/jannik-sch%C3%A4ffer)")
