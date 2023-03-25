########################################
# Import libraries
########################################
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression

########################################
# Configure Page
########################################
st.set_page_config(page_title="Country Food Loss Policy Assistant", page_icon="🥑",layout="wide")

########################################
# Add Title
########################################
st.header("Country Food Loss Policy Assistant")
st.sidebar.header("Country Food Loss Policy Assistant")

########################################
# Read Data
########################################
# LPI Data
df_LPI= pd.read_csv("data/LPI_Data.csv", encoding='latin-1')
# FAO Data
df_FAO = pd.read_csv("data/Data.csv")
# Country Categories by Worldbank 
country_categories_tidy= pd.read_csv('data/country_categories_tidy.csv', delimiter=';')
# Country Categories by Worldbank 
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
# Drop duplicates & name change
df=df_FAO.drop_duplicates()

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

#change name Back
df_FAO=df

#align country names for later merge
def Change_Country_Names( df_new_names,column_new_names ,df_to_change,column_name):
    name_dict=df_new_names.set_index(column_new_names).T.to_dict('list')
    df_to_change[column_name]=df_to_change[column_name].map(name_dict)
    df_to_change[column_name]=df_to_change[column_name].str.get(0)

Change_Country_Names(df_LPI_new_country_names,'LPI Countries',df_LPI, 'Country Name' )
#Change_Country_Names(df_FAO_new_country_names,'FAO Countries',df_FAO, 'country' )

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

## Add Country Categories to the Dataframe
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

#select Series
dict_series={
    'Ability to track and trace consignments, score (1=low to 5=high)':0,
    'Competence and quality of logistics services, score (1=low to 5=high)':1,
    'Ease of arranging competitively priced international shipments, score (1=low to 5=high)':2,
    'Efficiency of the clearance process, score (1=low to 5=high)':3,
    'Frequency with which shipments reach consignee within scheduled or expected time, score (1=low to 5=high)':4,
    'Logistics performance index: Overall score (1=low to 5=high)':5,
    'Logistics performance index: Overall score (1=low to 5=high), lower bound':6,
    'Logistics performance index: Overall score (1=low to 5=high), upper bound':7,
    'Quality of trade- and transport-related infrastructure, score (1=low to 5=high)':8,
}

########################################
# Test Statistic
########################################
# st.subheader('Test Statistic')

# selected_series=st.selectbox('Select series',options=dict_series.keys())



# test_df_1=df_reg[df_reg['country']==selected_country]
# test_df_2=df_reg[(df_reg['Income']==f'{selected_income_group}')]#& (df_reg['country']!=selected_country)]
# test_df_3=df_reg[(df_reg['Region']==f'{selected_region}')]#& (df_reg['country']!=selected_country)]

# test_data_1=test_df_1.iloc[:,dict_series[selected_series]]
# test_data_1=test_data_1.astype('float')

# test_data_2=test_df_2.iloc[:,dict_series[selected_series]]
# test_data_2=test_data_2.astype('float')

# test_data_3=test_df_3.iloc[:,dict_series[selected_series]]
# test_data_3=test_data_3.astype('float')


# #describe Dataframes
# df_describe=pd.DataFrame(test_data_1.describe())
# df_describe=df_describe.join(pd.DataFrame(test_data_2.describe()),lsuffix='_Country', rsuffix='_Income_group')
# df_describe=df_describe.join(pd.DataFrame(test_data_3.describe()),lsuffix='_Country', rsuffix='_Region_group')
# df_describe.columns=['Country','Income Group','Region']
# with st.expander('See Description of Selected Data'):
#     col1, col2 = st.columns(2)
#     with col1: 
#         st.write('Data Description')
#         st.dataframe(df_describe)
#     with col2: 
#         st.write('Histograms')
#         tab1, tab2, tab3 = st.tabs(["Country", "Income Group", "Region"])
#         with tab1:
#             fig1 = px.histogram(test_data_1,nbins=20, x=selected_series, labels={selected_series:'Country Data Distribution', 'y':'count'})
#             fig1.update_layout({
#                 'plot_bgcolor': 'rgba(0,0,0,0)',
#                 'paper_bgcolor': 'rgba(0,0,0,0)'
#                 })
#             st.plotly_chart(fig1, use_container_width=True)
#         with tab2:
#             fig2 = px.histogram(test_data_2,nbins=20, x=selected_series, labels={selected_series:'Income Group Data Distribution', 'y':'count'})
#             fig2.update_layout({
#                 'plot_bgcolor': 'rgba(0,0,0,0)',
#                 'paper_bgcolor': 'rgba(0,0,0,0)'
#                 })
#             st.plotly_chart(fig2, use_container_width=True)
#         with tab3:
#             fig3 = px.histogram(test_data_3,nbins=20, x=selected_series, labels={selected_series:'Region Data Distribution', 'y':'count'})
#             fig3.update_layout({
#                 'plot_bgcolor': 'rgba(0,0,0,0)',
#                 'paper_bgcolor': 'rgba(0,0,0,0)'
#                 })
#             st.plotly_chart(fig3, use_container_width=True)


# #Normality Test results
# normality_test_results=pd.DataFrame(stats.shapiro(test_data_1))
# normality_test_results=normality_test_results.join(pd.DataFrame(stats.shapiro(test_data_2)),lsuffix='_Country', rsuffix='_Income_group')
# normality_test_results=normality_test_results.join(pd.DataFrame(stats.shapiro(test_data_3)),lsuffix='_Country', rsuffix='_Region_group')
# normality_test_results.index=['statistic','pvalue']
# normality_test_results=normality_test_results.transpose()
# normality_test_results.index=['Country','Income Group','Region']
# normality_test_results['Normality condition']=(normality_test_results['pvalue'] >= 0.000005)
# # print(normality_test_results)
# col3, col4 = st.columns(2)
# with col3:
#     st.markdown('**Normality Test Results**')
#     st.dataframe(normality_test_results)
#     with st.expander('See Test Hypothesis'):
#         st.markdown(f'''
#             **H0**: Die Grundgesamtheit von Land/Income-Group/Region sind normalverteilt.

#             **H1**: Die Grundgesamtheit von Land/Income-Group/Region sind nicht normalverteilt. 
#         ''')


# #T Test results
# Ttest_results=pd.DataFrame(stats.ttest_ind(test_data_1, test_data_2, equal_var=False))
# Ttest_results=Ttest_results.join(pd.DataFrame(stats.ttest_ind(test_data_1, test_data_3, equal_var=False)),lsuffix='_Country/Region', rsuffix='_Country/Income')
# Ttest_results.index=['statistic','pvalue']
# Ttest_results=Ttest_results.transpose()
# Ttest_results.index=['Country/Region','Country/Income Group']
# Ttest_results['Ttest Check (pval<0.05)']=(Ttest_results['pvalue']<= 0.05)
# # print(Ttest_results)
# with col4:
#     st.markdown('**T-Test Results**')
#     st.dataframe(Ttest_results)
#     st.write('')
#     st.write('')
#     with st.expander('See Test Hypothesis'):
#         st.markdown(f'''
#         **H0**: Es gibt **keinen** Mittelwertsunterschied zwischen gewähltem Land und dessen zugehöriger Income-Gruppe/Region.

#         **H1**: Es gibt **einen** Mittelwertsunterschied zwischen gewähltem Land und dessen zugehöriger Income-Gruppe/Region. 
#         ''')
# st.markdown(f'''
#         Sind alle Vorraussetzungen des T-Tests erfüllt (inkl. *Normalverteilung der Stichproben*) kann folgende Aussage getroffen werden.

#         Ist der p-Wert (pvalue) kleiner 0,05 , dem festgelegten Signifikanzlevel, so kann die Null-Hypothese abgelehnt werden.
        
#         In diesem Fall existiert ein Mittelwertsunterschied zwischen dem gewählten Land (*{selected_country}*) und seiner Income-Group(*{selected_income_group}*) /Region(*{selected_region}*)''')


########################################
# Create Model
########################################
# def build_model_visualisation(model_data):
#     X=model_data.drop([
#         "loss_percentage",
#         'country',
#         'Logistics performance index: Overall score (1=low to 5=high)',
#         "Logistics performance index: Overall score (1=low to 5=high), lower bound",
#         "Logistics performance index: Overall score (1=low to 5=high), upper bound",
#         'Region',
#         'Income',
#         'Commodity Groups',
#         'food_supply_stage'
#         ], axis=1).values
#     y=model_data["loss_percentage"]
#     #X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=21, stratify=y)
#     names=model_data.drop([
#         "loss_percentage",
#         'country',
#         'Logistics performance index: Overall score (1=low to 5=high)',
#         "Logistics performance index: Overall score (1=low to 5=high), lower bound",
#         "Logistics performance index: Overall score (1=low to 5=high), upper bound",
#         'Region',
#         'Income',
#         'Commodity Groups',
#         'food_supply_stage'
#         ], axis=1).columns

#     reg = LinearRegression()
#     Regression_model=reg.fit(X,y)


#     #print(Regression_model.intercept_)
#     bar_chart_df=pd.DataFrame(list(Regression_model.coef_))
#     bar_chart_df.index=names
#     bar_chart_df.columns=['Coefficient Value']
#     st.bar_chart(bar_chart_df)

# st.subheader('Linear Regression')
# col5,col6,col7=st.columns(3)
# with col5:
#     st.write(f'''
#     **{selected_country}** 
#     specific models coefficients
#     ''')
#     build_model_visualisation(df_reg[df_reg['country']==selected_country])
# with col6:
#     st.write(f'''
#     **{selected_income_group}** 
#     specific models coefficients
#     ''')

#     build_model_visualisation(df_reg[(df_reg['Income']==f'{selected_income_group}')])
# with col7:
#     st.write(f'''
#     **{selected_region}** 
#     specific models coefficients
#     ''')
#     build_model_visualisation(df_reg[(df_reg['Region']==f'{selected_region}')])

# st.subheader('Income Groups Linear Regression')

#### Prepare Data for Income Goup Regressions
def Income_group_data(Income_Group):
    df_income_group=df_reg[df_reg['Income']==f'{Income_Group}']
    return df_income_group


# col8,col9,col10,col11=st.columns(4)
# with col8:
#     st.write(f'''
#     **Low Income** 
#     specific models coefficients
#     ''')
#     Income_Group='Low-Income Economies'
#     reg_data=Income_group_data(Income_Group)
#     build_model_visualisation(reg_data)

# with col9:
#     st.write(f'''
#     **Lower-Middle Income** 
#     specific models coefficients
#     ''')
#     Income_Group='Lower-Middle Income Economies'
#     reg_data=Income_group_data(Income_Group)
#     build_model_visualisation(reg_data)

# with col10:
#     st.write(f'''
#     **Upper-Middle Income** 
#     specific models coefficients
#     ''')
#     Income_Group='Upper-Middle-Income Economies'
#     reg_data=Income_group_data(Income_Group)
#     build_model_visualisation(reg_data)

# with col11:
#     st.write(f'''
#     **High Income** 
#     specific models coefficients
#     ''')
#     Income_Group='High-Income Economies'
#     reg_data=Income_group_data(Income_Group)
#     build_model_visualisation(reg_data)


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
    #X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=21, stratify=y)
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


    #print(Regression_model.intercept_)
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
st.write('Datatable with Regression Coef. per Income Group:')
st.dataframe(df_reg_income_total)

build_model(df_reg[df_reg['country']==selected_country])
df_reg_country=pd.DataFrame()
df_reg_country.index=LPI_cat_names
df_reg_country=df_reg_country.join(df_reg_values)
df_reg_country.columns=[f'{selected_country}']
st.write('country regression')
st.dataframe(df_reg_country)
# #### Prepare Data for Commodity Group Regressions
# def Commodity_group_data(Commodity_Group):
#     df_Commodity_group=df_reg[df_reg['Commodity Groups']==f'{Commodity_Group}']
#     return df_Commodity_group

# commodity_groups=[
#     'Cereals (excluding beer)',
#     'Roots and Tubers',
#     'Oilseeds and Pulses (including nuts)',
#     'Fruit and Vegetables (including bananas)',
#     'Meat',
#     'Dairy products'
#     ]

# df_reg_commodity_total=pd.DataFrame()
# df_reg_commodity_total.index=LPI_cat_names
# for i in range(len(commodity_groups)):
#     reg_data=Commodity_group_data(commodity_groups[i])
#     build_model(reg_data)
#     df_reg_values.columns=[commodity_groups[i]]
#     df_reg_commodity_total=df_reg_commodity_total.join(df_reg_values)
# st.write('Datatable with Regression Coef. per Commodity Group:')
# st.dataframe(df_reg_commodity_total)

# #### Prepare Data for Region  Regressions

# def Region_group_data(Region):
#     df_Region_group=df_reg[df_reg['Region']==f'{Region}']
#     return df_Region_group

# Region_groups=[
#     'SOUTH ASIA',
#     'SUB-SAHARAN AFRICA',
#     'EUROPE AND CENTRAL ASIA',
#     'NORTH AMERICA',
#     'MIDDLE EAST AND NORTH AFRICA',
#     'EAST ASIA AND PACIFIC',
#     'LATIN AMERICA AND THE CARIBBEAN',
# ]

# df_reg_Region_total=pd.DataFrame()
# df_reg_Region_total.index=LPI_cat_names
# for i in range(len(Region_groups)):
#     reg_data=Region_group_data(Region_groups[i])
#     build_model(reg_data)
#     df_reg_values.columns=[Region_groups[i]]
#     df_reg_Region_total=df_reg_Region_total.join(df_reg_values)
# st.write('Datatable with Regression Coef. per Region:')
# st.dataframe(df_reg_Region_total)




# #### Prepare Data for Supply Stage  Regressions
# def food_supply_stage_data(food_supply_stage):
#     df_food_supply_stage=df_reg[df_reg['food_supply_stage']==f'{food_supply_stage}']
#     return df_food_supply_stage

# food_supply_stage= [
#     'Whole supply chain',
#     #'Pre-harvest',
#     'Harvest',
#     'Post-harvest',
#     'Farm',
#     #'Grading',
#     #'Stacking',
#     'Storage',
#     'Transport',
#     #'Distribution',
#     'Processing',
#     #'Packing',
#     'Wholesale',
#     #'Export',
#     'Trader',
#     #'Market',
#     'Retail',
#     'Food Services',
#     'Households',    
# ]

# df_reg_food_supply_stage_total=pd.DataFrame()
# df_reg_food_supply_stage_total.index=LPI_cat_names
# for i in range(len(food_supply_stage)):
#     reg_data=food_supply_stage_data(food_supply_stage[i])
#     build_model(reg_data)
#     df_reg_values.columns=[food_supply_stage[i]]
#     df_reg_food_supply_stage_total=df_reg_food_supply_stage_total.join(df_reg_values)
# st.write('Datatable with Regression Coef. per Supply Chain stage:')
# st.dataframe(df_reg_food_supply_stage_total)



########################################
# Create MEAN Comparisons
########################################

# Country: selected_country

# test_df_1=df_reg[df_reg['country']==selected_country]
# test_df_2=df_reg[(df_reg['Income']==f'{selected_income_group}')]#& (df_reg['country']!=selected_country)]
# test_df_3=df_reg[(df_reg['Region']==f'{selected_region}')]#& (df_reg['country']!=selected_country)]

df_for_Means=df_reg#[df_reg.index==max(df_reg.index)]
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
# Region_mean=pd.pivot_table(df_reg,index=['Region'], values=LPI_cat,aggfunc=np.mean, fill_value=None)
# Commodity_group_mean=pd.pivot_table(df_reg,index=['Commodity Groups'], values=LPI_cat,aggfunc=np.mean, fill_value=None)
# Supply_stage_mean=pd.pivot_table(df_reg,index=['food_supply_stage'], values=LPI_cat,aggfunc=np.mean, fill_value=None)

st.write('Mean per Income Group')
st.dataframe(Income_mean)
# st.dataframe(Region_mean)
# st.dataframe(Commodity_group_mean)
# st.dataframe(Supply_stage_mean)
# test3=pd.concat([Income_mean,Region_mean,Commodity_group_mean,Supply_stage_mean])
# st.dataframe(test3)

df_selected_country=df_for_Means[df_for_Means['country']==selected_country]
Country_mean=pd.pivot_table(df_selected_country,index=['country'], values=LPI_cat,aggfunc=np.mean, fill_value=None)
st.write('Mean of Country')
st.dataframe(Country_mean)

st.write('Diff. of Means (Country & Income Groups)')
Diff_Mean_Income= Income_mean.sub(Country_mean.iloc[0, :])
Diff_Mean_Income['Abs_Sum_Mean_Diff']= abs(Diff_Mean_Income.sum(axis = 1))
Diff_Mean_Income=Diff_Mean_Income.sort_values('Abs_Sum_Mean_Diff', ascending=True)
st.dataframe(Diff_Mean_Income)

# Diff_Mean_Region= Region_mean.sub(Country_mean.iloc[0, :])
# Diff_Mean_Region['Abs_Sum_Mean_Diff']= abs(Diff_Mean_Region.sum(axis = 1))
# Diff_Mean_Region=Diff_Mean_Region.sort_values('Abs_Sum_Mean_Diff', ascending=True)
# st.dataframe(Diff_Mean_Region)

# Diff_Mean_Commodity_group= Commodity_group_mean.sub(Country_mean.iloc[0, :])
# Diff_Mean_Commodity_group['Abs_Sum_Mean_Diff']= abs(Diff_Mean_Commodity_group.sum(axis = 1))
# Diff_Mean_Commodity_group=Diff_Mean_Commodity_group.sort_values('Abs_Sum_Mean_Diff', ascending=True)
# st.dataframe(Diff_Mean_Commodity_group)

# Diff_Mean_Supply_stage= Supply_stage_mean.sub(Country_mean.iloc[0, :])
# Diff_Mean_Supply_stage['Abs_Sum_Mean_Diff']= abs(Diff_Mean_Supply_stage.sum(axis = 1))
# Diff_Mean_Supply_stage=Diff_Mean_Supply_stage.sort_values('Abs_Sum_Mean_Diff', ascending=True)
# st.dataframe(Diff_Mean_Supply_stage)



dict_comparison={'Actual Group': [df_selected_country['Income'].value_counts().idxmax()],'Nearest Group': [Diff_Mean_Income.index[0]]}#,df_selected_country['Region'].value_counts().idxmax(),df_selected_country['Commodity Groups'].value_counts().idxmax(),df_selected_country['food_supply_stage'].value_counts().idxmax()], 'Nearest Group': [Diff_Mean_Income.index[0],Diff_Mean_Region.index[0],Diff_Mean_Commodity_group.index[0],Diff_Mean_Supply_stage.index[0]]}
df_comparison=pd.DataFrame(dict_comparison)
df_comparison.index=['Income Group']#, 'Region', 'Commodity Group', 'Food Supply Stage']
df_comparison['is_equal']=(df_comparison['Nearest Group']==df_comparison['Actual Group'])
st.write('Is Actual Income Group in LPI-Categories also the closest?')
st.dataframe(df_comparison)


################
#TEST STATISTICS
################
st.write('Tests')

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
   

df_normality_test_country, df_normality_test_actual_income, df_normality_test_nearest_income =perform_Normalitytest(selected_country, selected_income_group,df_comparison.iloc[0,1])
st.write('Normality Test Country')
st.dataframe(df_normality_test_country)
st.write('Normality Test Actual Income Group')
st.dataframe(df_normality_test_actual_income)
st.write('Normality Test Nearest Income Group')
st.dataframe(df_normality_test_nearest_income)


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

df_ttest_actual_income, df_ttest_nearest_income =perform_Ttest(selected_country, selected_income_group,df_comparison.iloc[0,1])
st.write('Ttest Actual Income Group')
st.dataframe(df_ttest_actual_income)
st.write('Ttest Nearest Income Group')
st.dataframe(df_ttest_nearest_income)





#st.dataframe(df_policies)
Income_group_actual=selected_income_group
LPI_cat_order_country=df_reg_country[df_reg_country[f'{selected_country}']<0]
LPI_cat_order_country=LPI_cat_order_country.sort_values(f'{selected_country}',ascending=False)
LPI_cat_order_country=pd.DataFrame(LPI_cat_order_country)
LPI_cat_order_country['Weights']=np.arange(len(LPI_cat_order_country))+1
LPI_cat_order_country['Weights']=pd.to_numeric(LPI_cat_order_country['Weights'])
LPI_cat_order_country=LPI_cat_order_country.reset_index()
LPI_cat_order_country.columns=['category','Regression_Coef', 'weight']
st.write('LPI_cat_order_country')
st.dataframe(LPI_cat_order_country)

LPI_cat_order=df_reg_income_total[df_reg_income_total[Income_group_actual]<0][Income_group_actual]
LPI_cat_order=LPI_cat_order.sort_values(ascending=False)
LPI_cat_order=pd.DataFrame(LPI_cat_order)
LPI_cat_order['Weights']=np.arange(len(LPI_cat_order))+1
LPI_cat_order['Weights']=pd.to_numeric(LPI_cat_order['Weights'])
LPI_cat_order=LPI_cat_order.reset_index()
LPI_cat_order.columns=['category','Regression_Coef', 'weight']
selected_Commodity_Group = df_selected_country['Commodity Groups'].value_counts().idxmax()
selected_supply_stage = df_selected_country['food_supply_stage'].value_counts().idxmax()
df_cat_weights=LPI_cat_order[['category','weight']]
df_cat_weights=df_cat_weights.append(LPI_cat_order_country[['category','weight']])
df_cat_weights.loc[len(df_cat_weights.index)]=[selected_region, 1]
df_cat_weights.loc[len(df_cat_weights.index)]=[selected_supply_stage, 1]
df_cat_weights.loc[len(df_cat_weights.index)]=[selected_Commodity_Group, 1]
df_cat_weights=df_cat_weights.groupby(['category']).sum()
df_cat_weights=df_cat_weights.reset_index()
st.dataframe(df_cat_weights)
# st.dataframe(df_policies)

df_policies_test=df_policies
for i in [0,4,6-38]:
    df_policies_test.iloc[:,i]=pd.to_numeric(df_policies_test.iloc[:,i])

for i in range(len(df_cat_weights)):
    df_policies_test[df_cat_weights.iloc[i,0]]+=df_cat_weights.iloc[i,1]

df_policies_test=df_policies_test.replace(1,0)
df_policies_test['Total weight']=df_policies_test.iloc[:,4:38].sum(axis = 1)
df_policies_test=df_policies_test.sort_values('Total weight', ascending=False)


df_policy_recommendation=df_policies_test[['Policy Short', 'Policy Component', 'Policy Area', 'Total weight']]
st.dataframe(df_policy_recommendation)





