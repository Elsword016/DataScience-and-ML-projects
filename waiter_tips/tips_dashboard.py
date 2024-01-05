from hmac import new
from cv2 import exp
from matplotlib import category
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import os
import sys
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
import plotly_express as px
import plotly.figure_factory as ff
import time
import shap
from streamlit_shap import st_shap
sns.set_style('ticks')
sns.set_context('paper')

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_title="Tips Dashboard",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(r'D:\portfolio\tips.csv')
    return df
df = load_data()

# Title
st.title("Tips Dashboard- Regression analysis")
st.write(df)
# plot the data
st.subheader("Data Visualization")
fig1,fig2 = st.columns(2)
with fig1:
    st.markdown("### Tip by gender")
    fig = px.box(y='tip',x='sex',data_frame=df,hover_data=df.columns,color='sex')
    st.plotly_chart(fig)
with fig2:
    st.markdown("### Tip by smoker")
    fig = px.box(y='tip',x='smoker',data_frame=df,hover_data=df.columns,color='sex')
    st.plotly_chart(fig)

fig3,fig4 = st.columns(2)
with fig3:
    st.markdown("### Tip by day")
    fig = px.box(y='tip',x='day',color='day',data_frame=df,hover_data=df.columns)
    st.plotly_chart(fig)
with fig4:
    st.markdown("### Tip by time")
    fig = px.box(y='tip',x='time',color='time',data_frame=df,hover_data=df.columns)
    st.plotly_chart(fig)

fig5,fig6 = st.columns(2)
with fig5:
    st.markdown("### Tip distribution")
    group_labels = ['tip']
    fig = ff.create_distplot([df['tip']],group_labels,show_hist=False,show_rug=False)
    #change y axis title
    fig.update_layout(yaxis_title='Density')
    st.plotly_chart(fig)

with fig6:
    st.markdown("### Total bill distribution")
    group_labels = ['total_bill']
    fig = ff.create_distplot([df['total_bill']],group_labels,show_hist=False,show_rug=False)
    #change y axis title
    fig.update_layout(yaxis_title='Density')
    st.plotly_chart(fig)

st.markdown("### Correct skewness")
fig7,fig8 = st.columns(2)
with fig7:
    st.markdown("### Tip distribution")
    group_labels = ['tip']
    fig = ff.create_distplot([np.log(df['tip'])],group_labels,show_hist=False,show_rug=False)
    #change y axis title
    fig.update_layout(yaxis_title='Density')
    st.plotly_chart(fig)

with fig8:
    st.markdown("### Total bill distribution")
    group_labels = ['total_bill']
    fig = ff.create_distplot([np.log(df['total_bill'])],group_labels,show_hist=False,show_rug=False)
    #change y axis title
    fig.update_layout(yaxis_title='Density')
    st.plotly_chart(fig)

# Data preprocessing
df['tip'] = np.log(df['tip'])
df['total_bill'] = np.log(df['total_bill'])

# Label encoding
df['sex'] = df['sex'].map({'Female': 0, 'Male':1})
df['smoker'] = df['smoker'].map({'No': 0, 'Yes':1})
df['day'] = df['day'].map({'Sun': 0, 'Sat':1, 'Thur':2, 'Fri':3})
df['time'] = df['time'].map({'Dinner':0, 'Lunch':1})

#data preprocessing
st.subheader("Data preprocessing")
st.write("we first convert the categorical data into numerical data using label encoding and then we correct the skewness of the data using log transformation")
st.write("Corrected dataframe:")
st.write(df)

# Split the data
X_train,X_test,y_train,y_test = train_test_split(df.drop('tip',axis=1),df['tip'],test_size=0.2,random_state=42)
st.subheader("Split the data")
st.markdown(f"We split the data into train and test set with test size of 20% and random state of 42")
st.write("X_train:",X_train.shape)
st.write("y_train:",y_train.shape)
st.write("X_test:",X_test.shape)
st.write("y_test:",y_test.shape)

#models
models = {'Linear Regression': LinearRegression(),
          'Random Forest Regressor': RandomForestRegressor(),
          'XGBoost Regressor': XGBRegressor(),
          'Gradient Boosting Regressor': GradientBoostingRegressor()}
mae = []
mse = []
r2 = []
def build_model(model):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    st.write("Mean squared error:",mean_squared_error(y_test,y_pred))
    st.write("Mean absolute error:",mean_absolute_error(y_test,y_pred))
    st.write("R2 score:",r2_score(y_test,y_pred))
    return model

# Build the model
st.subheader("Build the model")
st.write("We build the model using 3 different algorithms and compare their performance")
ml_model = st.selectbox("Select the model",list(models.keys()))
model = models[ml_model]
# add a loading bar and then display the model performance after 2 seconds
with st.spinner("Training the model..."):
    time.sleep(3)
model = build_model(model)

lr = LinearRegression()
rf = RandomForestRegressor()
xgb = XGBRegressor()
gb = GradientBoostingRegressor()

for i in models.values():
    i.fit(X_train,y_train)
    y_pred = i.predict(X_test)
    mae.append(mean_absolute_error(y_test,y_pred))
    mse.append(mean_squared_error(y_test,y_pred))
    r2.append(r2_score(y_test,y_pred))

# Model performance
mae = pd.DataFrame(mae,index=models.keys(),columns=['MAE'])
mse = pd.DataFrame(mse,index=models.keys(),columns=['MSE'])
r2 = pd.DataFrame(r2,index=models.keys(),columns=['R2'])
df_metrics = pd.concat([mae,mse,r2],axis=1)

model_tab,model_summ = st.columns(2,gap='medium')
with model_tab:
    st.markdown("### Model performance summary")
    #add some space
    st.write("")
    st.write("")
    st.table(df_metrics)
    st.write("We can see that the XGBoost Regressor performs the best with the lowest MAE and MSE and highest R2 score")
with model_summ:
    st.markdown("### Model performance")
    fig = px.histogram(df_metrics,x=df_metrics.index,y=['MAE','MSE','R2'],barmode='group',color_discrete_sequence=['#636EFA','#EF553B','#00CC96'],title='Model performance')
    st.plotly_chart(fig)

st.subheader("Feature importances from different models")
fig10,fig11 = st.columns(2)
with fig10:
    lr.fit(X_train,y_train)
    coef = lr.coef_
    fig = px.bar(y=X_train.columns,x=coef,color=coef,orientation='h',color_continuous_scale='RdBu',title='Linear Regression feature importance')
    #scale the x axis
    fig.update_xaxes(range=[-1,1])
    fig.update_layout(xaxis_title='Coefficient')
    fig.update_layout(yaxis_title='Features')
    st.plotly_chart(fig)
with fig11:
    rf.fit(X_train,y_train)
    coef = rf.feature_importances_
    fig = px.bar(y=X_train.columns,x=coef,color=coef,orientation='h',color_continuous_scale='RdBu',title='Random Forest Regressor feature importance')
    #scale the x axis
    fig.update_xaxes(range=[0,1])
    fig.update_layout(xaxis_title='Coefficient')
    fig.update_layout(yaxis_title='Features')
    st.plotly_chart(fig)

fig12,fig13 = st.columns(2)
with fig12:
    xgb.fit(X_train,y_train)
    coef = xgb.feature_importances_
    fig = px.bar(y=X_train.columns,x=coef,color=coef,orientation='h',color_continuous_scale='RdBu',title='XGBoost Regressor feature importance')
    #scale the x axis
    fig.update_xaxes(range=[0,1])
    fig.update_layout(xaxis_title='Coefficient')
    fig.update_layout(yaxis_title='Features')
    st.plotly_chart(fig)
with fig13:
    gb.fit(X_train,y_train)
    coef = gb.feature_importances_
    fig = px.bar(y=X_train.columns,x=coef,color=coef,orientation='h',color_continuous_scale='RdBu',title='Gradient Boosting Regressor feature importance')
    #scale the x axis
    fig.update_xaxes(range=[0,1])
    fig.update_layout(xaxis_title='Coefficient')
    fig.update_layout(yaxis_title='Features')
    st.plotly_chart(fig)

st.markdown('**All the models gave total_bill as the most important feature**')

st.subheader("Model explainability with SHAP")
st.write("We use SHAP to explain the model predictions")
explainer = shap.Explainer(xgb)
shap_values = explainer(X_test)
st.write(f"SHAP values shape: {shap_values.shape}")
st.write(f"X_test shape: {X_test.shape}")

# SHAP plot
sh1,sh2 = st.columns(2)
with sh1:
    st_shap(shap.plots.waterfall(shap_values[0]),height=300)
with sh2:
    st_shap(shap.plots.beeswarm(shap_values),height=300)

# Prediction
st.subheader("Prediction")
st.write("We can now use the model to predict the tip for a new customer")
st.write("Enter the values for the new customer")
new_customer = st.columns(6)

with new_customer[0]:
    #total_bill		sex	smoker	day	time	size
    total_bill = st.number_input("Enter the total bill",min_value=0.0)
with new_customer[1]:
    sex = st.selectbox("Select the gender",['Female','Male'])
with new_customer[2]:
    smoker = st.selectbox("Is the customer a smoker",['No','Yes'])
with new_customer[3]:
    day = st.selectbox("Select the day",['Sun','Sat','Thur','Fri'])
with new_customer[4]:
    times = st.selectbox("Select the time",['Dinner','Lunch'])
with new_customer[5]:
    size = st.number_input("Enter the size of the group",min_value=0.0)

# pre-process
total_bill = np.log(total_bill)
pred_dict = {'total_bill':total_bill,
             'sex':sex,
                'smoker':smoker,
                'day':day,
                'time':times,
                'size':size}

pred_df = pd.DataFrame([pred_dict])

pred_df['sex'] = pred_df['sex'].map({'Female': 0, 'Male':1})
pred_df['smoker'] = pred_df['smoker'].map({'No': 0, 'Yes':1})
pred_df['day'] = pred_df['day'].map({'Sun': 0, 'Sat':1, 'Thur':2, 'Fri':3})
pred_df['time'] = pred_df['time'].map({'Dinner':0, 'Lunch':1})

# predict button
predict = st.button("Predict")
if predict:
    #use xgb model
    #add a loading bar and then display the result after 2 seconds
    with st.spinner("Predicting..."):
        time.sleep(2)
    pred = xgb.predict(pred_df)
    st.write(f"Total bill: ${np.exp(total_bill)}")
    st.write(f"The predicted tip is: ${np.exp(pred)[0]}")
    st.write(f"Tip percentage: {round(np.exp(pred)[0]/np.exp(total_bill)*100,2)}% of the total bill")