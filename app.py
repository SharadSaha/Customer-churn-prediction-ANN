import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier 
# from sklearn import metrics

# print(model)
x_train=pd.read_csv('X_train.csv')
x_test=pd.read_csv('X_test.csv')
y_train=pd.read_csv('Y_train.csv')
y_test=pd.read_csv('Y_test.csv')
dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
dataset.TotalCharges = pd.to_numeric(dataset.TotalCharges,errors='coerce')

st.title("CUSTOMER CHURN PREDICTION")

image = Image.open("img.webp")
st.image(image,use_column_width = True)

check = st.checkbox("View customer data")
if check:
    st.write(dataset.head(4))


st.markdown("### **Please enter the details to get prediction**")
SeniorCitizen = st.radio(
     "Is customer a senior citizen?",
     (0,1))

MonthlyCharges = st.slider("Customer monthly charges",
                    0.0,120.0,0.1)
TotalCharges = st.slider("Customer total charges",
                    0.0,9000.0,0.1)

col1,col2 = st.columns(2)
with col1:
    Gender = st.radio(
        'Gender',('Male','Female')
    )

    Partner = st.radio(
        'Partner',('Yes','No')
    )

    Dependents = st.radio('Dependents',
                ('Yes','No'))

    PhoneService = st.radio('Phone service',
                ('Yes','No'))

    MultipleLines = st.radio('Multiple lines',
                ('Yes','No','No phone service'))

    OnlineSecurity = st.radio('Online Security',
                ('No','No internet service','Yes'))

    InternetService = st.radio('Internet Service',
                ('DSL','Fiber optic','No'))

    OnlineBackup = st.radio('Online Backup',
                ('Yes','No','No internet service'))

    DeviceProtection = st.radio('Device Protection',
                ('Yes','No','No internet service'))

with col2:
    StreamingTV = st.radio('Streaming TV',
                ('Yes','No','No internet service'))

    StreamingMovies = st.radio('Streaming Movies',
                ('Yes','No','No internet service'))

    Contract = st.radio('Contract',
                ('Month-to-month','One year','Two year'))

    PaperlessBilling = st.radio('Paperless Billing',
                ('No','Yes'))

    PaymentMethod = st.radio('Payment Method',
                ('Bank transfer (automatic)','Credit card (automatic)',
                'Electronic check','Mailed check'))

    tenure = st.slider("Tenure",
                    1,72,1)

    TechSupport = st.radio('Tech Support',
            ('Yes','No','No internet service'))



data = [[SeniorCitizen,
    MonthlyCharges,
    TotalCharges,
    Gender,
    Partner,
    Dependents,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    tenure]]



new_df = pd.DataFrame(data, columns = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                           'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                           'PaymentMethod', 'tenure'])

df = dataset[['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                           'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                           'PaymentMethod', 'tenure']]


df_1 = pd.concat([df, new_df],axis=0,ignore_index = True) 
# Group the tenure in bins of 12 months
labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

df_1['tenure_group'] = pd.cut(df_1.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
#drop column customerID and tenure
df_1.drop(columns= ['tenure'], axis=1, inplace=True)
# final_data=df_1.drop(['MonthlyCharges','TotalCharges'],axis=1)
# st.write(final_data)
m = df_1['MonthlyCharges']
t = df_1['TotalCharges']

final_data=df_1.drop(['MonthlyCharges','TotalCharges'],axis=1)
final_data = pd.get_dummies(final_data)
final_data['MonthlyCharges'] = m
final_data['TotalCharges'] = t
# final_data=pd.concat([final_data,df_1[['MonthlyCharges','TotalCharges']]],axis=1,ignore_index=True)
final_data = final_data[['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender_Female',
       'gender_Male', 'Partner_No', 'Partner_Yes', 'Dependents_No',
       'Dependents_Yes', 'PhoneService_No', 'PhoneService_Yes',
       'MultipleLines_No', 'MultipleLines_No phone service',
       'MultipleLines_Yes', 'InternetService_DSL',
       'InternetService_Fiber optic', 'InternetService_No',
       'OnlineSecurity_No', 'OnlineSecurity_No internet service',
       'OnlineSecurity_Yes', 'OnlineBackup_No',
       'OnlineBackup_No internet service', 'OnlineBackup_Yes',
       'DeviceProtection_No', 'DeviceProtection_No internet service',
       'DeviceProtection_Yes', 'TechSupport_No',
       'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No',
       'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No internet service',
       'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
       'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
       'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
       'tenure_group_1 - 12', 'tenure_group_13 - 24', 'tenure_group_25 - 36',
       'tenure_group_37 - 48', 'tenure_group_49 - 60', 'tenure_group_61 - 72']]
# st.write(final_data.shape)

st.markdown("### **Data entered: **")
st.write(new_df)

# import os
# os.chdir("C:\Users\Sharad\Desktop\files\ML\telco-customer-churn")

def load():
    model = tf.keras.models.load_model('final_model')
    return model


# model = pickle.load(open('model.h5','rb'))
model = load()

data = final_data.tail(1)
print(data)
print(len(data))
data = np.asarray(data).astype('float32')

prediction = model.predict(data)

if(prediction>0.5):
    st.markdown("### **Probability of customer churn is high!**")
elif(prediction>0.4 and prediction<=0.5):
    st.markdown("### **Moderate Probability of cutomer churn.**")
else:
    st.markdown("### **Very low probability of cutomer churn!**")
# st.write(prediction)
