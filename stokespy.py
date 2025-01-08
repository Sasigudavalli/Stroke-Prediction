import polars as pl
import pandas as pd
import numpy as np
import seaborn as sns
data1 = pd.read_csv("C:\\Users\\user\\oneDrive\\Desktop\\Datasets\\Stroke\\healthcare-dataset-stroke-data.csv")
import warnings
warnings.filterwarnings('ignore')

data1.head()

import seaborn as sns
sns.countplot(data=data1, x='stroke')


counts = data1['stroke'].value_counts()
print(counts)

import numpy as np
data1["bmi"].fillna(np.mean(data1["bmi"]),inplace=True)

data1["gender"] = data1["gender"].replace({"Male": 1, "Female": 0, "Other": 2}).astype(int)
data1["ever_married"] = data1["ever_married"].replace({"Yes": 1, "No": 0}).astype(int)
data1["work_type"]=data1["work_type"].replace({"Private": 1, "Self-employed": 2,"Govt_job":3,"children":4,"Never_worked":5}).astype(int)
data1["Residence_type"]=data1["Residence_type"].replace({"Urban":1,"Rural":0}).astype(int)
data1["smoking_status"].unique()
import numpy as np
data1["bmi"].fillna(np.mean(data1["bmi"]),inplace=True)
data_encoded = pd.get_dummies(data1["smoking_status"],drop_first=True)
data_encoded=data_encoded.astype(int)
main=pd.concat([data1,data_encoded],axis=1)
main.drop(columns="smoking_status",inplace=True)

main.drop(columns=["work_type"],inplace=True)

x=main.drop(columns=["stroke"])

y=main["stroke"]


counts = main['stroke'].value_counts()
print(counts)

ss=main[main['stroke']==1]
ns=main[main['stroke']==0]
print(ss.shape,ns.shape)


from imblearn.combine import SMOTETomek
smk=SMOTETomek(random_state=42)
xnew,ynew=smk.fit_resample(x,y)

from collections import Counter
print("original/:    ",format(Counter(y)))
print("REsampled/:    ",format(Counter(ynew)))


import seaborn as sns


sns.countplot(x=ynew, palette="viridis")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error 

x_train,x_test,y_train,y_test=train_test_split(xnew,ynew,test_size=0.5,random_state=42)



from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()

xt1=sc1.fit_transform(x_train)
print(xt1[0])

xs1=sc1.transform(x_test)
print(xs1[0])


model1=LogisticRegression()

model1.fit(xt1,y_train)

ypred1=model1.predict(xs1)

score1=accuracy_score(ypred1,y_test)
print(score1)




import pickle

model_directory = r'C:\Users\user\OneDrive\Desktop\mdlpkls'

filename_model = model_directory + '\\stroke_model.sav'
filename_scaler = model_directory + '\\standard_model.pkl'
pickle.dump(model1, open(filename_model, 'wb'))
pickle.dump(sc1, open(filename_scaler, 'wb'))



import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
with open('stroke_model.sav', "rb") as model_file:
    model = pickle.load(model_file)

with open('standard_model.pkl','rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Sidebar for user input
st.sidebar.title("Stroke Prediction App")
st.sidebar.markdown("Provide the patient's details below:")

id = st.sidebar.number_input("Patient ID", value=12345, step=1)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
age = st.sidebar.slider("Age", 0, 120, 50)
hypertension = st.sidebar.selectbox("Hypertension", options=[0, 1], help="0: No, 1: Yes")
heart_disease = st.sidebar.selectbox("Heart Disease", options=[0, 1], help="0: No, 1: Yes")
ever_married = st.sidebar.selectbox("Ever Married", options=["Yes", "No"])
Residence_type = st.sidebar.selectbox("Residence Type", options=["Urban", "Rural"])
avg_glucose_level = st.sidebar.slider("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
smoking_status = st.sidebar.selectbox(
    "Smoking Status", options=["formerly smoked", "never smoked", "smokes", "Unknown"]
)

# Process inputs for the model
gender_encoded = 1 if gender == "Male" else 0
ever_married_encoded = 1 if ever_married == "Yes" else 0
Residence_type_encoded = 1 if Residence_type == "Urban" else 0

# One-hot encoding for smoking status
smoking_status_encoded = [0, 0, 0]
if smoking_status == "formerly smoked":
    smoking_status_encoded[0] = 1
elif smoking_status == "never smoked":
    smoking_status_encoded[1] = 1
elif smoking_status == "smokes":
    smoking_status_encoded[2] = 1

# Combine all inputs into a single row (including `id`)
input_data = [
    id,
    gender_encoded,
    age,
    hypertension,
    heart_disease,
    ever_married_encoded,
    Residence_type_encoded,
    avg_glucose_level,
    bmi,
] + smoking_status_encoded

# Convert to a DataFrame for scaling
input_df = pd.DataFrame([input_data], columns=[
    "id", "gender", "age", "hypertension", "heart_disease", 
    "ever_married", "Residence_type", "avg_glucose_level", 
    "bmi", "formerly smoked", "never smoked", "smokes"
])

# Button to trigger prediction
predict_button = st.sidebar.button("Predict Stroke")

if predict_button:
    # Scale the data (including `id`)
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)

    # Display results
    st.write("### Input Summary")
    st.dataframe(input_df)

    st.write("### Prediction Results")
    st.write("**Stroke Risk Prediction:**", "Yes" if prediction[0] == 1 else "No")
    st.write("**Probability of Stroke:**", f"{probability[0][1] * 100:.2f}%")

# Additional app information in the sidebar
st.markdown("""
- This app predicts the likelihood of stroke based on patient details.
- Adjust inputs using the sliders and dropdowns in the sidebar.
""")

