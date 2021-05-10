import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st



st.write(""" 
WELCOME!
DETECT DIABETES USING MACHINE LEARNING! 
""")

image = Image.open('C:/Users/shrey.DESKTOP-O934829/PycharmProjects/diabetesdetection/dia.jpg')
st.image(image, caption='Diabetes Detection with Python and ML', use_column_width= True)

df = pd.read_csv('C:/Users/shrey.DESKTOP-O934829/PycharmProjects/diabetesdetection/diabetes.csv')

st.subheader('Data Information from Different Sources:')

st.dataframe(df)

st.subheader('Estimates:')
st.write(df.describe())

st.subheader('Graphical Representation of Data:')
chart = st.bar_chart(df)

X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

#split into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

def get_user_input():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    Glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    Blood_pressure = st.sidebar.slider('Blood_pressure', 0, 122, 72)
    Skin_thickness = st.sidebar.slider('Skin_thickness', 0, 99, 23)
    Insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 30.0)
    BodyMassIndex = st.sidebar.slider('BodyMassIndex', 0.0, 67.1, 32.0)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', 0.078, 2.42, 0.3725)
    Age = st.sidebar.slider('Age', 21, 81, 29)



    user_data = { 'Pregnancies' : Pregnancies,
                  'Glucose' : Glucose,
                  'Blood_pressure' : Blood_pressure,
                  'Skin_thickness' : Skin_thickness,
                  'Insulin' : Insulin,
                  'BodyMassIndex' : BodyMassIndex,
                  'DiabetesPedigreeFunction' : DiabetesPedigreeFunction,
                  'Age' : Age

                 }


    #transform data into dataframe
    features = pd.DataFrame(user_data, index = [0])
    return features

user_input = get_user_input()

st.subheader('User Input:')
st.write(user_input)


#create model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100)+'%')

#store model predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

image1 = Image.open('C:/Users/shrey.DESKTOP-O934829/PycharmProjects/diabetesdetection/diabetes.jpg')
st.image(image1, caption='Eat Healthy Foods', use_column_width= True)

#create subheader and display classification
st.subheader('Classification:')
st.subheader('0 :: No Diabetes                      1 :: Diabetes')
st.write(prediction)

