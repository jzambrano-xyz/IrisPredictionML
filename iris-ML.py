#Define libraries and resources to import and use in the project
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

#The function st.write allows us to use Markdown in the Web app
st.write("""
# Dynamic _Iris flower_ prediction

Based on the user's input, this web app uses a **Random forest classifier** to predict the Iris flower variety.

""")

#Create the sidebar menu
#specifying the min, max and default values for each Range
st.sidebar.header('User input parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal lenght', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal lenght', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_lenght' : sepal_length,
            'sepal_width' : sepal_width,
            'petal_lenght' : petal_length,
            'petal_width' : petal_width}

#Get the flowers' characteristics from the Dataframe
    features = pd.DataFrame(data, index = [0])
    return features

#Create the section where the user's input values will be displayed
#Make a dynamic table with Streamlit to display the user's input
df = user_input_features()

st.subheader('User input parameters')
st.write(df)

#Create a table to display the Iris flower varieties
iris = datasets.load_iris()
X = iris.data
Y = iris.target

#Use a Random forest classifier to predict the flower's variety based on the user's input
clf = RandomForestClassifier()
clf.fit(X, Y)

#Create a prediction table with the result
prediction = clf.predict(df)

#Create a table to display the actual probabilities that confirm the result
prediction_proba = clf.predict_proba(df)

#Add text to refer to each table and display the results
st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction probability')
st.write(prediction_proba)
