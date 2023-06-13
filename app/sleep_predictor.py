import streamlit as st

# Main imports
import pandas as pd
import numpy as np

# Data Pre-processing imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Model and metric imports
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Create model function here with @st_cache_resource
@st.cache_resource
def MLPModelPrepper():
    # Read Sleep_health_and_lifestyle_dataset.csv
    url = 'https://drive.google.com/file/d/15koNKCD-31D9vevZgUbbLKNsAVHIXQ-R/view?usp=sharing'
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(url)

    # Drop unnecessary values
    df.drop(['Person ID', 'Occupation', 'Daily Steps', 'Heart Rate', 'Sleep Disorder', 'Blood Pressure'], inplace=True, axis=1)

    # For the BMI Category, we need to replace all entries with "Normal Weight" to "Normal"
    df.loc[df['BMI Category'] == "Normal Weight", 'BMI Category'] = "Normal"

    # Create arrays for the features and the response variable
    X = df.drop('Physical Activity Level', axis=1).values
    y = df['Physical Activity Level'].values

    # Turn Gender into a number using LabelEncoder
    labelencoder_gender = LabelEncoder()
    X[:, 0] = labelencoder_gender.fit_transform(X[:, 0]) # Gender

    # BMI Category into a number using Label Encoder
    labelencoder_bmi = LabelEncoder()
    X[:, 5] = labelencoder_bmi.fit_transform(X[:, 5]) # BMI Category

    # Testing StandardScaler vs MinMaxScaler
    # sc = StandardScaler()
    # X[:, 3:4] = sc.fit_transform(X[:, 3:4])
    ms = MinMaxScaler(feature_range = (-1, 1)) # tanh
    X[:, 3:5] = ms.fit_transform(X[:, 3:5])
    
    # Split into training (80% of the dataset) and test set (20% of the dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    # Create the regressor: reg_all
    reg_all = MLPRegressor(activation='tanh', alpha = 0.01, hidden_layer_sizes= (25), learning_rate = 'constant', max_iter = 100000, random_state= 25, solver ='lbfgs')

    # parameters = {'learning_rate': ['constant', 'invscaling', 'adaptive'],
    #             'solver': ['lbfgs'],
    #             'activation' : ['tanh'],
    #             'max_iter': [100000],
    #             'alpha': [0.01, 0.001, 0.0001],
    #             'hidden_layer_sizes': [(5), (10), (25)], 
    #             'random_state': [25],
    #             }
    # search = GridSearchCV(reg_all, parameters, refit=True,verbose=2)

    # Fit it to the training data
    search = reg_all
    search.fit(X_train, y_train)
    y_pred = search.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # print("Tuned MLP Parameters: {}".format(search.best_params_))
    # print("Best score is {}".format(search.best_score_))
    print("R^2: {}".format(r2_score(y_test, y_pred)))
    print("Root Mean Squared Error: {}".format(rmse))

    # Important for predict function - store model, encoders and scalers here
    model_and_scalers = {'model': search, 'labelEncoderGender': labelencoder_gender, 'labelEncoderBMI': labelencoder_bmi, 'MinMaxScaler': ms}

    return model_and_scalers;

#define predict function here
def predict(sex, age, weightclass, sleeptime, sleepqual, stresslvl):
    model_and_scalers = MLPModelPrepper()
    leG = model_and_scalers['labelEncoderGender']
    leB = model_and_scalers['labelEncoderBMI']
    sex_encoded = leG.transform(np.array([sex]))
    weightclass_encoded = leB.transform(np.array([weightclass]))

    ms = model_and_scalers['MinMaxScaler']
    sleepqual_stresslvl_comb = np.array([sleepqual, stresslvl]).reshape(1, -1)
    sleepqual_stresslvl_scaled = ms.transform(sleepqual_stresslvl_comb)
    print(sleepqual_stresslvl_scaled)

    mlpModel = model_and_scalers['model']
    print(sex_encoded, age, sleeptime, sleepqual_stresslvl_scaled[0][0], sleepqual_stresslvl_scaled[0][1], weightclass_encoded)
    input_arr = np.array([sex_encoded[0], age, sleeptime, sleepqual_stresslvl_scaled[0][0], sleepqual_stresslvl_scaled[0][1], weightclass_encoded[0]]).reshape(1, -1)
    prediction = mlpModel.predict(np.array(input_arr))
    pred_phys_act = round(prediction[0])

    return pred_phys_act

# Page Setup
header_row1_title, header_row1_description = st.columns(2)
header_row2_subheader, header_row2_authors = st.columns(2)

with header_row1_title:
    st.title("Physical Activity Level Predictor")

with header_row1_description:
    st.write("Insert long description here about the project and maybe some insight on to how it was made")

with header_row2_subheader:
    st.write("Made possible through the use of Multi-layer Perceptrons!")

with header_row2_authors:
    st.write("Prepared by Ellis Caluag, Sofia Canlas, Justin Ruaya, and Hans Salazar 😁")

st.divider() ##############################################

st.header("Input Parameters")

sex_inputcol, age_inputcol, weightclass_inputcol = st.columns(3)
sleeptime_inputcol, sleepqual_inputcol, stresslvl_inputcol = st.columns(3)

with sex_inputcol:
    st.subheader("Sex")
    sex_input = st.radio("Select one", ["Male", "Female"])
    #preprocess data here to fit the model input

with age_inputcol:
    st.subheader("Age")
    age_input = st.slider("Expressed in *years*", min_value=27, max_value=59)

with sleeptime_inputcol:
    st.subheader("Sleep Duration")
    sleeptime_input = st.slider("Expressed in *hours*", min_value=5.8, max_value=8.5, step=0.1)

with sleepqual_inputcol:
    st.subheader("Quality of Sleep")
    sleepqual_input = st.slider("Subjective level of sleep quality", min_value=1, max_value=10)

with stresslvl_inputcol:
    st.subheader("Stress Level")
    stresslvl_input = st.slider("Subjective level of stress", min_value=1, max_value=10)

with weightclass_inputcol:
    st.subheader("Weight Classification")
    weightclass_input = st.radio("BMI Category", ["Normal", "Overweight", "Obese"])
    #preprocess data here to fit the model input


st.divider() #############################################

st.header("Results")

if st.button("Predict Physical Activity Level"):
    pred = predict(sex_input, age_input, weightclass_input, sleeptime_input, sleepqual_input, stresslvl_input)
    st.write(f"Prediction: {pred} minutes of physical activity a day is recommended.")