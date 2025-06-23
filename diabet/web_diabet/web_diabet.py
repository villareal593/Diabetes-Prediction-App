#Импорт библиотек и модулей
import streamlit as st
import pickle
import numpy as np

#Загрузка файлов модели и масштабирования
model=pickle.load(open('model.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))

#Масштабирование входных данных
def preprocess_input(user_input, scaler):
    user_input_np = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(user_input_np)
    return scaled_input

#прогноз
def predict_diabetes(user_input):
    scaled_input = preprocess_input(user_input, scaler)
    prediction = model.predict(scaled_input)
    return prediction

#веб-приложение
st.markdown("""
    <h1 style='text-align: center; color: #ff5733;'>Diabetes Prediction App</h1>
    <p style='text-align: center;'>Enter the required information to predict whether diabetes is present or not.</p>
""", unsafe_allow_html=True)

#Пользовательский ввод из веб-приложения
user_input = []
for feature in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                    'BMI', 'DiabetesPedigreeFunction', 'Age']:user_input.append(st.number_input(f'Enter {feature}', step=0.01))

#кнопка
if st.button('Predict'):
    prediction = predict_diabetes(user_input)
    if prediction[0] == 0:
        st.write('No Diabetes')
    else:
        st.write('Diabetes Detected')
