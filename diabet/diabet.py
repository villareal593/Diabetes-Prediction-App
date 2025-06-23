#Импорт необходимых модулей и библиотек
import pandas as pd

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import warnings
import pickle
warnings.filterwarnings('ignore')

#Данные
df = pd.read_csv('diabetes.csv')

features=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
X=pd.get_dummies(df[features])
y=df.Outcome

#Масштабирование данных
scaler = StandardScaler()
standardized_data = scaler.fit_transform(X)
X=standardized_data

#разделение данных для обучения и тестирования
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=1/11,random_state=242)

#модель
lda = LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)
X_train_prediction = lda.predict(x_test)

#Связывание с streamlit для веб-приложения
pickle.dump(lda,open('model.pkl','wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
