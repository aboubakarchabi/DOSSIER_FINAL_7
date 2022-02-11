import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import pickle
st.title('Prédiction de l\'accord d\'un crédit )

st.write("L'application qui prédit l'accord du crédit")

#Collecter le profil d'entrée
st.sidebar.header("Les caracteristiques du client")

def client_caract_entree():
    AMT_INCOME_TOTAL = st.sidebar.slider('Revenu total',25650.0,337500.0,10000.0)
    AMT_CREDIT = st.sidebar.slider('Crédit',45000,1485000,10000)
    AMT_ANNUITY = st.sidebar.slider('Annuités',1615.5,57312.0,10000.0)
    REGION_POPULATION_RELATIVE = st.sidebar.slider('Région population relative', 0.00029, 0.072, 0.02)
    EXT_SOURCE_3 = st.sidebar.slider('Source extérieure 3',0.0005,0.896,0.05)
    CREDIT_INCOME_PERCENT = st.sidebar.slider('Pourcentage du revenu du crédit',0.14,35.47,0.92)
    ANNUITY_INCOME_PERCENT = st.sidebar.slider('Pourcentage du revenu des annuités',0.008,1.45,0.05)
    CREDIT_TERM = st.sidebar.slider('Terme du crédit',0.02,0.12,0.05)
    YEARS_REGISTRATION = st.sidebar.slider('Année d\'enregistremen', 0.0, 65.0, 5.0)
    YEARS_ID_PUBLISH = st.sidebar.slider("Nombre d'année de publication ID", 0.0, 20.0, 2.0)



    data={
    'AMT_INCOME_TOTAL':AMT_INCOME_TOTAL,
    'AMT_CREDIT':AMT_CREDIT,
    'AMT_ANNUITY':AMT_ANNUITY,
    'REGION_POPULATION_RELATIVE': REGION_POPULATION_RELATIVE,
    'EXT_SOURCE_3':EXT_SOURCE_3,
    'CREDIT_INCOME_PERCENT':CREDIT_INCOME_PERCENT,
    'ANNUITY_INCOME_PERCENT':ANNUITY_INCOME_PERCENT,
    'CREDIT_TERM':CREDIT_TERM,
    'YEARS_REGISTRATION':YEARS_REGISTRATION,
    'YEARS_ID_PUBLISH': YEARS_ID_PUBLISH

    }

    profil_client=pd.DataFrame(data,index=[0])
    return profil_client

input_df=client_caract_entree()

st.subheader('on veut trouver la catégorie de cette fleur')
st.write(input_df)

dat=pd.read_csv('train2.csv')

X = dat.drop(["SK_ID_CURR", "TARGET"], axis = 1)
y = dat.TARGET

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_train_norm = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_train_norm,y, test_size=0.2, random_state=1)
clf1 = RandomForestClassifier(random_state=0)
clf1.fit(X_train, y_train)

prevision = clf1.predict(input_df)

st.subheader("La catégorie est:")
st.write( dat.TARGET[prevision])
