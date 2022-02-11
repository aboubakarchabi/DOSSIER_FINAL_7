import streamlit as st
import pandas as pd
import numpy as np
import pickle

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

#Transformer les données d'entrée en données adaptées à notre modèle
#importer la base de données
df=pd.read_csv('train2.csv')
credit_input=df.drop(columns=['SK_ID_CURR', 'TARGET'])
donnee_entree=pd.concat([input_df,credit_input],axis=0)


#prendre uniquement la premiere ligne
donnee_entree=donnee_entree[:1]

#afficher les données transformées
st.subheader('Les caracteristiques transformés')
st.write(donnee_entree)

#importer le modèle
load_model=pickle.load(open('Prevision_crédit1.pkl','rb'))


#appliquer le modèle sur le profil d'entrée
prevision=load_model.predict(donnee_entree)

st.subheader('Résultat de la prévision')
st.write(prevision)



