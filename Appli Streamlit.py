# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor #il faut l'installer sur votre environnement, ouvrir le cmd + "pip install xgboost"
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold, cross_val_score
import pickle
from sklearn.datasets import load_diabetes


st.title("""
# Prédiction des nouvelles contamination de Covid 19 en France""")

selectbox=st.sidebar.selectbox("étapes du projet",["Data","Visualisation", "modelisation"])

if selectbox=="Data":
   df = pd.read_csv("Donnee_Covid_Traitees.csv")
   st.write("""
         ## On affiche notre dataframe
         """)
   st.write(df.head())
   df["date"] = pd.to_datetime(df["date"], format = '%Y/%m/%d')    
   st.write("Les données ont été récupérées dans les bases mises à disposition par le gouvernement")
   st.write("2 fichiers ont été choisis afin d'obtenir un modèle robuste")
   st.write("- un fichier récapitulant l'ensemble des indicateurs de l'épidémie")
   st.write("URL https://www.data.gouv.fr/fr/datasets/synthese-des-indicateurs-de-suivi-de-lepidemie-covid-19/")
   st.write("- un autre fichier récapitulant les données relatives à la vaccination")
   st.write("URL : https://www.data.gouv.fr/fr/datasets/donnees-relatives-aux-personnes-vaccinees-contre-la-covid-19-1/")

   st.write("Données des indicateurs de l'épidémie")
   st.write("- création de 2 variables indicatrices : la saison et le confinement")
   st.write("- Suppression des données avant le 19 Mai 2020 (peu de données disponibles avant cette date)")
   
   
elif selectbox=="Visualisation":
    df = pd.read_csv("Donnee_Covid_Traitees.csv")
    df["date"] = pd.to_datetime(df["date"], format = '%Y/%m/%d')    
    st.write("Visualisation des données" )
    
    
    plt.figure(figsize=(5,5))
    plt.title("Impact de l'évolution du taux de reproduction du virus sur le taux d'incidence")
    fig,ax=plt.subplots()
    pl=plt.plot(df['date'],df['tx_incid'],color='blue')
    plt.legend(pl,["taux d'incidence"],loc="upper left")
    plt.ylabel("taux d'incidence pour 100 000 habitants")
    plt.xlabel('date')
    plt.xticks(rotation=45)
    ax2 = plt.gca().twinx()
    pl2 = ax2.plot(df['date'],df['R'], color = 'red')
    plt.legend(pl2, ['taux de reproduction'],loc="upper right")
    plt.ylabel('R0')
    
    st.pyplot(fig)
    
    st.write("L'influence du R0 sur le taux d'incidence est bien vérifié")
    
    plt.figure(figsize=(7,5))
    plt.title("Corrélation entre le nombre d'hospitalisation et le taux d'occupation des lits dans les hôpitaux")
    fig1,ax=plt.subplots()
    pl=plt.plot(df['date'],df['hosp'],color='blue')
    plt.legend(pl,["nombre d'hospitalisations"],loc="upper left")
    plt.ylabel("nombre d'hospitalisations en France")
    plt.xlabel('date')
    plt.xticks(rotation=45)
    ax2 = plt.gca().twinx()
    pl2 = ax2.plot(df['date'],df['TO'], color = 'red')
    plt.legend(pl2, ["taux d'occupation"],loc="upper right")
    plt.ylabel('T0');

    st.pyplot(fig1)
    
    
    correlation=df.corr()
    fig2,ax=plt.subplots(figsize=(15,15))
    sns.heatmap(correlation,annot=True,ax=ax);
    st.pyplot(fig2)
    
    st.write("Les varaibles confinement_oui et confinement_non ainsi que saison_été et saison_hiver sont colinéaires. Nous allons donc par la suite supprimer l'une de ces variables")
    st.write("Les variables relatives à l'hospitalisation des patients (rad, dchosp,incid_hosp,incid_rea) ainsi que les données relatives à la vaccination (n_cum_dose1, n_cum_complet...) sont très liées les unes aux autres.")

elif selectbox=="modelisation":

    st.write("""
             Il faut déterminer notre variable cible.

    La période moyenne d'incubation du covid 19 est de 5 jours

    Nous supposons qu'à date T+0, les contaminés seront déclarés positifs à T+5 

    C'est une grosse simplification, car les temps d'incubation sont variables, et toutes les personnes ne se font pas tester

    Il semble cependant que cela soit le plus raisonnable pour nous

    La variable cible est donc : le nombre de positif à T+5

    Nous créons une colonne "target" qui correspond à cela
    """)


    df = pd.read_csv("Donnee_Covid_Traitees.csv")
    df.info()
    print("On regarde le df",df.head())
    # On constate des colonnes colinéaires : Saison été / Saison hiver, Convifinement Oui / Confinement Non"
    #On va donc les drop

    df = df.drop(["saison_hiver", "confinement_non"], axis = 1)



    df_cible = df.loc[5:,["pos"]]
    #je renomme la colonne
    df_cible = df_cible.rename(columns={"pos" : "pos+5"})
    #Je supprime les 5 dernière ligne de mon df de base, je n'ai pas de variable cible à mettre en face
    df = df.iloc[:-5, :]


    #On transforme nos dates en ordonnal
    df["date"] = pd.to_datetime(df["date"], format = '%Y/%m/%d')
    df["date"] = df['date'].apply(datetime.toordinal)


    #On scale les données de nos features sans la colonne date
    df_scale = df.drop(["date"], axis = 1)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_scale)
    df_scaled = pd.DataFrame(data = df_scaled)
    #on remet les noms de colonnes
    df_scaled.columns = df_scale.columns
    #on rajoute notre date
    df_scaled["date"] = df["date"]
    st.write("""
             # Voici notre data frame une fois le format date réalisé, les colonnes scalées...""")
    st.write(pd.DataFrame(df_scaled))

   


    st.write("""
             On fait notre train/test

             On n'utilise pas la fonction classique de train/test split, on va supposer que le temps a un impact, donc on garde l'ordre

    """)
    X_train = df_scaled[0:344]
    X_test = df_scaled[344:]
    y_train = df_cible[0:344]
    y_test = df_cible[344:]

    st.write("""
             ## Modèle "basique" de RandomForestRegressor, on va chercher les variables les plus importantes, que l'on va afficher juste en dessous
             """)
    #model = RandomForestRegressor(random_state = 52)
    # fit
    #model = model.fit(X_train, y_train.values.ravel())
    #On enregistre au format Pickle
    Pkl_Filename = "'model_RFR_variables_non_select.pkl'"  
    #with open(Pkl_Filename, 'wb') as file:  
        #pickle.dump(model, file)
    #on chargera avec cette ligne 
    pickled_model = pickle.load(open(Pkl_Filename, 'rb'))
    
    # get importance
    importances = pickled_model.feature_importances_
    #on va stocker là dedans les variables les plus importantes
    varRandomForest = []
    #On va trouver ces var les plus importantes
    std = np.std([tree.feature_importances_ for tree in pickled_model.estimators_], axis=0)


    forest_importances = pd.Series(importances, index=df_scaled.columns)
    forest_importances = forest_importances.sort_values(ascending = False)

    #On exécute ce bloc tout seul
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Importance de nos features dans le modèle")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    st.write(fig)

    #puis ce bloc
    data = pd.DataFrame(forest_importances[:20]).transpose()
    fig, ax = plt.subplots()
    ax = sns.barplot(data=data, orient = 'h')
    ax.set(xlabel='Importance feature avec MDI', ylabel='Features')
    #très intéressant : la date est nulle ! le pos n'est pas très élevé non plus !
    st.write(fig)

    st.write("""
             ## On garde les neufs premières variables qui semblent les plus importantes (on pourrait tester avec 5)

             On recrée le modèle avec ces nouvelles variables sélectionnées, et on affiche les mêmes graphiques
    """)

    varRandomForest = list(forest_importances[0:9].index)
    df_scaled_opti_RFR = pd.DataFrame()
    for i in varRandomForest:
        df_scaled_opti_RFR[i] = df_scaled[i]


    #On refait le même modèle mais avec nos nouvelles variables
    X_train = df_scaled_opti_RFR[0:344]
    X_test = df_scaled_opti_RFR[344:]
    #model_var_opti_RFR = RandomForestRegressor(random_state = 52)
    #model_var_opti_RFR = model_var_opti_RFR.fit(X_train, y_train.values.ravel())
    
    #On enregistre au format Pickle
    Pkl_Filename = "'model_RFR_variables_select.pkl'"  
    #with open(Pkl_Filename, 'wb') as file:  
        #pickle.dump(model, file)
    #on chargera avec cette ligne 
    pickled_model_var_opti_RFR = pickle.load(open(Pkl_Filename, 'rb'))
    
    importances = pickled_model_var_opti_RFR.feature_importances_
    std = np.std([tree.feature_importances_ for tree in pickled_model_var_opti_RFR.estimators_], axis=0)

    forest_importances = pd.Series(importances, index=df_scaled_opti_RFR.columns)
    forest_importances = forest_importances.sort_values(ascending = False)

    #on exécute ce bloc tout seul
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Importance de nos features dans le modèle opti")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    st.write(fig)

    #puis celui là
    fig, ax = plt.subplots()
    data = pd.DataFrame(forest_importances[:9]).transpose()
    ax = sns.barplot(data=data, orient = 'h')
    ax.set(xlabel='Importance feature avec MDI', ylabel='Features')
    st.write(fig)



    st.write("""
             # On refait la même chose (variables sélection) avec un XGBoost
             """)
    #X_train = df_scaled[0:344]
    #X_test = df_scaled[344:]
    #model_XG = XGBRegressor(seed = 52)
    #model_XG = model_XG.fit(X_train, y_train)
    Pkl_Filename = "model_XG_variables_non_select.pkl"  
    #with open(Pkl_Filename, 'wb') as file:  
        #pickle.dump(model_XG, file)
    #on chargera avec cette ligne 
    pickled_model_XG_variables_non_select = pickle.load(open(Pkl_Filename, 'rb'))
    importances = pickled_model_XG_variables_non_select.feature_importances_

    varXGB = []

    XGB_importances = pd.Series(importances, index=df_scaled.columns)
    XGB_importances = XGB_importances.sort_values(ascending = False)

    #on exécute ce premier bloc
    fig, ax = plt.subplots()
    XGB_importances.plot.bar( ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    st.write(fig)

    #on exécute ce bloc ensuite
    fig, ax = plt.subplots()
    data = pd.DataFrame(XGB_importances[:20]).transpose()
    ax = sns.barplot(data=data, orient = 'h')
    ax.set(xlabel='Importance feature avec MDI', ylabel='Features')
    st.write(fig)
    st.write("""
             ## On constate que les variables les plus importantes sont les mêmes : on garde les 9 premières et on re affiche les graphiques
             """)
    #On garde les neufs premières variables
    varXGB = list(XGB_importances[0:9].index)
    df_scaled_opti_XGB = pd.DataFrame()
    for i in varXGB:
        df_scaled_opti_XGB[i] = df_scaled[i]
    
    #On refait le modèle
    X_train = df_scaled_opti_XGB[0:344]
    X_test = df_scaled_opti_XGB[344:]
    #model_opti_XGB = XGBRegressor(seed = 52)
    #model_opti_XGB = model_opti_XGB.fit(X_train, y_train)
    
    Pkl_Filename = "model_XG_variables_select.pkl"  
    #with open(Pkl_Filename, 'wb') as file:  
        #pickle.dump(model_opti_XGB, file)
    #on chargera avec cette ligne
    pickled_model_opti_XGB= pickle.load(open(Pkl_Filename, 'rb'))
    importances = pickled_model_opti_XGB.feature_importances_

    XGB_importances = pd.Series(importances, index=df_scaled_opti_XGB.columns)
    XGB_importances = XGB_importances.sort_values(ascending = False)

    #on exécute ce premier bloc
    fig, ax = plt.subplots()
    XGB_importances.plot.bar( ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    st.write(fig)
    
    #on exécute ce bloc ensuite
    fig, ax = plt.subplots()
    data = pd.DataFrame(XGB_importances[:9]).transpose()
    ax = sns.barplot(data=data, orient = 'h')
    ax.set(xlabel='Importance feature avec MDI', ylabel='Features')
    st.write(fig)
    
    st.write("""
    ## On constate ici que tx_incid n'a plus d'importance. 
    
    On le drop, on refait les optimisations
    """)
    
    varXGB = list(XGB_importances[0:8].index)
    df_scaled_opti_XGB = pd.DataFrame()
    for i in varXGB:
        df_scaled_opti_XGB[i] = df_scaled[i]
    
    
    
    
    

    #Il faut load le modèle ici
    #Nom de la variable : pickled_res_cv_rf
    Pkl_Filename = "model_res_cv_rf.pkl"  
    #with open(Pkl_Filename, 'wb') as file:  
        #pickle.dump(model_XG, file)
    #on chargera avec cette ligne
    pickled_res_cv_rf= pickle.load(open(Pkl_Filename, 'rb'))
    st.write("""
    ## Voici les meilleurs hyperparamètres pour RFR après optimisation sauvé dans un pickle (CV=3, 100 itérations, tous coeurs utilisés pour optimiser le temps)
    """)
    st.write(pickled_res_cv_rf.best_estimator_)
    
    
    #Il faut load le modèle ici
    #choper la config et la mettre dans la variable "params_XGB"
    Pkl_Filename = "model_XGB_final.pkl"  
    #with open(Pkl_Filename, 'wb') as file:  
        #pickle.dump(model_XGB_final, file)
    #on chargera avec cette ligne
    pickled_model_XGB_final= pickle.load(open(Pkl_Filename, 'rb'))
    st.write("""
    ## Après optimisation (sauvés dans le pickle) Les meilleurs params XGB sont : """, 
    pickled_model_XGB_final.get_xgb_params())
    
    st.write("""
    # On va lancer nos modèles, puis les comparer
    """)
    
    st.write("""
    ## D'abord le XGBoost""")
    #TEST
    #XGB
    # evaluate model
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pickled_model_XGB_final, df_scaled_opti_XGB, df_cible, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force scores to be positive
    scores = np.absolute(scores)
    st.write("""Moyenne MAE et sa variance : %.3f (%.3f)""" % (scores.mean(), scores.std()) )
    X_test_opti_param_XGB = df_scaled_opti_XGB[344:]
    y_XGB_pred = pickled_model_XGB_final.predict(X_test_opti_param_XGB)
    y_XGB_pred = pd.DataFrame(y_XGB_pred)
    y_test = df_cible[344:]
    y_test = y_test.reset_index()
    y_test = y_test.drop("index", axis = 1)
    
    fig, ax = plt.subplots(figsize = (15,15))
    ax.plot(y_XGB_pred, label = "Prédictions XGB", color = "b")
    ax.plot(y_test, label = "Test", color = "red")
    ax.legend()
    ax.set_title("XGB VS Test avec samedi et dimanche")
    st.write(fig)
    st.write("""
    ## Notre Modèle capte complètement la tendance ! On voit sur les données de test des gros pics négatifs, ce sont les dimanche & les samedi. Et si on les supprimait ?
    """)
    
    y_test = df_cible[344:]
    y_test = y_test.reset_index()
    samedi = np.arange(5, 86, 7)
    dimanche = np.arange(6, 86, 7)
    for i in samedi:
        y_test = y_test.drop(i)
        y_XGB_pred = y_XGB_pred.drop(i)
    for i in dimanche:
        y_test = y_test.drop(i)
        y_XGB_pred = y_XGB_pred.drop(i)
    
    y_test = y_test.drop("index", axis = 1)
    fig, ax = plt.subplots(figsize = (15,15))
    ax.plot(y_XGB_pred, color = "b", label = "XGB")
    ax.plot(y_test, color = "r", label = "Test")
    ax.set_title("XGB VS Test sans les dimanche et samedi")
    ax.legend()
    st.write(fig)
    st.write("""
    ## Il reste encore des valeurs "abérantes" : jours féries ?
    
    ## Re-entrainer le modèle sans samedi/dimanche/jours fériés?
    
    ## Re intrainer le modèle en "flaguant" comme confinement/hiver les jours où l'on ne se teste pas ?
    """)
    
    
    
    st.write("""
             ## On passe au modèle RFR final""")
    #RFR final
    y_test = df_cible[344:]
    y_test = y_test.reset_index()
    Pkl_Filename = "model_RFR_final.pkl"  
    #with open(Pkl_Filename, 'wb') as file:  
        #pickle.dump(model_RFR_final, file)
    #on chargera avec cette ligne
    pickled_model_RFR_final= pickle.load(open(Pkl_Filename, 'rb'))
    # evaluate model
    X_test_opti_param_RFR= df_scaled_opti_RFR[344:]
    scores = cross_val_score(pickled_model_RFR_final, df_scaled_opti_RFR, df_cible, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force les scores à être positifs
    scores = np.absolute(scores)
    st.write('Moyenne de la MAE et sa variance: %.3f (%.3f)' % (scores.mean(), scores.std()) )
    
    y_RFR_pred = pickled_model_RFR_final.predict(X_test_opti_param_RFR)
    y_RFR_pred = pd.DataFrame(y_RFR_pred)
    
    y_test = y_test.drop("index", axis = 1)
    
    fig, ax = plt.subplots(figsize = (15,15))
    ax.plot(y_RFR_pred, color="b", label = "RFR")
    ax.plot(y_test, color="r", label = "test")
    ax.set_title("RFR VS Test avec samedi/dimanche")
    ax.legend()
    st.write("""
             Voici la figure entre test et pred pour le RFR avec samedi dimanche""", fig)
    y_test = df_cible[344:]
    y_test = y_test.reset_index()
    for i in samedi:
        y_test = y_test.drop(i)
        y_RFR_pred = y_RFR_pred.drop(i)
    for i in dimanche:
        y_test = y_test.drop(i)
        y_RFR_pred = y_RFR_pred.drop(i)
    
    
    
    y_test = y_test.drop("index", axis = 1)
    
    fig, ax = plt.subplots(figsize = (15,15))
    ax.plot(y_RFR_pred, color = "b", label ="RFR")
    ax.plot(y_test, color = "r", label ="Test")
    ax.legend()
    ax.set_title("RFR VS Test sans samedi dimanche")
    st.write("""
             Voici la figure entre test et pred pour le RFR sans samedi dimanche""", fig)
    from sklearn.metrics import mean_absolute_error
    st.write("""
             # MAE finale XGB boost sans samedi/dimanche """, mean_absolute_error(y_test, y_XGB_pred))
    st.write("""
             # MAE finale RFR sans samedi/dimanche """, mean_absolute_error(y_test, y_RFR_pred))
    st.write("""
             # Le modèle XGB est le meilleur""")