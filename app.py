"""
TalentKeep AI - Application web de prédiction du turnover.
Ce script utilise Streamlit pour fournir une interface interactive aux équipes RH.
Il charge le modèle Random Forest entraîné, applique les mêmes transformations
aux nouvelles données, et utilise SHAP pour expliquer le risque de départ.
"""

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION DE L'APPLICATION
# ==========================================
# Définition des métadonnées de la page web
st.set_page_config(page_title="TalentKeep AI", page_icon="👥", layout="wide")

st.title("👥 TalentKeep AI - Prédiction de l'Attrition (Turnover)")
st.markdown("Cette application aide les RH à identifier **en temps réel** les employés à risque de démission et fournit une explication de l'IA (SHAP) pour mettre en place des actions de rétention ciblées.")

# ==========================================
# 2. CHARGEMENT DU MODÈLE ET MISE EN CACHE
# ==========================================
# Utilisation de @st.cache_resource pour ne charger les fichiers .pkl qu'une seule fois,
# ce qui optimise considérablement la vitesse de l'application.
@st.cache_resource
def load_assets():
    model = joblib.load('models/hr_model.pkl')
    encoders = joblib.load('models/encoders.pkl')
    features = joblib.load('models/features.pkl')
    return model, encoders, features

try:
    rf_model, encoders, feature_names = load_assets()
    st.sidebar.success("✅ IA chargée et prête !")
except Exception as e:
    st.error("Erreur critique : Impossible de charger l'Intelligence Artificielle (.pkl introuvables).")
    st.stop() # Bloque l'application si le modèle est absent

# ==========================================
# 3. INTERFACE UTILISATEUR & UPLOAD
# ==========================================
st.sidebar.header("📥 Analyser une équipe")
st.sidebar.markdown("Uploadez la base de données de vos employés (CSV).")
uploaded_file = st.sidebar.file_uploader("Fichier RH (ex: employes_mars.csv)", type=["csv"])

if uploaded_file is not None:
    # Lecture des données importées
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("📊 Aperçu des données de l'équipe")
    st.dataframe(df_raw.head())
    
    if st.button("Lancer l'audit de risque de départ 🚀"):
        with st.spinner("L'IA analyse les profils..."):
            
            # ==========================================
            # 4. PRÉTRAITEMENT DES DONNÉES
            # ==========================================
            # On ignore la colonne cible 'Attrition' si le fichier RH la contient déjà
            df_process = df_raw.drop('Attrition', axis=1, errors='ignore')
            
            # Application des encodeurs sauvegardés lors de l'entraînement
            for col, encoder in encoders.items():
                if col in df_process.columns:
                    # Sécurité : Gérer les valeurs inconnues (non vues à l'entraînement) silencieusement
                    df_process[col] = df_process[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
                    df_process[col] = encoder.transform(df_process[col])
            
            # ==========================================
            # 5. PRÉDICTION DES RISQUES (SCORING)
            # ==========================================
            # Extraction de la probabilité de la classe 1 (Risque de départ)
            probabilites = rf_model.predict_proba(df_process)[:, 1]
            df_raw['Risque_Depart (%)'] = (probabilites * 100).round(1)
            
            # Définition du seuil d'alerte métier : Employés ayant > 60% de chances de démissionner
            haut_risque = df_raw[df_raw['Risque_Depart (%)'] >= 60].sort_values(by='Risque_Depart (%)', ascending=False)
            
            st.divider()
            
            # --- AFFICHAGE DES RÉSULTATS ---
            col1, col2 = st.columns(2)
            col1.metric("Total Employés analysés", len(df_raw))
            col2.metric("Alerte Haut Risque (>60%)", len(haut_risque), delta_color="inverse")
            
            if len(haut_risque) > 0:
                st.error(f"🚨 Attention : {len(haut_risque)} collaborateurs présentent un risque critique de démission imminente.")
                # Affichage des personnes à risque avec un dégradé de couleur (plus c'est rouge, plus le risque est élevé)
                st.dataframe(haut_risque.style.background_gradient(subset=['Risque_Depart (%)'], cmap='Reds'))
                
                # ==========================================
                # 6. EXPLAINABLE AI (SHAP)
                # ==========================================
                st.subheader("🧠 Pourquoi vont-ils partir ? (Explainable AI)")
                st.markdown(f"Analyse détaillée du collaborateur le plus à risque (Employé n°{haut_risque.index[0]} - Risque: {haut_risque.iloc[0]['Risque_Depart (%)']}%) :")
                
                # Isolation de l'employé présentant le score de départ le plus élevé
                idx_critique = haut_risque.index[0]
                ligne_critique = df_process.loc[[idx_critique]]
                
                # Initialisation de l'explicateur SHAP spécifique aux modèles d'arbres
                explainer = shap.TreeExplainer(rf_model)
                shap_obj = explainer(ligne_critique)
                
                # Génération du graphique en cascade (Waterfall plot)
                # On cible spécifiquement la classe 1 (Démission = Oui) de l'objet SHAP tridimensionnel
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(shap_obj[0, :, 1], show=False)
                
                st.pyplot(fig)
                
                # Conseils d'interprétation pour les managers RH
                st.info("""
                **💡 Recommandation RH :**
                * Les barres **rouges** représentent les facteurs de frustration de cet employé (ex: trajet trop long, salaire trop bas) qui augmentent son envie de partir. 
                * Les barres **bleues** sont ses points d'ancrage (ce qui le retient encore). 
                *Action suggérée : Organisez un entretien avec son manager pour discuter en priorité des points rouges.*
                """)
            else:
                st.success("✅ Bonne nouvelle : le climat social est sain, aucun risque majeur détecté.")
else:
    st.info("👈 En attente du fichier CSV pour démarrer l'audit.")

# ==========================================
# 7. PIED DE PAGE (FOOTER)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Créé par Thierry Maesen**")
st.sidebar.markdown("Un outil RH propulsé par Machine Learning.")
