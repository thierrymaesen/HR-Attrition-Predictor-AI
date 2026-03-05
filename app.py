import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# Configuration de la page
st.set_page_config(page_title="TalentKeep AI", page_icon="👥", layout="wide")

st.title("👥 TalentKeep AI - Prédiction de l'Attrition (Turnover)")
st.markdown("Cette application aide les RH à identifier **en temps réel** les employés à risque de démission et fournit une explication de l'IA (SHAP) pour mettre en place des actions de rétention ciblées.")

# Chargement du modèle et des encodeurs
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
    st.error("Erreur : Impossible de charger l'Intelligence Artificielle.")
    st.stop()

# Barre latérale : Upload du fichier
st.sidebar.header("📥 Analyser une équipe")
st.sidebar.markdown("Uploadez la base de données de vos employés (CSV).")
uploaded_file = st.sidebar.file_uploader("Fichier RH (ex: employes_mars.csv)", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("📊 Aperçu des données de l'équipe")
    st.dataframe(df_raw.head())
    
    if st.button("Lancer l'audit de risque de départ 🚀"):
        with st.spinner("L'IA analyse les profils..."):
            
            # Préparation des données (on ignore la colonne cible si elle y est)
            df_process = df_raw.drop('Attrition', axis=1, errors='ignore')
            
            # Encodage du texte en chiffres pour l'IA
            for col, encoder in encoders.items():
                if col in df_process.columns:
                    # Gérer les valeurs non vues avec un try/except silencieux
                    df_process[col] = df_process[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
                    df_process[col] = encoder.transform(df_process[col])
            
            # Prédiction des risques (Probabilité)
            # On prend la colonne 1 (probabilité de partir)
            probabilites = rf_model.predict_proba(df_process)[:, 1]
            df_raw['Risque_Depart (%)'] = (probabilites * 100).round(1)
            
            # Définir qui est à haut risque (> 60%)
            haut_risque = df_raw[df_raw['Risque_Depart (%)'] >= 60].sort_values(by='Risque_Depart (%)', ascending=False)
            
            st.divider()
            
            # --- RÉSULTATS ---
            col1, col2 = st.columns(2)
            col1.metric("Total Employés analysés", len(df_raw))
            col2.metric("Alerte Haut Risque (>60%)", len(haut_risque), delta_color="inverse")
            
            if len(haut_risque) > 0:
                st.error(f"🚨 Attention : {len(haut_risque)} collaborateurs présentent un risque critique de démission imminente.")
                st.dataframe(haut_risque.style.background_gradient(subset=['Risque_Depart (%)'], cmap='Reds'))
                
                # --- EXPLAINABLE AI (SHAP) ---
                st.subheader("🧠 Pourquoi vont-ils partir ? (Explainable AI)")
                st.markdown(f"Analyse détaillée du collaborateur le plus à risque (Employé n°{haut_risque.index[0]} - Risque: {haut_risque.iloc[0]['Risque_Depart (%)']}%) :")
                
                # Récupérer l'employé le plus à risque
                idx_critique = haut_risque.index[0]
                ligne_critique = df_process.loc[[idx_critique]]
                
                                # --- EXPLAINABLE AI (SHAP) ---
                explainer = shap.TreeExplainer(rf_model)
                # On utilise la nouvelle API SHAP qui gère mieux les graphiques Waterfall
                shap_obj = explainer(ligne_critique)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                # On sélectionne l'employé [0] et la classe [1] (démission = Oui)
                shap.plots.waterfall(shap_obj[0, :, 1], show=False)

                
                st.pyplot(fig)
                
                st.info("""
                **💡 Recommandation RH :**
                Les barres **rouges** représentent les facteurs de frustration de cet employé (ex: trajet trop long, salaire trop bas). 
                Les barres **bleues** sont ses points d'ancrage (ce qui le retient). 
                *Action suggérée : Organisez un entretien avec son manager pour discuter en priorité des points rouges.*
                """)
            else:
                st.success("✅ Bonne nouvelle : le climat social est sain, aucun risque majeur détecté.")
else:
    st.info("👈 En attente du fichier CSV pour démarrer l'audit.")

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Créé par Thierry Maesen**")
st.sidebar.markdown("Un outil RH propulsé par Machine Learning.")
