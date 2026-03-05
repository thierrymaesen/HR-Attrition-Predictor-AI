# 👥 TalentKeep AI - Prédiction de l'Attrition (Turnover)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hr-attrition-predictor-ai-thierrymaesen.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Le Problème
Le turnover coûte cher aux entreprises. Souvent, les Ressources Humaines découvrent le mécontentement d'un employé au moment où il dépose sa démission, lorsqu'il est déjà trop tard pour agir.

## 💡 La Solution & Value Proposition
**TalentKeep AI** est une application web de Machine Learning conçue pour les équipes RH. Elle ne se contente pas de prédire *qui* va partir, elle explique *pourquoi*. 

**Plus-value du projet :**
1. **Anticipation :** Analyse des bases de données RH pour identifier les employés à haut risque.
2. **Explicabilité (Explainable AI) :** Utilisation de la librairie SHAP pour générer un graphique en cascade. Les RH peuvent voir immédiatement les "points de friction" (ex: salaire, trajet) et les "points d'ancrage" propres à chaque individu.
3. **Actionnabilité :** Permet aux managers de proposer des solutions ciblées avant la démission.

---

## 📊 Source des données (Dataset)
Ce projet utilise le dataset synthétique public **"IBM HR Analytics Employee Attrition & Performance"** créé par les data scientists d'IBM et hébergé sur [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset). 
Ce jeu de données a été conçu spécifiquement à des fins éducatives pour démontrer la valeur de l'analytique RH.

## 🧪 Comment tester l'application ?

Pour tester l'application en direct sur Streamlit, vous pouvez télécharger et utiliser le fichier de démonstration fourni dans ce dépôt :
👉 **[Télécharger le fichier de test : `HR_Analytics_Data.csv`](./HR_Analytics_Data.csv)**

**Format requis pour vos propres données :**
Si vous souhaitez tester l'application avec vos propres données d'entreprise, votre fichier CSV doit contenir (au minimum) les colonnes suivantes avec la même casse :
* `Age` (Numérique)
* `BusinessTravel` (Texte : Non-Travel, Travel_Rarely, Travel_Frequently)
* `Department` (Texte)
* `DistanceFromHome` (Numérique)
* `JobRole` (Texte)
* `MonthlyIncome` (Numérique)
* `OverTime` (Texte : Yes, No)

*L'application est capable de nettoyer et d'encoder automatiquement les données textuelles grâce au pipeline intégré.*

---

## 🚀 Fonctionnalités
- Upload de fichiers de données RH au format CSV.
- Algorithme **Random Forest Classifier** optimisé.
- Interface web interactive propulsée par **Streamlit**.
- Analyse SHAP en temps réel pour le profil le plus critique.

---

## 🛠️ Stack Technique
* **Data Science :** Pandas, NumPy, Scikit-Learn
* **Explainable AI :** SHAP, Matplotlib
* **Déploiement :** Streamlit Cloud

---

## 🐛 Défis techniques surmontés
1. **Bug d'exécution silencieuse sous Windows 11 :** Contournement du conflit avec l'App Execution Alias de Microsoft via la migration de l'entraînement vers un environnement Jupyter Notebook local (`.ipynb`).
2. **Incompatibilité SHAP Waterfall Plot et Scikit-Learn :** Refactorisation du code d'explicabilité en utilisant la nouvelle API `shap.plots.waterfall(shap_obj[0, :, 1])` pour gérer la tridimensionnalité des sorties d'un Random Forest multiclasse.

---

## 📄 Licence
Code sous licence **MIT**. Le dataset original reste la propriété d'IBM / Kaggle et est utilisé ici dans un cadre démonstratif et éducatif.
