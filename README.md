# 👥 TalentKeep AI - Prédiction de l'Attrition (Turnover)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hr-attrition-predictor-ai-thierrymaesen.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Le Problème
Le turnover coûte cher aux entreprises (recrutement, formation, perte de productivité). Souvent, les Ressources Humaines découvrent le mécontentement d'un employé au moment où il dépose sa démission, lorsqu'il est déjà trop tard pour agir.

## 💡 La Solution & Value Proposition
**TalentKeep AI** est une application web de Machine Learning conçue pour les équipes RH. 
Elle ne se contente pas de prédire *qui* va partir, elle explique *pourquoi*. 

**Plus-value du projet :**
1. **Anticipation :** Analyse des bases de données RH pour identifier les employés à haut risque (>60% de chances de départ).
2. **Explicabilité (Explainable AI) :** Utilisation de la librairie SHAP pour générer un graphique en cascade (Waterfall). Les RH peuvent voir immédiatement les "points de friction" (ex: salaire, trajet) et les "points d'ancrage" propres à chaque individu.
3. **Actionnabilité :** Permet aux managers de proposer des solutions ciblées avant la démission.

---

## 🚀 Fonctionnalités
- Upload de fichiers de données RH au format CSV.
- Nettoyage et encodage automatique des données textuelles.
- Algorithme **Random Forest Classifier** optimisé pour les classes déséquilibrées.
- Interface web interactive propulsée par **Streamlit**.
- Analyse SHAP en temps réel pour le profil le plus critique de l'équipe.

---

## 🛠️ Stack Technique
* **Data Science :** Pandas, NumPy, Scikit-Learn
* **Machine Learning :** Random Forest, Joblib (sauvegarde des modèles)
* **Explainable AI :** SHAP, Matplotlib
* **Déploiement :** Streamlit Cloud

---

## 🐛 Défis techniques rencontrés & Solutions apportées

Lors du développement de ce projet, plusieurs obstacles techniques ont été surmontés :

1. **Bug d'exécution silencieuse sous Windows 11 :**
   * *Problème :* Le script Python d'entraînement se fermait instantanément sans erreur à cause du conflit avec l'App Execution Alias du Microsoft Store.
   * *Solution :* Migration de l'entraînement vers un environnement Jupyter Notebook local (`.ipynb`) dans VS Code, permettant un suivi pas-à-pas de l'exécution et la génération réussie des fichiers `.pkl`.

2. **Incompatibilité SHAP Waterfall Plot et Scikit-Learn :**
   * *Problème :* Erreur `IndexError` lors de la génération du graphique SHAP sur l'application Streamlit, due au format tridimensionnel des valeurs de sortie du Random Forest multiclasse.
   * *Solution :* Refactorisation du code d'explicabilité en utilisant la nouvelle API `shap.plots.waterfall(shap_obj[0, :, 1])` pour cibler précisément la classe positive (risque de démission).

---

## 💻 Installation en local

Si vous souhaitez faire tourner ce projet sur votre machine :

```bash
# 1. Cloner le dépôt
git clone https://github.com/thierrymaesen/HR-Attrition-Predictor-AI.git

# 2. Se rendre dans le dossier
cd HR-Attrition-Predictor-AI

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application web
streamlit run app.py
