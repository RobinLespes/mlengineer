# ml_engineer

## Objectif du projet

Le projet **ml_engineer** a pour but de prédire le nombre de locations de vélos en libre-service (vlib) par heure, en fonction de plusieurs variables influençant la demande. Ce projet est principalement pédagogique et vise à illustrer les bonnes pratiques d'industrialisation d'un projet de Machine Learning. 

Nous utiliserons plusieurs outils pour montrer l'ensemble du cycle de vie d'un projet ML, incluant :

- **Random Forest** et **Gridsearch** pour l'entrainement du modèle
- **MLflow** pour la reproductibilité des expériences.
- **Flask** pour le déploiement du modèle en tant que service web.
- **Dash** pour la surveillance et la visualisation des performances du modèle.

Le modèle utilisé est un **Random Forest**, et nous utilisons une approche de **GridSearch** avec validation croisée pour l'optimisation des hyperparamètres.

## Architecture du projet

Le projet se divise en plusieurs modules pour couvrir les différentes étapes du cycle de vie du projet de Machine Learning :

1. **Préparation des données** : Nettoyage, transformation, et préparation des données nécessaires à l'entraînement du modèle.
2. **Entraînement du modèle** : Entraînement d'un modèle Random Forest avec GridSearch pour la recherche des meilleurs hyperparamètres.
3. **Tracking des expériences** : Utilisation de MLflow pour suivre les expériences, sauvegarder les modèles et les paramètres associés.
4. **Serving de modèle** : Utilisation de Flask pour déployer le modèle entraîné en tant que service RESTful, permettant des prédictions en temps réel.
5. **Monitoring du modèle** : Utilisation de Dash pour visualiser les performances du modèle et les prédictions en temps réel.

## Fonctionnalités

- **Prédiction** : Le modèle prédit le nombre de locations de vélos par heure en fonction de variables telles que la météo, l'heure de la journée, le jour de la semaine, etc.
- **Tracking des expériences** : Chaque entraînement du modèle est enregistré avec MLflow, permettant de retracer les différentes versions du modèle et d'évaluer les résultats obtenus.
- **Déploiement** : Le modèle peut être servi via une API Flask pour effectuer des prédictions en temps réel.
- **Monitoring** : Les performances du modèle peuvent être surveillées et visualisées à l'aide d'une application Dash.


### Prérequis

- Python 3.7+
- Pip

