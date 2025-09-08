# ADEN-ML101: Curso Práctico de Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter)](https://jupyter.org/)
[![Colab](https://img.shields.io/badge/Google-Colab-yellow.svg?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Library-f7931e.svg?logo=scikitlearn)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)

Este repositorio contiene el material de apoyo, lecturas y laboratorios prácticos para el curso **Machine Learning 101**.  
El curso sigue un enfoque **90% práctico**, apoyado en **Google Colab** y **GitHub** para la gestión de versiones.

---

## Objetivos
- Comprender los fundamentos del Machine Learning y sus aplicaciones.
- Diferenciar entre aprendizaje supervisado y no supervisado.
- Implementar modelos de regresión, clasificación y ensambles.
- Evaluar métricas, evitar el overfitting y aplicar optimización de modelos.
- Desarrollar un proyecto de Machine Learning de principio a fin.

---

## Sesiones y Materiales

| Sesión | Tema | Lectura Fundamental | Módulo de Google | Laboratorio |
|--------|------|---------------------|------------------|-------------|
| 1 | ¿Qué es el ML? | ["The Bitter Lesson"](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) | [Problem Framing](https://developers.google.com/machine-learning/problem-framing/problem?hl=es-419) | [Cap. 1: Panorama ML](https://github.com/gonzalezulises/handson-ml3/blob/main/01_the_machine_learning_landscape.ipynb) |
| 2 | Sistemas y Datos | ["Data-Centric AI"]([https://www.deeplearning.ai/the-batch/a-chat-with-andrew-ng-about-data-centric-ai/](https://landing.ai/data-centric-ai) | [Preparación de datos](https://developers.google.com/machine-learning/data-prep) | [Cap. 2: Proyecto ML](https://github.com/gonzalezulises/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb) |
| 3 | Supervisado vs. No Supervisado | ["Autonomous Machine Intelligence"](https://openreview.net/pdf?id=BZ5a1r-kVsf) | [Clustering](https://developers.google.com/machine-learning/clustering/overview?hl=es-419) | [Cap. 9: Unsupervised](https://github.com/gonzalezulises/handson-ml3/blob/main/09_unsupervised_learning.ipynb) |
| 4 | Ingeniería de Características | ["Word2Vec"](https://arxiv.org/abs/1301.3781) | [Feature Engineering](https://developers.google.com/machine-learning/data-prep) | [Cap. 13: Preprocesamiento](https://github.com/gonzalezulises/handson-ml3/blob/main/13_loading_and_preprocessing_data.ipynb) |
| 5 | Introducción a Scikit-Learn | ["Scikit-learn: Machine Learning in Python"](https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf) | [Crash Course ML](https://developers.google.com/machine-learning/crash-course?hl=es-419) | [Cap. 2: Proyecto ML](https://github.com/gonzalezulises/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb) |
| 6 | Modelos de Regresión | ["An Introduction to Statistical Learning"](https://www.statlearning.com/) (Cap. 3) | [Crash Course ML - Regresión](https://developers.google.com/machine-learning/crash-course?hl=es-419) | [Cap. 4: Regresión Lineal](https://github.com/gonzalezulises/handson-ml3/blob/main/04_training_linear_models.ipynb) |
| 7 | Clasificación | ["Nearest neighbor pattern classification"](https://ieeexplore.ieee.org/document/4037264) | [Clasificación de imágenes](https://developers.google.com/machine-learning/image-classification) | [Cap. 3: Clasificación](https://github.com/gonzalezulises/handson-ml3/blob/main/03_classification.ipynb) |
| 8 | Árboles de Decisión | ["Random Forests"](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf), ["XGBoost"](https://arxiv.org/abs/1603.02754) | [Decision Forests](https://developers.google.com/machine-learning/decision-forests?hl=es-419) | [Cap. 6: Árboles](https://github.com/gonzalezulises/handson-ml3/blob/main/06_decision_trees.ipynb), [Cap. 7: Ensambles](https://github.com/gonzalezulises/handson-ml3/blob/main/07_ensemble_learning_and_random_forests.ipynb) |
| 9 | Métricas de Evaluación | ["Precision-Recall vs ROC"](https://www.researchgate.net/publication/220387544_The_Relationship_Between_Precision-Recall_and_ROC_Curves) | [Testing & Debugging](https://developers.google.com/machine-learning/testing-debugging) | [Cap. 3: Métricas](https://github.com/gonzalezulises/handson-ml3/blob/main/03_classification.ipynb) |
| 10 | Overfitting y Validación | ["Dropout"](https://jmlr.org/papers/v15/srivastava14a.html) | [Generalización](https://developers.google.com/machine-learning/crash-course?hl=es-419) | [Cap. 2: Cross-Validation](https://github.com/gonzalezulises/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb) |
| 11 | Optimización | ["Adam"](https://arxiv.org/abs/1412.6980), ["Bayesian Optimization"](https://arxiv.org/abs/1206.2944) | [Producción ML](https://developers.google.com/machine-learning/crash-course?hl=es-419) | [Cap. 2: GridSearchCV](https://github.com/gonzalezulises/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb) |

---

## Requisitos
- Computadora con **Python 3.9+** instalado.
- Navegador para usar **Google Colab**.
- Cuenta de GitHub para clonar y versionar el repo.
- Conocimientos básicos de programación en Python.

---

## Cómo usar este repositorio
1. Clona el repo:
   ```bash
   git clone https://github.com/gonzalezulises/ADEN-ML101.git
   cd ADEN-ML101
