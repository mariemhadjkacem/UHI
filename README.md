# Projet P2M – Urban Heat Island (UHI) Detector

## Description générale

Le projet P2M est un site web dédié à l’étude et la détection de l’effet **îlot de chaleur urbain (Urban Heat Island - UHI)**. Il regroupe plusieurs aspects clés autour de ce phénomène : ses causes, ses impacts sur l’environnement et la santé publique, ainsi que des outils d’analyse avancés basés sur l’imagerie satellite et l’intelligence artificielle.

Le projet propose notamment une **segmentation automatique des différentes surfaces urbaines** à partir d’images satellite haute résolution, affichée sur une **carte interactive** permettant d’identifier précisément les zones concernées par l’effet UHI.

---

## Objectifs du projet

- **Comprendre et expliquer** les causes et impacts de l’îlot de chaleur urbain.  
- **Segmenter automatiquement** les différentes surfaces urbaines à l’aide d’un modèle de **réseau de neurones convolutifs (CNN)** basé sur l’architecture **Unet**.  
- **Prédire** l’indice UHI à l’aide d’un **réseau de neurones artificiels (ANN)**, un modèle de deep learning, basé sur différentes données climatiques et géospatiales.  
- Offrir une **interface web interactive** pour explorer les données, visualiser les segmentations et les résultats.

---

## Fonctionnalités principales

### 1. Analyse des causes et impacts  
- Présentation pédagogique des facteurs responsables de l’UHI.  
- Effets sur l’environnement et la santé publique.

### 2. Segmentation des surfaces urbaines  
- Modèle de segmentation d’images basé sur un **CNN de type Unet**.  
- Classification et segmentation des différentes surfaces (bâtiments, végétation, sols, etc.) sur images satellite.  
- Visualisation des résultats **à l’aide des masques prédits** sur une carte interactive.

### 3. Prédiction de l’indice UHI  
- Modèle **ANN** pour estimer l’indice de l’îlot de chaleur urbain (UHI index) à partir des données climatiques et géospatiales.

---

## Technologies utilisées

- Python (TensorFlow/Keras, scikit-learn, Streamlit, GDAL)  
- Deep Learning :  
  - **CNN (Unet)** pour la segmentation d’images  
  - **ANN** pour la prédiction de l’indice UHI  
- Imagerie satellite :  
  - Landsat 8 (multispectral images)  
  - NDVI (Normalized Difference Vegetation Index)  
  - LST (Land Surface Temperature)  
  - Emissivity  
  - Urban surface luminance  
  - Building footprints  
- Interface Web : Streamlit avec intégration de cartes interactives

---


