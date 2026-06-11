# Projet Étudiant : Système de Détection de Chute pour Senior

Salut ! Bienvenue sur le dépôt de mon projet. 

L'objectif de ce travail est de concevoir un **système embarqué capable de détecter les chutes**, pensé spécifiquement pour la sécurité des personnes âgées. C'est un projet technique complet qui fait le pont entre l'intelligence artificielle (machine learning) et l'électronique embarquée (microcontrôleur).

---

## Architecture du dépôt

Pour que le projet soit facile à explorer, j'ai divisé le code en deux dossiers distincts, correspondant aux deux grandes phases de développement :

* **`CDC_Chute_Senior_v1.docx`** : Situé à la racine du dépôt, c'est le cahier des charges du projet. Vous y trouverez le contexte, l'étude du besoin, les contraintes techniques et l'architecture globale du système.

### Dossier `Colab_Python` (Conception de l'IA)
Ce dossier contient la partie "Recherche et Apprentissage" du modèle.
* **`DETECTION_CHUTE_code_Reseau_neuronne.ipynb`** : Le notebook (qui a tourné sur Google Colab). C'est ici que j'ai traité les données des capteurs et entraîné le réseau de neurones pour qu'il apprenne à différencier un mouvement classique (marcher, s'asseoir) d'une véritable chute.
* **`metadata.json`** : Fichier contenant les métadonnées et la configuration liées au modèle d'apprentissage.

### Dossier `Arduino` (Système Embarqué)
Ce dossier contient le code qui tourne physiquement sur la carte électronique.
* **`modele_chute.h`** : C'est le "cerveau" embarqué ! C'est le réseau de neurones (entraîné côté Python) qui a été converti en langage C pour être compréhensible par le microcontrôleur.
* **`Detection_chute.ino`** : Le programme principal de l'Arduino. Il s'occupe de lire les données des capteurs en temps réel, de les envoyer au modèle mathématique (`modele_chute.h`), et de déclencher la logique d'alerte en cas de chute avérée.

---

## La démarche technique (Comment ça marche ?)

Si vous souhaitez comprendre le flux du projet, voici la logique que j'ai suivie :
1.  **Entraînement (Cloud) :** Le modèle d'intelligence artificielle est d'abord entraîné sur Python, car cela demande de la puissance de calcul.
2.  **Conversion (TinyML) :** Une fois que l'IA est assez précise, le modèle est "figé" et converti en un fichier d'en-tête très léger (`.h`).
3.  **Inférence (Embarqué) :** L'Arduino utilise ce fichier léger pour faire ses prédictions en direct de manière totalement autonome, avec ses propres petites ressources de calcul.

N'hésitez pas à lire le cahier des charges pour plus de détails sur le matériel choisi et les scénarios d'usage. Toute remarque ou piste d'amélioration est la bienvenue !
