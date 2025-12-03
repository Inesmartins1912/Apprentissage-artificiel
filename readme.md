# Projet DEFT 2009 — Classification des interventions par parti politique

Ce dépôt contient un **notebook Jupyter unique** qui met en œuvre la **tâche 3 de l’édition 2009 du DÉfi Fouille de Texte (DEFT)** : 
la **classification supervisée des interventions du Parlement européen selon le parti politique**.

Toute la démarche (vérification XML, extraction des données, analyse, apprentissage, évaluation) est réalisée **exclusivement dans le notebook**, sans scripts séparés.

---

## 1. Objectif du projet

L’objectif est de :

- **charger et analyser le corpus DEFT 2009** fourni au format XML,
- **vérifier la structure réelle des fichiers XML**,
- **extraire les textes et les étiquettes de partis politiques**,
- **entraîner plusieurs classifieurs supervisés**,
- **évaluer leurs performances**,
- et **comparer les résultats aux scores de référence DEFT 2009**.

Ce travail s’inscrit dans le cadre du projet final du cours d'apprentissage artificiel.

---

## 2. Organisation du dépôt

```bash
projet_final/
├── deft09/
│   ├── Corpus d_apprentissage/
│   │   └── deft09_parlement_appr_fr.xml
│   └── Corpus de test/
│       └── deft09_parlement_test_fr.xml
│
├── projet_deft2009_jupyterlab.ipynb
└── README.md
````

Les deux fichiers XML utilisés sont :

```python
projet_final/deft09/Corpus d_apprentissage/deft09_parlement_appr_fr.xml
projet_final/deft09/Corpus de test/deft09_parlement_test_fr.xml
```

---

## 3. Démarche méthodologique dans le notebook

Le notebook suit une **progression logique stricte**, conçue pour être reproductible et compréhensible par tout contributeur.

### Étape 1 — Chargement et vérification de la structure XML

Avant toute modélisation, le notebook :

* charge les fichiers XML avec `xml.etree.ElementTree`,
* affiche :

  * la balise racine (`<corpus>`),
  * les balises enfants (`<doc>`, `<description>`),
* inspecte un **premier document réel** :

  * attribut `id`,
  * balise `EVALUATION/EVAL_PARTI/PARTI`,
  * contenu textuel dans `<texte>/<p>`.

Cette étape garantit que :

* les chemins XML sont corrects,
* les annotations de partis sont bien présentes,
* la structure correspond aux attentes du sujet.

---

### Étape 2 — Extraction des données dans un DataFrame

Pour chaque balise `<doc>` du corpus d’apprentissage, on extrait :

* `id_intervention` : attribut `id` du document,
* `parti` : attribut `valeur` de la balise `<PARTI>`,
* `nombre_mots` : attribut `nombre` si présent,
* `texte` : concaténation du contenu des balises `<p>`.

Ces informations sont stockées dans un `DataFrame pandas` nommé `df_apprentissage`.

---

### Étape 3 — Analyse exploratoire

Le notebook calcule ensuite :

* le nombre total d’interventions,
* la liste des partis politiques présents,
* la répartition des classes,
* la longueur moyenne, minimale et maximale des interventions.

Cette étape met en évidence :

* le **déséquilibre des classes**,
* la **variabilité importante des longueurs de texte**.

---

### Étape 4 — Préparation des données pour l’apprentissage

Les données sont transformées sous la forme :

```python
X_train = df_apprentissage["texte"]
y_train = df_apprentissage["parti"]
```

---

### Étape 5 — Vectorisation TF–IDF

Le texte est vectorisé avec :

* `TfidfVectorizer`,
* `max_features = 20000`,
* `min_df = 5`,
* `max_df = 0.8`,
* `ngram_range = (1, 2)`.

Ce choix permet de :

* réduire le bruit,
* conserver l’information discriminante,
* rester compatible avec un entraînement sur machine standard.

---

### Étape 6 — Entraînement des modèles

Plusieurs classifieurs supervisés sont entraînés :

* **Naive Bayes multinomial (baseline)**,
* **SVM linéaire (LinearSVC)**,
* **Régression logistique**.

Une **validation croisée stratifiée** est utilisée avec la métrique :

* `F1-macro`.

---

### Étape 7 — Évaluation et comparaison avec DEFT 2009

Le notebook affiche :

* la moyenne des scores `F1-macro`,
* l’écart-type,
* la comparaison directe avec le score de référence DEFT 2009.

Cela permet de mesurer :

* les progrès,
* ou les écarts par rapport aux systèmes officiels.

---

## 4. Exécution du notebook

### Prérequis

* Python 3.12+
* Bibliothèques :

  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `matplotlib` (si visualisations activées)

### Lancement

Depuis la racine du dépôt :

```bash
jupyter lab projet_deft2009_jupyterlab.ipynb
```

Puis exécuter **toutes les cellules dans l’ordre**.



---

## 5. Finalité du projet

Ce notebook constitue :

* la **base expérimentale du rapport final** (format article ACL),
* un **pipeline reproductible** pour tous les contributeurs,
* un support d’expérimentation pour tester :

  * d’autres vectorisations,
  * d’autres classifieurs,
  * ou d’autres réglages d’hyperparamètres.

---

Si vous reprenez ce projet :

* commencez toujours par vérifier la structure XML,
* puis vérifiez la distribution des classes,
* avant d’ajouter de nouveaux modèles.

```



