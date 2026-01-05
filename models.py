# Entraînement et évaluation de modèles de classification
# Ici on utilise des modèles qu'on a principalement vu en cours notamment en fouille de texte l'an dernier.²

# en raison de la lenteur de randomforest, on peut choisir de ne pas l'exécuter avec --skip_rf
# exemple de commande:
# uv run python models.py --data_dir ./CSV --out_dir results --cv 2 --jobs 1 --skip_rf
# elle traitera tous les tableaux dans ./CSV avec svm et naive bayes seulement, et mettra les résultats dans ./results

# explications des arguments :
# --data_dir : dossier contenant les fichiers CSV d'entraînement et de test
# --out_dir : dossier de sortie pour les résultats
# --cv : nombre de folds pour la validation croisée dans la recherche d'hyperparamètres
# --jobs : nombre de jobs parallèles pour la recherche d'hyperparamètres
# --seed : seed fixe pour pouvoir refaire l'expérience à l'identique
# --only : ne traiter qu'un seul suffixe de dataset (ex: fr_sansdoublons)

import os
import re
import nltk
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, ConfusionMatrixDisplay 

# On a décidé d'utiliser des stopwords pour le nettoyage du corpus

# nltk.download("stopwords")

EXTRA_FR = {"monsieur","madame","président","présidente","vice","rapport","rapporteur","rapporteure",
    "commission","commissaire","parlement","parlementaire","européen","européenne","europe",
    "union","conseil","état","états","membre","membres","pays","gouvernement",
    "directive","règlement","amendement","amendements","proposition","propositions","projet",
    "texte","article","articles","vote","voter","votation","séance","session","plénière",
    "débat","débats","collègue","collègues","mesdames","messieurs","merci"}

EXTRA_EN = {"mr","madam","president","vice","rapporteur","report","reports",
    "commission","commissioner","parliament","parliamentary","european","europe",
    "union","council","state","states","member","members","country","government",
    "directive","regulation","amendment","amendments","proposal","proposals","text","article","articles",
    "vote","voting","session","plenary","debate","debates","colleague","colleagues",
    "ladies","gentlemen","thank","thanks"}

EXTRA_IT = {"signor","signora","presidente","vice","relatore","relazione","relazioni",
    "commissione","commissario","parlamento","parlamentare","europeo","europea","europa",
    "unione","consiglio","stato","stati","membro","membri","paese","governo",
    "direttiva","regolamento","emendamento","emendamenti","proposta","proposte","testo","articolo","articoli",
    "voto","votare","sessione","seduta","plenaria","dibattito","dibattiti","collega","colleghi",
    "grazie"}

STOP_FR = set(stopwords.words("french")) | EXTRA_FR
STOP_EN = set(stopwords.words("english")) | EXTRA_EN
STOP_IT = set(stopwords.words("italian")) | EXTRA_IT

def clean_text(text, lang):
    text = str(text).lower()
    words = text.split()

    if lang == "fr":
        sw = STOP_FR
    elif lang == "en":
        sw = STOP_EN
    elif lang == "it":
        sw = STOP_IT
    else:
        sw = set()

    words = [w for w in words if w not in sw]
    return " ".join(words)

def detect_lang(path): # on se base sur le nom du fichier si le csv ne contient pas de colonne Langue
    base = os.path.basename(path)
    if "_fr_" in base:
        return "fr"
    if "_en_" in base:
        return "en"
    if "_it_" in base:
        return "it"
    return None


def load_csv(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["Discours", "Parti"])

    if "Langue" in df.columns:
        df["Discours"] = df.apply(lambda row: clean_text(row["Discours"], row["Langue"]), axis=1)
    else:
        lang = detect_lang(path)
        if lang is not None:
            df["Discours"] = df["Discours"].apply(lambda t: clean_text(t, lang))
        else:
            # si on ne sait pas, on laisse tel quel
            df["Discours"] = df["Discours"].astype(str)

    X = df["Discours"].astype(str).tolist()
    y = df["Parti"].astype(str).tolist()
    return X, y


def macro_scores(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return float(p), float(r), float(f1) # on évalue avec f1-macro

def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred)) # on donne aussi l'accuracy pour information meme si moins pertinente dans le cas de classes déséquilibrées


def save_conf_matrix(y_true, y_pred, labels, title, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def save_f1_plot(results, out_png):
    names = [r["model"] for r in results]
    f1s = [r["F1_macro"] for r in results]

    fig, ax = plt.subplots()
    ax.bar(names, f1s)
    ax.set_ylabel("F1 macro")
    ax.set_title("Comparaison des modèles (F1 macro)")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def save_class_plot(df_cls, out_png, title):
    # df_cls colonnes: Parti, Precision, Recall (float)
    partis = df_cls["Parti"].tolist()
    prec = df_cls["Precision"].tolist()
    rec = df_cls["Recall"].tolist()

    fig, ax = plt.subplots()
    x = list(range(len(partis)))
    width = 0.4

    ax.bar([i - width/2 for i in x], prec, width=width, label="Precision")
    ax.bar([i + width/2 for i in x], rec, width=width, label="Recall")

    ax.set_xticks(x)
    ax.set_xticklabels(partis, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# On utilise SVM, Naive Bayes et Random Forest avec SVD pour réduire la dimensionnalité
def run(train_csv, test_csv, out_dir, cv, jobs, seed, skip_rf):
    os.makedirs(out_dir, exist_ok=True)

    X_train, y_train = load_csv(train_csv)
    X_test, y_test = load_csv(test_csv)
    labels = sorted(list(set(y_train) | set(y_test)))

    tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=50000) # on utilise des bigrammes aussi

    results = []

    # 1) SVM
    pipe_svm = Pipeline([("tfidf", tfidf), ("clf", LinearSVC())])
    grid_svm = {"clf__C": [0.3, 1.0, 3.0], "clf__class_weight": [None, "balanced"]}
    gs_svm = GridSearchCV(pipe_svm, grid_svm, scoring="f1_macro", cv=cv, n_jobs=jobs)
    gs_svm.fit(X_train, y_train)
    best_svm = gs_svm.best_estimator_
    pred_svm = best_svm.predict(X_test)
    p, r, f1 = macro_scores(y_test, pred_svm)
    acc = accuracy(y_test, pred_svm)
    results.append({"model": "SVM (LinearSVC)", "P_macro": p, "R_macro": r, "F1_macro": f1, "Accuracy": acc, "best_params": gs_svm.best_params_})

    # 2) Naive Bayes
    pipe_nb = Pipeline([("tfidf", tfidf), ("clf", MultinomialNB())])
    grid_nb = {"clf__alpha": [0.1, 0.5, 1.0]}
    gs_nb = GridSearchCV(pipe_nb, grid_nb, scoring="f1_macro", cv=cv, n_jobs=jobs)
    gs_nb.fit(X_train, y_train)
    best_nb = gs_nb.best_estimator_
    pred_nb = best_nb.predict(X_test)
    p, r, f1 = macro_scores(y_test, pred_nb)
    acc = accuracy(y_test, pred_nb)
    results.append({"model": "Naive Bayes (MultinomialNB)", "P_macro": p, "R_macro": r, "F1_macro": f1, "Accuracy": acc, "best_params": gs_nb.best_params_})

    pred_rf = None

    if not skip_rf:
        # 3) Random Forest + SVD 
        pipe_rf = Pipeline([("tfidf", tfidf), ("svd", TruncatedSVD(random_state=seed)), ("rf", RandomForestClassifier(random_state=seed, n_jobs=1))])
        grid_rf = {"svd__n_components": [200], "rf__n_estimators": [200, 400], "rf__max_depth": [None]}

        gs_rf = GridSearchCV(pipe_rf, grid_rf, scoring="f1_macro", cv=cv, n_jobs=jobs)
        gs_rf.fit(X_train, y_train)
        best_rf = gs_rf.best_estimator_
        pred_rf = best_rf.predict(X_test)
        p, r, f1 = macro_scores(y_test, pred_rf)
        acc = accuracy(y_test, pred_rf)
        results.append({"model": "RandomForest (TF-IDF+SVD)", "P_macro": p, "R_macro": r, "F1_macro": f1, "Accuracy": acc, "best_params": gs_rf.best_params_})

    results.sort(key=lambda d: d["F1_macro"], reverse=True)

    best = results[0]
    best_name = best["model"]
    if best_name.startswith("SVM"):
        best_pred = pred_svm
    elif best_name.startswith("Naive Bayes"):
        best_pred = pred_nb
    else:
        best_pred = pred_rf 
    if best_pred is not None:
        p_cls, r_cls, f1_cls, support_cls = precision_recall_fscore_support(y_test, best_pred, labels=labels, average=None, zero_division=0)
        df_cls = pd.DataFrame({"Parti": labels,"Precision": p_cls, "Recall": r_cls, "F1": f1_cls, "Support": support_cls})
        per_class_csv = os.path.join(out_dir, "best_per_class.csv")
        df_cls.to_csv(per_class_csv, index=False, encoding="utf-8")
        per_class_png = os.path.join(out_dir, "best_precision_recall_by_party.png")
        save_class_plot(df_cls, out_png=per_class_png, title=f"Precision / Recall par parti - {best_name}")

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    preds = [("SVM", pred_svm), ("NaiveBayes", pred_nb)]
    if pred_rf is not None:
        preds.append(("RandomForest", pred_rf))
        
    for name, pred in preds:
        save_conf_matrix(y_test, pred, labels, title=f"Matrice de confusion - {name}", out_png=os.path.join(out_dir, f"confusion_{name}.png"))

    save_f1_plot(results, os.path.join(out_dir, "compare_f1.png"))

    return results


def find_pairs(data_dir):
    """Recherche les paires de fichiers CSV train/test dans le dossier donné.
    Les fichiers doivent être nommés comme :
      deft_train_<suffix>.csv
      deft_test_<suffix>.csv
    Exemple suffix :
      en_avecdoublons
      fr_sansdoublons
      multilingue_avecdoublons
    """
    files = os.listdir(data_dir)
    train_re = re.compile(r"^deft_train_(.+)\.csv$")

    pairs = []
    for fn in files:
        m = train_re.match(fn)
        if not m:
            continue
        suffix = m.group(1)
        test_fn = f"deft_test_{suffix}.csv"
        if test_fn in files:
            pairs.append((
                suffix,
                os.path.join(data_dir, fn),
                os.path.join(data_dir, test_fn)
            ))

    pairs.sort(key=lambda x: x[0])
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--cv", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--skip_rf", action="store_true", help="Ne pas lancer RandomForest (très lent)")
    parser.add_argument("--only", default=None, help="Ne lancer qu'un dataset suffix (ex: fr_sansdoublons)")
    args = parser.parse_args()

    pairs = find_pairs(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    summary_rows = []
    for suffix, train_csv, test_csv in pairs:
        if args.only is not None and suffix != args.only:
            continue

        out_subdir = os.path.join(args.out_dir, suffix)
        results = run(train_csv, test_csv, out_subdir, args.cv, args.jobs, args.seed, args.skip_rf)
        best = results[0]
        print(f"Résultats pour le dataset '{suffix}':")
        print(f"Meilleur: {best['model']}  F1_macro={best['F1_macro']:.4f}  params={best['best_params']}")

        for r in results:
            summary_rows.append({"dataset": suffix, "model": r["model"], "P_macro": r["P_macro"], "R_macro": r["R_macro"], "F1_macro": r["F1_macro"], "Accuracy": r["Accuracy"], "best_params": json.dumps(r["best_params"], ensure_ascii=False)})

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

if __name__ == "__main__":
    main()
