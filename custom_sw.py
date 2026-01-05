# création de listes de stopwords personnalisées à partir d'un corpus donné.
# Nous avons tenté de jouer avec le paramètre de fréquence minimale pour obtenir des listes de différentes tailles.
# Le corpus utilisé est "deft_train_multilingue_sansdoublons.csv" car il contient toutes les langues.
# Les listes générées sont destinées à être utilisées dans le script models.py afin d'améliorer les performances des modèles de classification de texte.
# plus les listes étaient longues, et moins les performances étaient bonnes, probablement à cause de la suppression excessive d'informations pertinentes.
# Nous nous sommes donc cantonnées à des listes basiques de stopwords personnalisées en plus de nltk.
# ce script est juste un artefact de notre expérimentation.

import re
import pandas as pd
from collections import Counter

CSV_PATH = "CSV/deft_train_multilingue_sansdoublons.csv"
MIN_FREQ = 500

df = pd.read_csv(CSV_PATH)

def tokenize(text):
    text = text.lower()
    return re.findall(r"\b\w+\b", text)

custom = {"fr": [], "en": [], "it": []}

for lang in ["fr", "en", "it"]:
    texts = df[df["Langue"] == lang]["Discours"].astype(str)
    counter = Counter()

    for t in texts:
        counter.update(tokenize(t))

    for word, freq in counter.items():
        if freq >= MIN_FREQ and len(word) > 2:
            custom[lang].append(word)

for lang, words in custom.items():
    print(f"\n# Custom stopwords {lang} ({len(words)})")
    print(words)
 
