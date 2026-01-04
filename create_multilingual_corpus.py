import pandas as pd

# Fichiers d'entrée

FILES = {
    "fr": "./CSV/deft_train_fr_avecdoublons.csv",
    "it": "./CSV/deft_train_it_avecdoublons.csv",
    "en": "./CSV/deft_train_en_avecdoublons.csv"
}

# Fichier de sortie

OUTPUT = "deft_train_multilingue_avecdoublonss.csv"


# Chargement des CSV et ajout de la langue

dfs = {}

for langue, path in FILES.items():
    df = pd.read_csv(path)
    df["Langue"] = langue
    dfs[langue] = df.reset_index(drop=True)


# Entrelacement ligne par ligne

rows = []
max_len = max(len(df) for df in dfs.values())

for i in range(max_len):
    for langue in FILES:
        df = dfs[langue]
        if i < len(df):
            rows.append(df.iloc[i])


# Création du DataFrame final

df_multilingue = pd.DataFrame(rows)

df_multilingue.to_csv(OUTPUT, index=False, encoding="utf-8")
