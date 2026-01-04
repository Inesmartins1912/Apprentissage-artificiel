import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.model_selection import train_test_split

# Mise en place de la reproductibilité

RANDOM_SEED = 57

# Listes des partis vides

PARTIS = {
    "ELDR": [],
    "GUE-NGL": [],
    "PPE-DE": [],
    "PSE": [],
    "Verts-ALE": []
}

# Fonctions d'extraction du texte

def extract_text_from_doc(doc):
    paragraphs = doc.findall(".//p")
    text = " ".join(
        " ".join(p.itertext()).strip()
        for p in paragraphs
        if "".join(p.itertext()).strip()
    )
    return text

# Fonctions de parsing

def parse_train(path_train):
    partis = {p: [] for p in PARTIS}

    tree = ET.parse(path_train)
    root = tree.getroot()

    for doc in root.findall(".//doc"):
        parti_elem = doc.find(".//PARTI")
        if parti_elem is None:
            continue

        parti = parti_elem.get("valeur")
        if parti not in partis:
            continue

        texte = extract_text_from_doc(doc)
        if texte:
            partis[parti].append(texte)

    return partis

def parse_test(path_test, path_ref):
    partis = {p: [] for p in PARTIS}

    # Fichier de référence : id -> parti
    id_to_parti = {}
    with open(path_ref, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                continue

            doc_id, parti = parts
            if parti in partis:
                id_to_parti[doc_id] = parti

    # Parsing XML
    tree = ET.parse(path_test)
    root = tree.getroot()

    for doc in root.findall(".//doc"):
        doc_id = doc.get("id")
        if doc_id not in id_to_parti:
            continue

        parti = id_to_parti[doc_id]
        texte = extract_text_from_doc(doc)

        if texte:
            partis[parti].append(texte)

    return partis

# Pipeline principale

train = parse_train(
    "./Fichiers_xml_et_txt/deft09_parlement_appr_fr.xml"
)

test = parse_test(
    path_test = "./Fichiers_xml_et_txt/deft09_parlement_appr_fr.xml",
    path_ref = "./Fichiers_xml_et_txt/deft09_parlement_ref_fr.txt"
)

# Fusion train/test
all_partis = {
    parti: train.get(parti, []) + test.get(parti, [])
    for parti in PARTIS
}

texts = []
labels = []

for parti, discours in all_partis.items():
    texts.extend(discours)
    labels.extend([parti] * len(discours))

df = pd.DataFrame({
    "Discours": texts,
    "Parti": labels
})

# Suppression des doublons à mettre ou non selon ce que l'on veut comme résultat
df = df.drop_duplicates(subset = ["Discours"])

# Équilibrage et mélange des classes
df = (
    df.groupby("Parti", group_keys = False)
      .apply(lambda x: x.sample(n = min(len(x), 2700), random_state = RANDOM_SEED))
      .sample(frac = 1, random_state = RANDOM_SEED)
      .reset_index(drop=True)
)

df_train, df_test = train_test_split(
    df,
    stratify = df["Parti"],
    random_state = RANDOM_SEED
)

# Export CSV
df_train.to_csv("deft_train.csv", index=False, encoding = "utf-8")
df_test.to_csv("deft_test.csv", index=False, encoding = "utf-8")
