import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
import pickle

from sympy.logic.inference import valid

# 1. Siapkan direktori
os.makedirs("data2/news", exist_ok=True)

# 3. Load stopwords
def load_stopwords(path):
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

# Path ke semua file stopwords
stopwords_paths = [
    "stopwords/stopwords.en.txt",
    "stopwords/stopwords.kp20k.txt",
    "stopwords/stopwords.SE.txt",
    "stopwords/stopwords.twitter.txt"
]

# 6. Preprocessing ringan
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return text
def normalize_stopwords(stopwords_set):
    return set(
        word for word in
        (clean_text(w) for w in stopwords_set)
        if word.isalpha() and len(word) > 1
    )

# Gabungkan semua stopwords jadi satu set
custom_stopwords = set()
for path in stopwords_paths:
    stop_set = load_stopwords(path)
    custom_stopwords.update(normalize_stopwords(stop_set))

# Ubah jadi list (jika diperlukan oleh TfidfVectorizer)
custom_stopwords = list(custom_stopwords)

# 4. Load dataset
df = pd.read_json("content/data/News_Category_Dataset_v3.json", lines=True)

# 5. Gabungkan headline + short_description
df['text'] = df['headline'] + ". " + df['short_description']
df = df[['text']].dropna()

df['clean_text'] = df['text'].apply(clean_text)

# 7. TF-IDF

# with open("data/news/tfidf_vectorizer.pkl", "rb") as f:
#     vectorizer_new = pickle.load(f)

vectorizer = TfidfVectorizer(stop_words=custom_stopwords, max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
# X_new = vectorizer_new.fit_transform(df['clean_text'])
terms = vectorizer.get_feature_names_out()
# terms_new = vectorizer_new.get_feature_names_out()

def get_top_keywords(X_row, top_n=5):
    sorted_indices = X_row.toarray().flatten().argsort()[::-1]
    keywords = [terms[i] for i in sorted_indices[:top_n]]
    # keywords_new = [terms_new[i] for i in sorted_indices[:top_n]]
    return "; ".join(keywords)

df['keyphrases'] = [get_top_keywords(X[i], 5) for i in range(len(df))]
# df['keyphrases'] = [get_top_keywords(X_new[i], 5) for i in range(len(df))]

# 8. Split: train, valid, test
total = len(df)

train_end = int(0.7 * total)
valid_end = int(0.9 * total)

train, valid, test = df[:train_end], df[train_end:valid_end], df[valid_end:]

# 9. Simpan ke file
def save_to_file(df_part, prefix):
    with open(f"{prefix}.src", "w", encoding="utf-8") as src_file, \
         open(f"{prefix}.trg", "w", encoding="utf-8") as trg_file:
        for _, row in df_part.iterrows():
            src_file.write(row['text'].strip() + "\n")
            trg_file.write(row['keyphrases'].strip() + "\n")

save_to_file(train, "data2/news/train")
save_to_file(valid, "data2/news/valid")
save_to_file(test, "data2/news/test")

print(len(train))
print(len(valid))
print(len(test))