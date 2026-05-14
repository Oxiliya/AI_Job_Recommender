"""
=============================================================
 TAHAP OFFLINE: Preprocessing & Embedding Generation
=============================================================
 Jalankan SEKALI sebelum menjalankan aplikasi:
   python src/preprocess.py
=============================================================
"""

import os
import pickle
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "job_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_PKL = os.path.join(MODEL_DIR, "job_embeddings.pkl")
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def resolve_location(row) -> str:
    """
    Aturan lokasi:
    - city & country ada  → "Kota, Negara"
    - city NaN, country ada → "Negara"
    - keduanya NaN          → "Remote"
    """
    city    = row["city"]    if pd.notna(row["city"])    else None
    country = row["country"] if pd.notna(row["country"]) else None

    if city and country:
        return f"{city}, {country}"
    elif country:
        return country
    else:
        return "Remote"


def load_dataset(path: str) -> pd.DataFrame:
    print(f"[1/4] Membaca dataset: {path}")
    df = pd.read_csv(path, low_memory=False)

    required = {"job_title", "skills_required", "salary_usd", "remote_option", "city", "country"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    # Buat kolom location yang sudah di-resolve
    df["location"] = df.apply(resolve_location, axis=1)

    print(f"      ✓ {len(df):,} baris, {df['job_title'].nunique()} job title unik")
    print(f"      Distribusi location type:")
    city_country = (df["city"].notna() & df["country"].notna()).sum()
    country_only = (df["city"].isna()  & df["country"].notna()).sum()
    remote_only  = (df["city"].isna()  & df["country"].isna()).sum()
    print(f"        City+Country : {city_country:,}")
    print(f"        Country only : {country_only:,}")
    print(f"        Remote only  : {remote_only:,}")
    return df


def build_corpus(df: pd.DataFrame) -> list[str]:
    print("[2/4] Membangun corpus (job_title + skills_required)...")
    corpus = (
        df["job_title"].fillna("").str.strip()
        + " | "
        + df["skills_required"].fillna("").str.strip()
    ).tolist()
    print(f"      ✓ {len(corpus):,} entri")
    return corpus


def generate_embeddings(corpus: list[str]) -> np.ndarray:
    print(f"[3/4] Memuat SBERT '{SBERT_MODEL}'...")
    model = SentenceTransformer(SBERT_MODEL)
    print(f"      Encoding {len(corpus):,} baris (mungkin 5-15 menit untuk 260K baris)...")
    start = time.time()
    embeddings = model.encode(
        corpus, batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"      ✓ Selesai dalam {time.time()-start:.1f}s | shape: {embeddings.shape}")
    return embeddings


def save_artifacts(df, corpus, embeddings, path):
    print(f"[4/4] Menyimpan ke: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "embeddings": embeddings,
        "corpus":     corpus,
        "dataframe":  df,
        "model_name": SBERT_MODEL,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"      ✓ {os.path.getsize(path)/(1024*1024):.1f} MB tersimpan")


def main():
    print("=" * 60)
    print("  AI JOB RECOMMENDER — OFFLINE PREPROCESSING")
    print("=" * 60)
    df         = load_dataset(DATA_PATH)
    corpus     = build_corpus(df)
    embeddings = generate_embeddings(corpus)
    save_artifacts(df, corpus, embeddings, OUTPUT_PKL)
    print()
    print("  ✅ SELESAI! Jalankan: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
