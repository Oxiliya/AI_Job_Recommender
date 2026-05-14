"""
=============================================================
 CORE ENGINE v5 — Dual Mode + Improved Skill Matching
=============================================================
 Fix: job index menggunakan top-15 skill dominan (bukan semua
 skill unik) + exact skill match bonus untuk scoring yang
 lebih relevan.

 MODE 1 (skill_search): Skill → posisi yang cocok
 MODE 2 (salary_lookup): Posisi → gaji per lokasi
=============================================================
"""

import os
import pickle
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_PATH    = os.path.join(BASE_DIR, "models", "job_embeddings.pkl")
INDEX_PKL   = os.path.join(BASE_DIR, "models", "job_index.pkl")
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

SKIP_LOC = ("", "semua", "all", "tidak spesifik", "any")

ALL_JOB_TITLES: list[str] = []


@dataclass
class UserPreferences:
    mode:        str  = "skill_search"
    skills:      str  = ""
    job_title:   str  = ""
    location:    str  = ""
    min_salary:  int  = 0
    remote_only: bool = False
    top_k:       int  = 5


def resolve_location(row) -> str:
    city    = row["city"]    if pd.notna(row.get("city"))    else None
    country = row["country"] if pd.notna(row.get("country")) else None
    if city and country:
        return f"{city}, {country}"
    elif country:
        return country
    else:
        return "Remote"


class JobRecommender:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self) -> bool:
        if self._initialized:
            return True
        if not os.path.exists(PKL_PATH):
            return False

        print("[Engine] Memuat artefak dataset...")
        with open(PKL_PATH, "rb") as f:
            data = pickle.load(f)
        self.df: pd.DataFrame = data["dataframe"]

        if "location" not in self.df.columns:
            self.df["location"] = self.df.apply(resolve_location, axis=1)

        global ALL_JOB_TITLES
        ALL_JOB_TITLES = sorted(self.df["job_title"].dropna().unique().tolist())

        print(f"[Engine] Memuat SBERT: {SBERT_MODEL}")
        self.model = SentenceTransformer(SBERT_MODEL)

        # Load cached job index jika ada, rebuild jika tidak
        if os.path.exists(INDEX_PKL):
            print("[Engine] Memuat job index dari cache...")
            with open(INDEX_PKL, "rb") as f:
                self.job_index = pickle.load(f)
        else:
            print("[Engine] Membangun job index (pertama kali, ~30 detik)...")
            self._build_job_index()
            with open(INDEX_PKL, "wb") as f:
                pickle.dump(self.job_index, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("[Engine] Job index disimpan ke cache.")

        self._initialized = True
        print(f"[Engine] ✓ Siap! {len(self.job_index)} posisi | {len(self.df):,} baris")
        return True

    def _build_job_index(self):
        """
        Satu embedding representatif per job title.
        Gunakan TOP-15 skill paling dominan agar vektor fokus & tidak noise.
        Simpan juga skill_set lengkap untuk exact-match boosting.
        """
        self.job_index = {}
        titles, corpus, subdfs, skill_sets = [], [], {}, {}

        for title, sub in self.df.groupby("job_title"):
            # Hitung frekuensi tiap skill
            skill_freq: dict[str, int] = {}
            for row in sub["skills_required"].dropna():
                for s in str(row).split(","):
                    s = s.strip()
                    if s:
                        skill_freq[s] = skill_freq.get(s, 0) + 1

            # Top-15 skill terdominan untuk corpus embedding
            top15 = sorted(skill_freq, key=skill_freq.get, reverse=True)[:15]
            # Semua skill (lowercase) untuk exact match bonus
            skill_sets[title] = {s.lower() for s in skill_freq}

            titles.append(title)
            corpus.append(f"{title}. Key skills: {', '.join(top15)}")
            subdfs[title] = sub.reset_index(drop=True)

        vecs = self.model.encode(
            corpus, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=True,
            batch_size=64,
        )
        for i, title in enumerate(titles):
            self.job_index[title] = {
                "vec":       vecs[i],
                "df":        subdfs[title],
                "skill_set": skill_sets[title],
            }

    # ──────────────────────────────────────────────────
    # HELPER: Filter lokasi
    # ──────────────────────────────────────────────────
    def _filter_by_location(self, sub: pd.DataFrame, location: str) -> pd.DataFrame:
        if not location or location.lower().strip() in SKIP_LOC:
            return sub
        mask = sub["location"].str.lower().str.contains(
            location.lower().strip(), na=False
        )
        filtered = sub[mask]
        return filtered if len(filtered) > 0 else sub

    # ──────────────────────────────────────────────────
    # HELPER: Filter remote
    # ──────────────────────────────────────────────────
    def _filter_remote(self, sub: pd.DataFrame) -> pd.DataFrame:
        r = sub[sub["remote_option"].str.lower() == "remote"]
        return r if len(r) > 0 else sub

    # ──────────────────────────────────────────────────
    # HELPER: Remote label
    # ──────────────────────────────────────────────────
    def _remote_label(self, counts: pd.Series) -> str:
        total = counts.sum()
        if total == 0:
            return "❓ Data tidak tersedia"
        parts = []
        for key, label in [("remote", "Remote"), ("hybrid", "Hybrid"), ("onsite", "Onsite")]:
            pct = counts.get(key, 0) / total
            if pct > 0:
                parts.append(f"{label} {pct:.0%}")
        return " | ".join(parts) if parts else "❓"

    # ──────────────────────────────────────────────────
    # MODE 1: Skill Search — semantic + exact match bonus
    # ──────────────────────────────────────────────────
    def skill_search(self, prefs: UserPreferences) -> list[dict]:
        # Tokenize skill user untuk exact match
        user_skills = {
            s.strip().lower()
            for s in re.split(r"[,\s]+", prefs.skills)
            if len(s.strip()) > 1
        }

        query_vec = self.model.encode(
            prefs.skills, normalize_embeddings=True, convert_to_numpy=True
        )

        # Score = cosine similarity + exact skill match bonus
        scored = []
        for title, info in self.job_index.items():
            semantic = float(np.dot(info["vec"], query_vec))
            exact_matches = len(user_skills & info["skill_set"])
            # Setiap skill yang cocok persis +0.05, max bonus 0.25
            skill_bonus = min(exact_matches * 0.05, 0.25)
            scored.append((title, semantic + skill_bonus))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for title, _ in scored[: prefs.top_k]:
            sub = self.job_index[title]["df"].copy()

            if prefs.location and prefs.location.lower().strip() not in SKIP_LOC:
                sub = self._filter_by_location(sub, prefs.location)
            if prefs.remote_only:
                sub = self._filter_remote(sub)
            if prefs.min_salary > 0:
                s = sub[sub["salary_usd"] >= prefs.min_salary * 0.8]
                if len(s) > 0:
                    sub = s

            # Top-5 skill paling sering di posisi ini
            skill_freq: dict[str, int] = {}
            for row in sub["skills_required"].dropna():
                for sk in str(row).split(","):
                    sk = sk.strip()
                    if sk:
                        skill_freq[sk] = skill_freq.get(sk, 0) + 1
            top_skills = sorted(skill_freq, key=skill_freq.get, reverse=True)[:5]

            avg_salary = int(sub["salary_usd"].mean())
            remote_counts = sub["remote_option"].str.lower().value_counts()
            remote_label = self._remote_label(remote_counts)

            results.append({
                "job_title":    title,
                "skills_req":   top_skills,
                "avg_salary":   avg_salary,
                "remote_label": remote_label,
            })

        return results

    # ──────────────────────────────────────────────────
    # MODE 2: Salary Lookup per Lokasi
    # ──────────────────────────────────────────────────
    def salary_lookup(self, prefs: UserPreferences) -> dict:
        matched_title = self._match_job_title(prefs.job_title)
        sub = self.job_index[matched_title]["df"].copy()

        if prefs.remote_only:
            sub = self._filter_remote(sub)
        if prefs.min_salary > 0:
            s = sub[sub["salary_usd"] >= prefs.min_salary * 0.8]
            if len(s) > 0:
                sub = s
        if prefs.location and prefs.location.lower().strip() not in SKIP_LOC:
            sub = self._filter_by_location(sub, prefs.location)

        city_stats = []
        for loc, loc_df in sub.groupby("location"):
            remote_counts = loc_df["remote_option"].str.lower().value_counts()
            city_stats.append({
                "location":     loc,
                "avg":          int(loc_df["salary_usd"].mean()),
                "min":          int(loc_df["salary_usd"].min()),
                "max":          int(loc_df["salary_usd"].max()),
                "remote_label": self._remote_label(remote_counts),
                "count":        len(loc_df),
            })

        city_stats.sort(key=lambda x: x["avg"], reverse=True)

        return {
            "job_title":   matched_title,
            "global_avg":  int(sub["salary_usd"].mean()),
            "by_location": city_stats,
        }

    # ──────────────────────────────────────────────────
    # HELPER: Match job title dari input user
    # ──────────────────────────────────────────────────
    def _match_job_title(self, query: str) -> str:
        for t in ALL_JOB_TITLES:
            if query.lower() in t.lower() or t.lower() in query.lower():
                return t
        qvec = self.model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        return max(
            self.job_index.items(),
            key=lambda x: float(np.dot(x[1]["vec"], qvec))
        )[0]

    # ──────────────────────────────────────────────────
    # ROUTER
    # ──────────────────────────────────────────────────
    def recommend(self, prefs: UserPreferences):
        if not self._initialized:
            raise RuntimeError("Engine belum diinisialisasi.")
        if prefs.mode == "salary_lookup":
            return self.salary_lookup(prefs)
        return self.skill_search(prefs)