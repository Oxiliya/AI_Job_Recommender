"""
=============================================================
 GEMINI LAYER v5 — Fixed Mode Detection
=============================================================
 Fix: extract_preferences menerima job_titles langsung dari
 engine (bukan global variable) sehingga mode detection
 salary_lookup selalu akurat sejak query pertama.
=============================================================
"""

import os
import re
import time

import google.generativeai as genai
from dotenv import load_dotenv

from src.recommender import UserPreferences

load_dotenv()
_GEMINI_MODEL = None

# ── Peta kota → nama resmi ──
VALID_CITIES = {
    "tokyo": "Tokyo", "london": "London", "new york": "New York",
    "dubai": "Dubai", "berlin": "Berlin", "san francisco": "San Francisco",
    "seattle": "Seattle", "denver": "Denver", "boston": "Boston",
    "toronto": "Toronto", "munich": "Munich", "singapore": "Singapore",
    "sydney": "Sydney", "amsterdam": "Amsterdam", "stockholm": "Stockholm",
    "paris": "Paris", "bangalore": "Bangalore", "amsterdam": "Amsterdam",
}

# ── Peta negara → nama resmi ──
VALID_COUNTRIES = {
    "singapore": "Singapore", "japan": "Japan",
    "uk": "UK", "united kingdom": "UK", "england": "UK",
    "usa": "USA", "united states": "USA", "america": "USA", "us": "USA",
    "uae": "UAE", "dubai": "UAE",
    "germany": "Germany", "canada": "Canada", "india": "India",
    "australia": "Australia", "sweden": "Sweden", "netherlands": "Netherlands",
    "france": "France", "brazil": "Brazil",
}

# ── Kata kunci yang menandakan query gaji ──
SALARY_QUERY_KEYWORDS = (
    "berapa gaji", "gaji berapa", "gaji untuk", "gaji posisi",
    "salary", "penghasilan", "pendapatan", "kisaran gaji",
    "gaji di", "bayaran", "kompensasi", "tunjukkan gaji",
    "berapa penghasilan", "rate gaji", "gaji rata", "berapa rata",
    "how much", "gaji seorang", "gaji sebagai",
)

# ── Pola kalimat yang jelas salary_lookup ──
# "gaji [posisi]", "[posisi] berapa", "[posisi] di [lokasi]"
SALARY_PATTERNS = [
    r"gaji\s+(.+?)(?:\s+di\s+|\s+berapa|\?|$)",
    r"(.+?)\s+berapa\s+gaji",
    r"berapa\s+gaji\s+(.+?)(?:\s+di\s+|\?|$)",
    r"salary\s+(?:for\s+)?(.+?)(?:\s+in\s+|\?|$)",
]


def get_gemini_model():
    global _GEMINI_MODEL
    if _GEMINI_MODEL is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY belum diatur!\n"
                "Dapatkan di: https://aistudio.google.com/app/apikey"
            )
        genai.configure(api_key=api_key)
        _GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")
    return _GEMINI_MODEL


def _detect_location(text_lower: str) -> str:
    """Deteksi lokasi — prioritas kota > negara."""
    # Cek multi-word kota dulu
    for key, val in sorted(VALID_CITIES.items(), key=lambda x: -len(x[0])):
        if key in text_lower:
            return val
    # Cek negara
    for key, val in sorted(VALID_COUNTRIES.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(key)}\b", text_lower):
            return val
    return ""


def _match_job_title(query: str, job_titles: list[str]) -> str:
    """
    Cari job title yang disebut dalam query.
    Strategi berlapis:
    1. Exact substring match (case-insensitive)
    2. Partial word match (semua kata dalam title ada di query)
    """
    q_lower = query.lower()

    # Pass 1: full title ada di query
    for title in job_titles:
        if title.lower() in q_lower:
            return title

    # Pass 2: semua kata dalam title ada di query (urutan bebas)
    for title in job_titles:
        words = title.lower().split()
        if len(words) >= 2 and all(w in q_lower for w in words):
            return title

    # Pass 3: minimal 50% kata dalam title cocok
    best_title = ""
    best_score = 0
    for title in job_titles:
        words = title.lower().split()
        matches = sum(1 for w in words if w in q_lower)
        score = matches / len(words)
        if score > best_score and score >= 0.5:
            best_score = score
            best_title = title

    return best_title


def extract_preferences(user_message: str, job_titles: list[str]) -> UserPreferences:
    """
    Rule-based extractor — 0 API call.
    job_titles diambil langsung dari engine.job_index.keys()
    sehingga selalu up-to-date.
    """
    text_lower = user_message.lower().strip()

    # ── Lokasi ──
    location = _detect_location(text_lower)

    # ── Remote ──
    remote_only = any(kw in text_lower for kw in
                      ("remote", "wfh", "kerja dari rumah", "work from home"))

    # ── Gaji minimum ──
    min_salary = 0
    for pat in [r"\$\s*(\d+)[kK]", r"\$\s*([\d,]+)",
                r"(\d+)[kK]\s*(?:usd|dolar)", r"(?:gaji|salary)\D{0,20}?(\d+)[kK]"]:
        m = re.search(pat, text_lower)
        if m:
            num = int(m.group(1).replace(",", ""))
            min_salary = num * 1000 if num < 1000 else num
            break

    # ── Top-K ──
    top_k = 5
    k_m = re.search(r"(\d+)\s*(?:lowongan|posisi|pekerjaan|job|karier|rekomendasi)", text_lower)
    if k_m:
        top_k = min(int(k_m.group(1)), 9)

    # ═══════════════════════════════════════════
    # MODE DETECTION — salary_lookup vs skill_search
    # ═══════════════════════════════════════════

    # Cek 1: ada keyword gaji?
    has_salary_kw = any(kw in text_lower for kw in SALARY_QUERY_KEYWORDS)

    # Cek 2: ada nama job title eksplisit di query?
    matched_title = _match_job_title(user_message, job_titles)

    # Cek 3: pola kalimat salary query (berapa gaji X, gaji X di Y, dsb)
    has_salary_pattern = False
    for pat in SALARY_PATTERNS:
        if re.search(pat, text_lower):
            has_salary_pattern = True
            break

    # → salary_lookup jika: ada keyword/pola gaji DAN ada nama posisi
    # → salary_lookup juga jika: ada nama posisi + lokasi (implisit tanya gaji)
    is_salary_lookup = (
        (has_salary_kw or has_salary_pattern) and bool(matched_title)
    ) or (
        bool(matched_title) and bool(location) and not any(
            skill_kw in text_lower for skill_kw in
            ("skill", "cocok", "rekomendasi", "saya", "jago", "ahli", "bisa")
        )
    )

    if is_salary_lookup and matched_title:
        return UserPreferences(
            mode="salary_lookup",
            job_title=matched_title,
            location=location,
            remote_only=remote_only,
            min_salary=min_salary,
        )

    # ── Skill search — bersihkan noise dari query ──
    noise = [
        "saya", "jago", "ahli", "bisa", "mau", "minta", "cari", "ada",
        "lowongan", "pekerjaan", "kerja", "rekomendasi", "gaji", "minimal",
        "minimum", "remote", "wfh", "dari rumah", "work from home",
        "dengan", "untuk", "yang", "dan", "atau", "tolong", "bantu",
        "tunjukkan", "carikan", "berikan", "cocok", "cocokkan", "posisi",
        "karier", "pekerjaan apa", "apa saja",
    ]
    skills_text = user_message
    for nw in noise:
        skills_text = re.sub(rf"\b{re.escape(nw)}\b", " ", skills_text, flags=re.IGNORECASE)
    skills_text = re.sub(r"\s+", " ", skills_text).strip()

    return UserPreferences(
        mode="skill_search",
        skills=skills_text or user_message,
        location=location,
        min_salary=min_salary,
        remote_only=remote_only,
        top_k=top_k,
    )


# ── Prompt Mode 1: Skill Search ──
_PROMPT_SKILL = """Kamu adalah konsultan karier profesional. Sistem AI telah menemukan posisi pekerjaan yang cocok dengan skill user.
Sajikan dalam Bahasa Indonesia yang ramah dan informatif.

ATURAN WAJIB:
- Gunakan HANYA data di bawah, jangan mengarang
- JANGAN tampilkan skor atau angka match apapun
- Gaji: tampilkan sebagai rata-rata (contoh: ~$150,000/tahun)
- Remote: tampilkan statusnya (persentase Remote/Hybrid/Onsite)
- Format markdown rapi, gunakan emoji
- Tutup dengan 1 kalimat saran karier singkat

Data posisi yang direkomendasikan:
{jobs}

Berikan respons:"""

# ── Prompt Mode 2: Salary Lookup ──
_PROMPT_SALARY = """Kamu adalah konsultan karier profesional. Berikut data gaji berdasarkan lokasi untuk posisi yang diminta user.
Sajikan dalam Bahasa Indonesia yang terstruktur dan mudah dibaca.

ATURAN WAJIB:
- Gunakan HANYA data di bawah, jangan mengarang
- Tampilkan breakdown per lokasi, sudah diurutkan dari gaji rata-rata tertinggi
- Lokasi "Remote" artinya posisi tersedia fully remote tanpa kota/negara tertentu
- Format tabel atau list rapi, gunakan emoji
- Tutup dengan 1 insight singkat (lokasi dengan gaji terbaik)

Data gaji per lokasi:
{salary_data}

Berikan respons:"""


def humanize_response(user_message: str, prefs: UserPreferences, results,
                      max_retries: int = 2) -> str:
    model = get_gemini_model()

    if prefs.mode == "salary_lookup":
        loc_lines = [
            f"- {c['location']}: rata-rata ${c['avg']:,}/thn "
            f"(min ${c['min']:,} – maks ${c['max']:,}) | {c['remote_label']} | {c['count']} data"
            for c in results["by_location"]
        ]
        salary_data = (
            f"Posisi: {results['job_title']}\n"
            f"Rata-rata global: ${results['global_avg']:,}/tahun\n\n"
            f"Breakdown per lokasi (diurutkan dari gaji tertinggi):\n"
            + "\n".join(loc_lines)
        )
        prompt = _PROMPT_SALARY.format(salary_data=salary_data)
    else:
        job_lines = [
            f"#{i+1}: {r['job_title']} | Skills: {', '.join(r['skills_req'])} | "
            f"Rata-rata ${r['avg_salary']:,}/thn | {r['remote_label']}"
            for i, r in enumerate(results)
        ]
        prompt = _PROMPT_SKILL.format(jobs="\n".join(job_lines))

    for attempt in range(max_retries + 1):
        try:
            return model.generate_content(prompt).text
        except Exception as e:
            if "429" in str(e) and attempt < max_retries:
                time.sleep(5 * (attempt + 1))
                continue
            raise


def test_gemini_connection() -> tuple[bool, str]:
    try:
        resp = get_gemini_model().generate_content("Jawab dengan 'OK' saja.")
        return True, f"Koneksi berhasil: {resp.text.strip()}"
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error API: {str(e)}"