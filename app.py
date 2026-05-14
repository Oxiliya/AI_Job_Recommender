"""
=============================================================
 APLIKASI UTAMA: Streamlit Chatbot UI
=============================================================
 Jalankan dengan:
   streamlit run app.py

 Pastikan sudah menjalankan:
   python src/preprocess.py
 dan sudah mengisi .env dengan GEMINI_API_KEY
=============================================================
"""

import os
import sys
import time

import pandas as pd
import streamlit as st

# Tambahkan root dir ke sys.path agar import src.* berfungsi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gemini_layer import extract_preferences, humanize_response, test_gemini_connection
from src.recommender import JobRecommender, UserPreferences

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Job Recommender",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Warna utama */
    :root {
        --primary: #4F46E5;
        --bg-card: #1E1E2E;
    }

    /* Chat message styling */
    .chat-user {
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-bot {
        background: #F1F5F9;
        color: #1E293B;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 90%;
        border-left: 4px solid #4F46E5;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Status badge */
    .badge-online  { background: #D1FAE5; color: #065F46; padding: 4px 10px; border-radius: 20px; font-size: 12px; }
    .badge-offline { background: #FEE2E2; color: #991B1B; padding: 4px 10px; border-radius: 20px; font-size: 12px; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "engine_ready" not in st.session_state:
    st.session_state.engine_ready = False
if "gemini_ready" not in st.session_state:
    st.session_state.gemini_ready = False
if "last_recommendations" not in st.session_state:
    st.session_state.last_recommendations = None


# ──────────────────────────────────────────────
# INITIALIZE ENGINE (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_recommender():
    """Load engine sekali, cache di memory Streamlit."""
    engine = JobRecommender()
    success = engine.initialize()
    return engine, success


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Konfigurasi Sistem")
    st.divider()

    # ── API Key Input ──
    st.markdown("### 🔑 Gemini API Key")
    api_key_input = st.text_input(
        "Masukkan API Key",
        type="password",
        placeholder="AIza...",
        help="Dapatkan dari https://aistudio.google.com/app/apikey",
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input

    # ── Test Connection ──
    if st.button("🔌 Test Koneksi Gemini", use_container_width=True):
        with st.spinner("Menguji koneksi..."):
            ok, msg = test_gemini_connection()
            if ok:
                st.success(f"✅ {msg}")
                st.session_state.gemini_ready = True
            else:
                st.error(f"❌ {msg}")
                st.session_state.gemini_ready = False

    st.divider()

    # ── Engine Status ──
    st.markdown("### 📊 Status Sistem")
    engine, engine_ok = load_recommender()
    st.session_state.engine_ready = engine_ok

    col1, col2 = st.columns(2)
    with col1:
        if engine_ok:
            st.markdown('<span class="badge-online">✅ Engine</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-offline">❌ Engine</span>', unsafe_allow_html=True)
    with col2:
        if st.session_state.gemini_ready:
            st.markdown('<span class="badge-online">✅ Gemini</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-offline">⚠️ Gemini</span>', unsafe_allow_html=True)

    if not engine_ok:
        st.warning(
            "⚠️ **Embeddings belum dibuat!**\n\n"
            "Jalankan perintah berikut di terminal:\n"
            "```\npython src/preprocess.py\n```"
        )

    st.divider()

    # ── Stats ──
    if engine_ok:
        st.markdown("### 📈 Dataset Info")
        df_stats = engine.df
        st.metric("Total Lowongan", f"{len(df_stats):,}")
        st.metric("Kota Tersedia", df_stats["location"].nunique())
        st.metric("Remote Jobs", f"{(df_stats['remote_option'] == 'Yes').sum():,}")

    st.divider()

    # ── Contoh Pertanyaan ──
    st.markdown("### 💡 Contoh Pertanyaan")
    examples = [
        "Saya jago Python dan Machine Learning, cocok jadi apa?",
        "Skill saya React dan TypeScript, ada posisi remote?",
        "Berapa gaji Data Scientist di tiap lokasi?",
        "Gaji Software Engineer di UK berapa?",
        "Saya ahli Kubernetes dan Docker, rekomendasikan posisi",
        "Berapa gaji Backend Developer di Germany?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state["prefill_message"] = ex
            st.rerun()

    st.divider()

    # ── Clear Chat ──
    if st.button("🗑️ Bersihkan Riwayat Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_recommendations = None
        st.rerun()


# ──────────────────────────────────────────────
# MAIN AREA
# ──────────────────────────────────────────────
st.markdown("# 💼 AI Job Recommender")
st.markdown(
    "Sistem rekomendasi lowongan kerja berbasis **Hybrid RAG** — "
    "SBERT untuk semantic search + Gemini untuk percakapan alami."
)

# ── Architecture Info ──
with st.expander("🏗️ Arsitektur Sistem", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔵 NLP Engine (SBERT)**")
        st.markdown(
            "- Model: `paraphrase-multilingual-MiniLM-L12-v2`\n"
            "- Task: Cosine Similarity Search\n"
            "- Bobot Skill: **70%**"
        )
    with col2:
        st.markdown("**🟡 Filtering & Re-ranking**")
        st.markdown(
            "- Lokasi: Hard Filter (15%)\n"
            "- Gaji: Bonus Score (15%)\n"
            "- Remote: Hard Filter"
        )
    with col3:
        st.markdown("**🟢 Gemini Layer**")
        st.markdown(
            "- Ekstrak entitas dari input\n"
            "- Humanize hasil rekomendasi\n"
            "- *Tidak* memilih data"
        )

st.divider()

# ── CHAT HISTORY DISPLAY ──
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.info(
            "👋 Selamat datang! Ceritakan skill dan preferensi karier Anda, "
            "saya akan merekomendasikan lowongan terbaik untuk Anda."
        )
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

# ── RECOMMENDATION TABLE (jika ada) ──
if st.session_state.last_recommendations is not None:
    with st.expander("📋 Data Rekomendasi (Raw)", expanded=False):
        saved_prefs, saved_results = st.session_state.last_recommendations
        if saved_prefs.mode == "salary_lookup":
            rows = [
                {"Lokasi": c["location"], "Rata-rata": f"${c['avg']:,}",
                 "Min": f"${c['min']:,}", "Maks": f"${c['max']:,}",
                 "Remote": c["remote_label"], "Data": c["count"]}
                for c in saved_results["by_location"]
            ]
        else:
            rows = [
                {"Posisi": r["job_title"], "Skills": ", ".join(r["skills_req"]),
                 "Avg Salary": f"${r['avg_salary']:,}", "Remote": r["remote_label"]}
                for r in saved_results
            ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# CHAT INPUT
# ──────────────────────────────────────────────
# Handle prefill dari contoh pertanyaan di sidebar
prefill = st.session_state.pop("prefill_message", None)

user_input = st.chat_input(
    "Ceritakan skill dan preferensi karier Anda...",
    key="chat_input",
)

# Jika ada prefill dari sidebar, gunakan itu
if prefill and not user_input:
    user_input = prefill

# ──────────────────────────────────────────────
# HELPER: Fallback response tanpa Gemini
# ──────────────────────────────────────────────
def _build_fallback_response(prefs: UserPreferences, results) -> str:
    if prefs.mode == "salary_lookup":
        lines = [f"### 💰 Gaji untuk: **{results['job_title']}**\n",
                 f"Rata-rata global: **${results['global_avg']:,}/tahun**\n"]
        for c in results["by_location"]:
            lines.append(
                f"- **{c['location']}**: ${c['avg']:,}/thn "
                f"(${c['min']:,}–${c['max']:,}) | {c['remote_label']}"
            )
        return "\n".join(lines)
    else:
        lines = ["### 🎯 Posisi yang Cocok dengan Skill Anda\n"]
        for i, r in enumerate(results):
            lines.append(
                f"**{i+1}. {r['job_title']}**\n"
                f"- 🛠️ **Skills** : {', '.join(r['skills_req'])}\n"
                f"- 💰 **Gaji**   : rata-rata ${r['avg_salary']:,}/tahun\n"
                f"- 🌐 **Remote** : {r['remote_label']}\n"
            )
        return "\n".join(lines)


if user_input:
    # Validasi engine siap
    if not st.session_state.engine_ready:
        st.error(
            "❌ **Engine belum siap!** Jalankan `python src/preprocess.py` terlebih dahulu."
        )
        st.stop()

    # Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # ── Proses Rekomendasi ──
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Menganalisis preferensi Anda..."):
            try:
                prefs = extract_preferences(user_input, list(engine.job_index.keys()))
            except ValueError as e:
                st.warning(f"⚠️ Gemini tidak tersedia: {e}\nMenggunakan mode langsung.")
                prefs = UserPreferences(mode="skill_search", skills=user_input, top_k=5)

        with st.spinner("⚡ Memproses dengan SBERT..."):
            t_start = time.time()
            results = engine.recommend(prefs)
            elapsed = time.time() - t_start
            st.session_state.last_recommendations = (prefs, results)

        with st.spinner("✍️ Gemini sedang merangkum hasil..."):
            try:
                final_response = humanize_response(user_input, prefs, results)
            except Exception:
                final_response = _build_fallback_response(prefs, results)

        st.markdown(final_response)
        mode_label = "Salary Lookup" if prefs.mode == "salary_lookup" else "Skill Search"
        st.caption(f"⚡ Selesai dalam **{elapsed * 1000:.0f}ms** | Mode: {mode_label}")

    # Simpan ke history
    st.session_state.messages.append({"role": "assistant", "content": final_response})
    st.rerun()