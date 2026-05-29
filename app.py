import streamlit as st
import joblib
import pandas as pd
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Movie Spoiler Detector", 
    page_icon="🎬",
    layout="centered"
)

# --- FUNGSI UNTUK LOAD MODEL ---
@st.cache_resource
def load_components():
    try:
        # Dapatkan jalur folder secara absolut (pasti akurat)
        # base_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        
        # Susun jalur untuk 2 file yang benar-benar ada di foldermu
        pipeline_path = os.path.join(base_dir, 'spoiler_detection_pipeline.pkl')
        model_name_path = os.path.join(base_dir, 'best_model_name.pkl')
        
        # Load filenya
        pipeline = joblib.load(pipeline_path)
        model_name = joblib.load(model_name_path)
        
        return pipeline, model_name
    
    except FileNotFoundError as e:
        st.error(f"Error: File tidak ditemukan di jalur ini -> {e.filename}")
        return None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None, None

# --- LOAD KOMPONEN ---
pipeline, model_name = load_components()

# --- SET STANDARD THRESHOLD ---
threshold = 0.50

# --- TAMPILAN UI STREAMLIT ---
st.title("🎬 Movie Spoiler Detector")
st.markdown("Cek apakah ulasan film di bawah ini mengandung *spoiler* atau aman untuk dibaca!")

# Tampilkan info model
if model_name is not None:
    st.caption(f"Powered by: **{model_name}** | Threshold: **{threshold:.2%}**")

# Kotak input teks
user_review = st.text_area(
    "Masukkan ulasan film di sini:", 
    height=150, 
    placeholder="Contoh: Filmnya seru banget, apalagi pas adegan plot twist di akhir ketika..."
)

# Tombol Prediksi
if st.button("Deteksi Spoiler", type="primary"):
    if not user_review.strip():
        st.warning("Ulasannya jangan dikosongin ya! Tulis sesuatu dulu.")
    elif pipeline is None:
        st.error("Model gagal dimuat. Pastikan file .pkl sudah benar lokasinya.")
    else:
        with st.spinner("Menganalisis teks dan metadata..."):
            try:
                # 1. Ekstraksi Meta Fitur (Menghitung jumlah kata)
                word_count = len(user_review.split())
                
                # 2. Bungkus input ke dalam DataFrame agar formatnya sesuai dengan Pipeline
                input_df = pd.DataFrame({
                    'review_text': [user_review],
                    'word_count': [word_count]
                })
                
                # 3. Prediksi menggunakan Pipeline
                probabilities = pipeline.predict_proba(input_df)[0]
                spoiler_prob = probabilities[1]
                
                # 4. Tentukan hasil berdasarkan standard threshold 50%
                is_spoiler = spoiler_prob >= threshold
                
                # 5. Tampilkan hasil ke UI
                st.markdown("---")
                st.subheader("Hasil Analisis:")
                
                if is_spoiler:
                    st.error("🚨 **SPOILER DETECTED!** 🚨")
                    st.write("Hati-hati! Ulasan ini kemungkinan besar mengandung bocoran cerita penting.")
                else:
                    st.success("✅ **SAFE TO READ!** ✅")
                    st.write("Aman! Ulasan ini sepertinya tidak membocorkan plot penting.")
                    
                # Progress bar
                st.write("### Tingkat Keyakinan Model")
                st.progress(float(spoiler_prob))
                st.caption(f"Skor Probabilitas Spoiler: **{spoiler_prob:.2%}** (Batas Aman: < {threshold:.2%}) | Panjang Teks: {word_count} kata")
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")