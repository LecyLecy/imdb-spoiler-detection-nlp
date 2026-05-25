# IMDb Spoiler Detection NLP

Project ini membangun sistem klasifikasi teks untuk mendeteksi apakah sebuah review IMDb mengandung spoiler atau tidak. Model menggunakan pendekatan Natural Language Processing (NLP) klasik berbasis TF-IDF, fitur tambahan panjang ulasan, dan beberapa algoritma machine learning ringan tanpa deep learning. Project ini juga menyediakan aplikasi Streamlit sederhana untuk mencoba prediksi spoiler secara interaktif.

Try the app:

```text
https://imdb-spoiler-detection-nlp.streamlit.app/
```

## Overview

Sistem mengklasifikasikan review ke dalam dua kategori:

- `Spoiler`
- `Non-Spoiler`

Tujuan utama project ini adalah membantu proses filter review secara otomatis agar pengguna dapat membaca ulasan film atau serial tanpa terkena bocoran alur cerita, plot twist, atau ending.

## Dataset

Dataset yang digunakan adalah IMDb Spoiler Dataset dari Kaggle:

https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset

File utama:

```text
IMDB_reviews.json
```

Kolom utama yang digunakan:

| Column | Description |
|---|---|
| `review_text` | Teks lengkap review IMDb |
| `is_spoiler` | Label target, 1 untuk spoiler dan 0 untuk non-spoiler |

Project ini menggunakan seluruh dataset tanpa balancing manual agar distribusi data tetap merepresentasikan kondisi asli.

## Project Workflow

Alur utama notebook:

1. Load dataset dari Kaggle atau cache lokal
2. Exploratory Data Analysis (EDA)
3. Preprocessing dan feature engineering
4. Data splitting dengan rasio 80% train dan 20% test
5. Feature extraction menggunakan TF-IDF unigram dan bigram
6. Training beberapa model machine learning
7. Evaluasi model menggunakan Accuracy, Precision, Recall, F1-Score, Confusion Matrix, dan Classification Report
8. Pemilihan best model berdasarkan F1-Score
9. Pengujian 20 random sample review
10. Penyimpanan best model pipeline

## Exploratory Data Analysis

EDA dilakukan untuk memahami karakteristik dataset sebelum training model:

- Distribusi label spoiler dan non-spoiler
- Proporsi kelas untuk melihat imbalance data
- Distribusi panjang review berdasarkan `word_count`
- Top words pada review spoiler dan non-spoiler

Visualisasi hasil evaluasi model disimpan di:

```text
figures/best_model_visualization.png
```

## Preprocessing & Feature Engineering

Preprocessing mengikuti pendekatan notebook `NLPB.ipynb`, yaitu mempertahankan teks review asli dan melakukan transformasi di dalam pipeline.

Fitur yang digunakan:

- `review_text`: teks review mentah
- `word_count`: jumlah kata dalam review sebagai meta-feature

Feature extraction:

- `TfidfVectorizer`
- `ngram_range=(1, 2)` untuk unigram dan bigram
- `max_features=15000`
- custom English stop words, termasuk stop words umum dan beberapa kata domain film seperti `movie`, `film`, `character`, dan `story`
- `StandardScaler` untuk menstandarkan fitur numerik `word_count`

Pendekatan ini menjaga konteks kalimat tetap tersedia, sekaligus mengurangi noise dari kata yang terlalu umum.

## Models

Notebook membandingkan beberapa model machine learning:

- SGD Classifier + Meta Feature
- Linear SVC + Meta Feature
- Passive Aggressive Classifier + Meta Feature
- Multinomial Naive Bayes
- Complement Naive Bayes

Logistic Regression tidak digunakan pada versi terbaru. Model terbaik dipilih berdasarkan nilai F1-Score tertinggi pada test set.

## Evaluation Metrics

Metrik evaluasi:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

F1-Score digunakan sebagai metrik utama karena deteksi spoiler membutuhkan keseimbangan antara kemampuan menemukan review spoiler dan menghindari terlalu banyak false positive.

## Saved Model

Best model disimpan sebagai pipeline utuh sehingga preprocessing, feature extraction, dan classifier dapat digunakan kembali dalam satu objek.

Output model:

```text
models/spoiler_detection_pipeline.pkl
models/best_model_name.pkl
models/model_comparison_results.pkl
```

File `app.py` memuat `spoiler_detection_pipeline.pkl` dan `best_model_name.pkl` dari folder `models/`. Input pengguna akan diubah menjadi DataFrame dengan kolom `review_text` dan `word_count`, lalu diprediksi menggunakan pipeline yang sama seperti hasil training notebook.

## Streamlit App

Project ini memiliki aplikasi web interaktif berbasis Streamlit:

```text
app.py
```

Aplikasi melakukan langkah berikut:

- Memuat best model pipeline dari folder `models/`
- Menghitung `word_count` dari review yang dimasukkan pengguna
- Membuat input dengan format kolom `review_text` dan `word_count`
- Mengambil probabilitas spoiler menggunakan `predict_proba`
- Menentukan hasil akhir dengan threshold manual `0.40`
- Menampilkan label `SPOILER DETECTED` atau `SAFE TO READ`

Jalankan aplikasi dengan:

```bash
streamlit run app.py
```

Pastikan file model berikut sudah tersedia sebelum menjalankan aplikasi:

```text
models/spoiler_detection_pipeline.pkl
models/best_model_name.pkl
```

Jika aplikasi gagal memuat model karena perbedaan versi dependency, jalankan notebook ulang pada environment aktif untuk menghasilkan ulang file `.pkl`.

## Deployment

Deployment yang direkomendasikan adalah Streamlit Community Cloud karena project ini sudah memiliki `app.py`, `requirements.txt`, dan file model di folder `models/`.

Langkah deploy:

1. Pastikan file penting sudah ada di repository:

```text
app.py
requirements.txt
models/spoiler_detection_pipeline.pkl
models/best_model_name.pkl
```

2. Pastikan semua perubahan sudah dipush ke GitHub:

```bash
git status
git add README.md requirements.txt app.py models/
git commit -m "Prepare Streamlit deployment"
git push origin main
```

3. Buka Streamlit Community Cloud:

```text
https://share.streamlit.io
```

4. Login menggunakan akun GitHub yang memiliki akses ke repository ini.

5. Klik `Create app`, lalu pilih opsi untuk deploy app dari repository yang sudah ada.

6. Isi konfigurasi deployment:

```text
Repository : LecyLecy/imdb-spoiler-detection-nlp
Branch     : main
Main file  : app.py
App URL    : bebas, sesuai nama yang tersedia
```

7. Jika diminta memilih Python version, gunakan Python 3.11 agar sesuai dengan environment development project.

8. Klik deploy dan tunggu proses install dependencies dari `requirements.txt` selesai.

9. Setelah deploy berhasil, salin link aplikasi dari Streamlit dan letakkan pada placeholder `Try the app` di bagian atas README.

10. Jika deploy gagal, cek bagian logs di Streamlit Cloud. Error yang paling mungkin terjadi adalah dependency mismatch atau file model tidak ditemukan.

## Repository Structure

```text
imdb-spoiler-detection-nlp/
|-- data/                         # Local dataset folder, ignored by Git
|-- figures/                      # Saved visualizations
|-- models/                       # Saved trained model pipeline
|-- imdb_venv/                    # Local virtual environment, ignored by Git
|-- .env                          # Kaggle credentials, ignored by Git
|-- .env.example                  # Environment variable template
|-- .gitignore
|-- app.py                        # Streamlit app for spoiler prediction
|-- imdb-spoiler-detection-old.ipynb
|-- README.md
|-- requirements.txt
```

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/LecyLecy/imdb-spoiler-detection-nlp.git
cd imdb-spoiler-detection-nlp
```

### 2. Create Python Virtual Environment

```bash
py -3.11 -m venv imdb_venv
imdb_venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Register Jupyter Kernel

```bash
python -m ipykernel install --user --name imdb_spoiler_nlp --display-name "Python (imdb_spoiler_nlp)"
```

Setelah itu, buka notebook dan pilih kernel:

```text
Python (imdb_spoiler_nlp)
```

## Kaggle API Setup

Buat file `.env` di root project:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

Kaggle API key bisa dibuat dari:

```text
Kaggle Account Settings -> API -> Create New Token
```

Pastikan `.env` tidak di-commit ke GitHub.

## Running the Notebook

Buka notebook:

```text
imdb-spoiler-detection-old.ipynb
```

Lalu jalankan cell secara berurutan. Notebook akan:

- Mengecek dataset lokal
- Download dataset dari Kaggle jika belum tersedia
- Load seluruh dataset
- Melakukan EDA dan visualisasi
- Membuat fitur TF-IDF dan `word_count`
- Melatih beberapa model
- Memilih model terbaik berdasarkan F1-Score
- Menguji 20 random sample review
- Menyimpan best model pipeline

Setelah model tersimpan, aplikasi Streamlit dapat dijalankan dengan:

```bash
streamlit run app.py
```

## Example Prediction

```text
sample_review = The movie ends with the main character dying in the final scene.
Model yakin 82.89% bahwa sample_review adalah spoiler
Model yakin 17.11% bahwa sample_review adalah bukan spoiler
Final prediction = spoiler
```

## Git Ignore Recommendation

File dan folder berikut tidak perlu dipush ke GitHub:

```gitignore
.env
data/
imdb_venv/
__pycache__/
.ipynb_checkpoints/
*.zip
```

## Notes

Dataset tidak disertakan dalam repository karena ukuran file besar dan dapat diunduh ulang dari Kaggle. Cache lokal juga dapat dibuat ulang dengan menjalankan notebook.

## License

Project ini menggunakan IMDb Spoiler Dataset dari Kaggle. Silakan merujuk ke halaman dataset asli untuk detail lisensi dan ketentuan penggunaan.
