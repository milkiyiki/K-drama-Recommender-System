# Laporan Proyek Machine Learning - Risqie Nur Salsabila Ilman

## Project Overview

Popularitas K-Drama secara global telah meningkat pesat dalam dekade terakhir, terutama di kalangan penonton internasional. Stasiun televisi dan studio produksi kini menghadapi tantangan dalam memahami preferensi audiens global, guna merancang drama yang tidak hanya sukses di dalam negeri tetapi juga diterima luas secara internasional. Oleh karena itu, sistem rekomendasi menjadi alat penting untuk mendukung strategi produksi dan distribusi konten.

Dalam proyek ini, dibangun dua sistem rekomendasi berdasarkan data Korean Drama 2015–2023 yang dikumpulkan dari [MyDramaList](https://mydramalist.com/) dan tersedia di [Kaggle](https://www.kaggle.com/datasets/chanoncharuchinda/korean-drama-2015-23-actor-and-reviewmydramalist). Sistem rekomendasi ini bertujuan untuk:

- Memberikan rekomendasi drama berdasarkan konten (content-based filtering)
- Memberikan rekomendasi personal berdasarkan penilaian pengguna lain (collaborative filtering)

## Business Understanding

### Problem Statements

1. Bagaimana mengidentifikasi kesamaan antar K-Drama berdasarkan elemen konten seperti sinopsis, sutradara, dan penulis naskah?
2. Bagaimana memberikan rekomendasi personal untuk pengguna berdasarkan preferensi dan ulasan pengguna lain?
3. Siapa figur kreatif (sutradara, penulis, aktor) yang paling sering muncul dalam drama-drama yang direkomendasikan kepada pengguna?

### Goals

1. Menghasilkan sistem rekomendasi content-based yang dapat mengusulkan drama serupa berdasarkan metadata.
2. Membangun model collaborative filtering yang memberikan rekomendasi drama personal kepada pengguna berdasarkan skor keseluruhan.
3. Mengekstrak insight dari rekomendasi untuk memahami tren kreator dan preferensi usia konten yang paling disukai pengguna.

### Solution Statements

- Menggunakan pendekatan **TF-IDF dan cosine similarity** untuk content-based filtering karena metode ini mampu menangkap kemiripan semantik antar drama berdasarkan teks sinopsis, nama sutradara, dan penulis naskah. TF-IDF memberi bobot penting pada kata-kata yang unik dalam konteks drama tertentu, sementara cosine similarity mengukur kemiripan vektor antar drama secara efisien dalam ruang dimensi tinggi.
- Menggunakan **Surprise SVD (Singular Value Decomposition)** untuk collaborative filtering, karena model ini cocok dalam memprediksi skor pengguna terhadap item berdasarkan pola interaksi dan mampu mengatasi masalah sparsity pada matriks user-item.

## Data Understanding

### Pengantar Pembahasan

Dataset yang digunakan merupakan kompilasi dari 100 K-Drama terpopuler pada situs komunitas **MyDramaList** selama tahun 2015–2023. Dataset ini diunduh dari [Kaggle](https://www.kaggle.com/datasets/chanoncharuchinda/korean-drama-2015-23-actor-and-reviewmydramalist) dan terdiri dari empat file utama.

### Struktur dan URL Dataset

- `korean_drama.csv`: Data utama berisi 1.752 baris dan 17 kolom.
- `review.csv`: Berisi 10.625 baris dan 10 kolom ulasan pengguna.
- `recommendations.csv`: 1.753 baris dan 2 kolom.
- `wiki_actor.csv`: 8.659 baris dan 5 kolom informasi aktor.

### Kondisi Data

- Beberapa kolom pada `korean_drama.csv` memiliki missing value, misalnya:
  - `director`: hanya 1036 non-null
  - `screenwriter`: hanya 959 non-null
- Tidak ditemukan duplikat utama berdasarkan ID drama.
- `review.csv` memiliki 6 missing value pada kolom `review_text`, namun tidak memengaruhi modeling karena hanya `overall_score` yang digunakan.

### Uraian Fitur Dataset Utama

#### `korean_drama.csv`

- `kdrama_id`: ID unik drama
- `drama_name`: Nama drama
- `year`: Tahun rilis
- `director`: Sutradara
- `screenwriter`: Penulis naskah
- `country`: Negara asal
- `type`: Tipe (Drama/TV/Short)
- `tot_eps`: Total episode
- `duration`: Durasi per episode
- `start_dt` & `end_dt`: Tanggal tayang
- `aired_on`: Hari tayang
- `org_net`: Saluran penyiar
- `content_rt`: Rating usia
- `synopsis`: Ringkasan cerita
- `rank`: Ranking popularitas
- `pop`: Skor popularitas

#### `review.csv`

- `user_id`: ID pengguna
- `title`: Nama drama
- `story_score`, `acting_cast_score`, `music_score`, `rewatch_value_score`, `overall_score`: Skor masing-masing aspek
- `review_text`: Teks ulasan
- `ep_watched`: Episode yang ditonton
- `n_helpful`: Jumlah helpful votes

#### `wiki_actor.csv`

- `actor_id`: ID aktor
- `actor_name`: Nama aktor
- `drama_name`: Drama yang dibintangi
- `character_name`: Nama karakter
- `role`: Tipe peran (utama, pendukung)

**EDA Highlights:**

- Kim Eun Hee adalah screenwriter paling produktif di antara 100 drama terpopuler.
- Rating usia paling umum: Teen & 15+.
- Drama populer cenderung berdurasi < 20 episode.
- Skor "story" sangat berkorelasi dengan "overall score" (r = 0.91).
- Lee Yoo Jin adalah aktor dengan keterlibatan terbanyak (>18 drama).

## Data Preparation

### Persiapan Umum:

- Menghapus duplikat berdasarkan ID jika ada.
- Memeriksa dan menangani missing value di kolom penting seperti `director`, `screenwriter`, dan `synopsis`.
- Menyesuaikan tipe data agar konsisten (contoh: konversi list string ke list asli untuk sutradara dan penulis).

### Content-Based:

- Membersihkan kolom `director`, `screenwriter`, dan `synopsis` dari missing value dengan mengganti token seperti `no_director`, `no_screenwriter`, `no_synopsis`.
- Mengubah format kolom list string menjadi list Python menggunakan fungsi `eval()` dan pemrosesan string.
- Menyatukan ketiga kolom menjadi fitur gabungan `combined_features`:

```python
# Gabungkan beberapa kolom menjadi satu fitur teks gabungan
drama_df['combined_features'] = (
    drama_df['director'] + ' ' +
    drama_df['screenwriter'] + ' ' +
    drama_df['synopsis']
)
```

- Melakukan vektorisasi menggunakan `TfidfVectorizer` dengan 5000 fitur maksimal:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize menggunakan TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(drama_df['combined_features'])
```

### Collaborative-Based:

- Menggabungkan `reviews_df` dengan `drama_df` berdasarkan nama drama untuk mendapatkan ID drama:

```python
collab_df = reviews_df[['user_id', 'title', 'overall_score']].copy()
collab_df = collab_df.merge(drama_df[['drama_name', 'kdrama_id']], left_on='title', right_on='drama_name', how='left')
```

- Menghapus baris dengan ID kosong:

```python
collab_df = collab_df.dropna(subset=['kdrama_id'])
```

- Membentuk user-item matrix dan mengisi missing value dengan 0:

```python
user_item_matrix = collab_df.pivot_table(index='user_id', columns='kdrama_id', values='overall_score')
user_item_matrix = user_item_matrix.fillna(0)
```

- Menyiapkan data menggunakan `Surprise`:

```python
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split

reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(collab_df[['user_id', 'drama_name', 'overall_score']], reader)
```

- Membagi data menjadi train dan test set, lalu melatih model:

```python
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
model = SVD()
model.fit(trainset)
```

## Modeling

### Content-Based Filtering:

Menggunakan cosine similarity terhadap hasil TF-IDF untuk mengukur kemiripan antar drama. Model memberikan 10 rekomendasi teratas berdasarkan input judul drama.

```python
recommend_drama("Weak Hero Class 1")
```

Contoh output rekomendasi berdasarkan "Weak Hero Class 1":

| drama\_name         | year |
| ------------------- | ---- |
| Somebody            | 2022 |
| Bargain             | 2022 |
| Yonder              | 2022 |
| The Miracle         | 2016 |
| The Glory           | 2022 |
| Recipe for Farewell | 2022 |
| Revenge of Others   | 2022 |
| Celebrity           | 2023 |
| The King of Pigs    | 2022 |
| The Glory Part 2    | 2023 |

Evaluasi:

- Rata-rata Precision\@10: **0.0110** menunjukkan bahwa sekitar 1.1% dari rekomendasi top-10 adalah drama yang memang relevan menurut histori pengguna. Hal ini mengindikasikan bahwa pendekatan berbasis konten masih dapat ditingkatkan, terutama dalam representasi fitur atau pemrosesan teks.

### Collaborative Filtering:

Model collaborative filtering menggunakan `SVD` dari Surprise. Model ini dilatih pada `trainset` dan diuji pada `testset` untuk mengevaluasi performa prediksi menggunakan metrik RMSE.

```python
from surprise import SVD
model = SVD()
model.fit(trainset)
```

Model memberikan prediksi terhadap rating drama yang belum ditonton oleh pengguna dan hasilnya digunakan untuk memberi rekomendasi personal.

Contoh:

```python
recommend_for_user('user_100')
```

Model ini juga digunakan untuk menganalisis figur kreatif favorit user berdasarkan drama yang direkomendasikan padanya. Berdasarkan hasil prediksi model collaborative filtering terhadap user tertentu, rekomendasi teratas mencakup drama-drama seperti "Six Flying Dragons", "Good Manager", dan "My Mister". Drama-drama ini mendapatkan skor prediksi tinggi, menunjukkan bahwa model mampu menangkap kecenderungan preferensi pengguna.

Contoh output:

| title                | predicted_rating |
|----------------------|------------------|
| My Mister            | 8.869208          |
| Good Manager         | 8.837711          |
| Six Flying Dragons   | 8.823583          |
| Youth of May         | 8.814755          |
| Awaken               | 8.754926          |
| Children of Nobody   | 8.752513          |
| Mr. Queen            | 8.750298          |
| Crazy Love           | 8.728046          |
| The Fiery Priest     | 8.724150          |
| Our Blues            | 8.692143          |

## Evaluation

### Content-Based:

1. **Rekomendasi mirip secara tema & atmosfer**
   Sistem merekomendasikan drama yang **memiliki kemiripan kuat dalam sutradara, penulis naskah, dan sinopsis**. Sebagai contoh, saat dimasukkan *"Weak Hero Class 1"*, drama yang direkomendasikan seperti *"Somebody"*, *"Bargain"*, *"The Glory"* memiliki **tone gelap, drama remaja-dewasa, dan konflik psikologis yang intens**, mencerminkan pendekatan konten yang konsisten.

2. **Akurat dalam menangkap semantik konten, tapi tidak personal**
   Karena model ini **tidak mempertimbangkan data interaksi pengguna**, rekomendasi yang diberikan sangat relevan terhadap konten, **namun belum tentu sesuai preferensi tiap user**.

3. **Rendahnya Precision\@10 (0.0110)** mengindikasikan:

   * Sebagian besar rekomendasi **tidak termasuk dalam daftar drama yang benar-benar disukai user berdasarkan skor ≥8**.
   * Ini menunjukkan bahwa **kemiripan konten saja tidak cukup** untuk memprediksi ketertarikan pengguna, apalagi jika selera mereka tidak konsisten dengan isi sinopsis atau kredensial kreator.

4. **Potensi peningkatan**:

   * **Penggabungan fitur lain** seperti genre atau jaringan TV berpotensi menambah konteks dan meningkatkan kualitas rekomendasi. Tetapi ini tidak ada di dataset
   * **Hybrid filtering** dapat membantu menangani kelemahan personalisasi di content-based filtering ini.

### Collaborative-Based:

- Digunakan metrik **Root Mean Squared Error (RMSE)** pada data uji:

```python
RMSE: 1.8451
```

### Insight Tambahan:

- Model menunjukkan kecenderungan merekomendasikan drama dengan rating remaja dan dewasa muda, yang umumnya mencakup tema percintaan, konflik keluarga, dan drama sosial.
- Top 5 Sutradara yang muncul di hasil rekomendasi: Kim Kyu Tae, Lee Jung Mook, Kim Jung Hyun, Song Min Yeop, Lee Dae Kyung.
- Top 5 Penulis Naskah: Park Jae Bum, Noh Hee Kyung, Kim Bo Gyeom, Lee Kang, Shin Yoo Dam.
- RMSE menunjukkan performa prediksi rating; semakin rendah nilainya, semakin baik model memprediksi preferensi pengguna. RMSE dari model collaborative filtering (menggunakan SVD) adalah 1.8451, menunjukkan bahwa rata-rata selisih prediksi rating dengan rating sebenarnya cukup moderat. Meskipun belum mencapai tingkat akurasi yang sangat tinggi, hasil ini masih dapat digunakan untuk memberikan rekomendasi yang relevan secara umum.

**---Ini adalah bagian akhir laporan---**

