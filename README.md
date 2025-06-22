# K-Drama Recommender System (2015–2023)

Sistem rekomendasi ini dibangun menggunakan data Korean Drama dari tahun 2015 hingga 2023, bertujuan untuk memberikan rekomendasi drama Korea terbaik bagi pengguna berdasarkan dua pendekatan:

* **Content-Based Filtering** (berdasarkan kemiripan sinopsis, sutradara, dan penulis naskah)
* **Collaborative Filtering** (berdasarkan preferensi dan penilaian pengguna lain)

Proyek ini cocok untuk studio produksi, penulis naskah, atau stasiun TV yang ingin memahami tren kreatif dan audiens global dengan lebih baik.

---

## Project Structure

```
├── kdrama_recommender.ipynb     # Notebook utama berisi analisis, preprocessing, modeling
├── recommend.py                 # Versi Python script untuk eksekusi di luar notebook
├── Kdrama Rekomendasi Laporan.md # Laporan akhir dalam format Markdown
├── data/
│   ├── korean_drama.csv
│   ├── review.csv
│   ├── recommendations.csv
│   └── wiki_actor.csv
└── README.md                    # Deskripsi proyek ini
```

---

## Dataset

* Sumber: [Kaggle - Korean Drama 2015–2023](https://www.kaggle.com/datasets/chanoncharuchinda/korean-drama-2015-23-actor-and-reviewmydramalist)
* Data terdiri dari:

  * Metadata drama (`korean_drama.csv`)
  * Ulasan pengguna (`review.csv`)
  * Data aktor (`wiki_actor.csv`)
  * Rekomendasi dari MyDramaList (`recommendations.csv`)

---

## Features

* Analisis eksploratif terhadap rating, penulis, sutradara, dan tren usia konten.
* Penerapan **TF-IDF + Cosine Similarity** untuk sistem rekomendasi berbasis konten.
* Penerapan **SVD (Singular Value Decomposition)** menggunakan Surprise library untuk collaborative filtering.
* Evaluasi menggunakan:

  * **RMSE** untuk collaborative filtering
  * **Precision\@10** untuk content-based filtering

---

## How to Run

1. Clone repository ini:

   ```bash
   git clone https://github.com/milkiyiki/K-drama-Recommender-System.git
   cd K-drama-Recommender-System
   ```

2. Jalankan di Jupyter Notebook:

   ```bash
   jupyter notebook kdrama_recommender.ipynb
   ```

3. Atau jalankan script Python (opsional):

   ```bash
   python recommend.py
   ```

> Pastikan kamu sudah menginstal dependensi yang diperlukan di `requirements.txt`.

---

## Model Highlights

* **Content-Based**

  * Precision\@10: `0.0110`
* **Collaborative-Based**

  * RMSE: `1.8432`

---

## Author

* **Risqie Nur Salsabila Ilman**
---

## License

This project is licensed under the MIT License.
