# Laporan Proyek Machine Learning Terapan - Muhammad Alif

## Project Overview

Proyek ini bertujuan untuk membangun sistem rekomendasi film yang dapat memberikan saran film yang dipersonalisasi kepada pengguna. Di era digital dengan melimpahnya pilihan konten, sistem rekomendasi menjadi krusial untuk membantu pengguna menemukan film yang sesuai dengan selera mereka, sekaligus meningkatkan engagement pada platform penyedia film. Dengan banyaknya platform streaming dan katalog film yang terus berkembang, pengguna seringkali dihadapkan pada "paradox of choice", yaitu kesulitan dalam memilih tontonan di antara banyaknya opsi yang tersedia. Sistem rekomendasi yang efektif dapat mengatasi masalah ini dengan menyaring dan menyajikan pilihan yang paling relevan bagi masing-masing individu.

Proyek ini akan mengeksplorasi dua pendekatan utama dalam sistem rekomendasi:
1.  **Content-Based Filtering**: Merekomendasikan item berdasarkan kemiripan atribut atau konten item itu sendiri dengan item yang pernah disukai pengguna.
2.  **Collaborative Filtering**: Merekomendasikan item berdasarkan preferensi dari pengguna lain yang memiliki selera serupa.

Tujuan akhir adalah untuk mengembangkan model yang mampu memberikan rekomendasi yang akurat dan bermanfaat, serta memahami kelebihan dan kekurangan dari masing-masing pendekatan yang diterapkan pada dataset MovieLens.

**Referensi:**
-   Ricci, F., Rokach, L., & Shapira, B. (Eds.). (2011). *Recommender systems handbook*. Springer US.
-   Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, *42*(8), 30-37.
-   Aggarwal, C. C. (2016). *Recommender systems: the textbook*. Springer International Publishing.

## Business Understanding

### Problem Statements
1.  Bagaimana cara memberikan rekomendasi film yang relevan dan dipersonalisasi kepada pengguna dari katalog film yang besar untuk membantu mereka menemukan tontonan yang sesuai dengan preferensi mereka?
2.  Bagaimana cara memanfaatkan data rating dan tag yang diberikan oleh komunitas pengguna untuk meningkatkan kualitas dan relevansi rekomendasi film?
3.  Pendekatan sistem rekomendasi manakah (Content-Based Filtering atau Collaborative Filtering) yang lebih cocok atau memberikan hasil yang berbeda dan wawasan yang berguna untuk dataset MovieLens yang digunakan?

### Goals
1.  Mengembangkan model Content-Based Filtering yang mampu merekomendasikan film berdasarkan kemiripan konten, khususnya genre film dan tag yang diberikan pengguna.
2.  Mengembangkan model Collaborative Filtering, menggunakan algoritma Singular Value Decomposition (SVD), yang merekomendasikan film berdasarkan pola rating dari pengguna lain dengan selera serupa.
3.  Mengevaluasi kinerja kedua model menggunakan metrik yang sesuai (seperti RMSE untuk prediksi rating, serta Precision@k dan Recall@k untuk kualitas Top-N rekomendasi) dan menyajikan contoh Top-N rekomendasi dari masing-masing model.
4.  Menganalisis kelebihan dan kekurangan dari kedua pendekatan yang diimplementasikan dalam konteks dataset yang digunakan.

### Solution Approach
Untuk mencapai tujuan-tujuan di atas, dua pendekatan solusi akan diimplementasikan dan dievaluasi:

1.  **Content-Based Filtering:**
    *   **Algoritma/Teknik:**
        *   Ekstraksi fitur teks dari kombinasi `genres` dan `tags` film.
        *   Pembobotan fitur menggunakan TF-IDF (Term Frequency-Inverse Document Frequency).
        *   Pengukuran kemiripan antar film menggunakan Cosine Similarity.
    *   **Cara Kerja:** Sistem akan mencari film yang memiliki profil konten (genre dan tag) paling mirip dengan film yang dijadikan referensi (misalnya, film yang pernah disukai pengguna).

2.  **Collaborative Filtering:**
    *   **Algoritma/Teknik:**
        *   Singular Value Decomposition (SVD) dari library `Surprise`.
    *   **Cara Kerja:** SVD akan menguraikan matriks interaksi user-item (rating) menjadi faktor laten pengguna dan item. Berdasarkan faktor laten ini, sistem akan memprediksi rating yang mungkin diberikan pengguna pada film yang belum ia tonton, dan merekomendasikan film dengan prediksi rating tertinggi.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah bagian dari dataset **MovieLens (ml-latest-small)** yang disediakan oleh GroupLens Research. Dataset ini cukup populer untuk penelitian dan pengembangan sistem rekomendasi.
Sumber data: [MovieLens - ml-latest-small](https://grouplens.org/datasets/movielens/latest/)

Dataset ini terdiri dari beberapa file CSV, namun yang utama digunakan dalam proyek ini adalah:
-   `movies.csv`: Berisi informasi dasar mengenai film.
-   `ratings.csv`: Berisi data rating yang diberikan oleh pengguna untuk film.
-   `tags.csv`: Berisi tag atau kata kunci yang diberikan oleh pengguna untuk film.

### Informasi Dasar Dataset
-   **movies.csv**:
    -   Jumlah Film: 9742
    -   Kolom: `movieId`, `title`, `genres`
-   **ratings.csv**:
    -   Jumlah Rating: 100836
    -   Jumlah Pengguna Unik: 610
    -   Jumlah Film Unik yang Dirating: 9724
    -   Kolom: `userId`, `movieId`, `rating`, `timestamp`
-   **tags.csv**:
    -   Jumlah Tag: 3683
    -   Kolom: `userId`, `movieId`, `tag`, `timestamp`

### Variabel pada Dataset
1.  **movies.csv:**
    *   `movieId`: ID unik numerik untuk setiap film. (Tipe: int64)
    *   `title`: Judul film beserta tahun rilis dalam tanda kurung. (Tipe: object/string)
    *   `genres`: Genre film, beberapa genre dapat dipisahkan oleh karakter `|`. (Tipe: object/string)

2.  **ratings.csv:**
    *   `userId`: ID unik numerik untuk setiap pengguna. (Tipe: int64)
    *   `movieId`: ID film yang dirating, merujuk ke `movieId` di `movies.csv`. (Tipe: int64)
    *   `rating`: Rating yang diberikan pengguna untuk film, dengan skala 0.5 hingga 5.0 dalam interval 0.5. (Tipe: float64)
    *   `timestamp`: Waktu saat rating diberikan, dalam format Unix timestamp. (Tipe: int64)

3.  **tags.csv:**
    *   `userId`: ID unik pengguna yang memberikan tag. (Tipe: int64)
    *   `movieId`: ID film yang diberi tag, merujuk ke `movieId` di `movies.csv`. (Tipe: int64)
    *   `tag`: Kata kunci atau deskripsi singkat yang diberikan pengguna untuk film. (Tipe: object/string)
    *   `timestamp`: Waktu saat tag diberikan, dalam format Unix timestamp. (Tipe: int64)

Tidak ada nilai null yang signifikan pada kolom-kolom utama yang digunakan dalam analisis ini.

### Exploratory Data Analysis (EDA)
Beberapa temuan dari EDA:

1.  **Distribusi Rating Film**:
    *   Rating yang paling sering diberikan adalah 4.0, diikuti oleh 3.0 dan 5.0.
    *   Rating 3.5 dan 4.5 juga cukup umum.
    *   Rating rendah (0.5 - 2.5) lebih jarang diberikan, menunjukkan kecenderungan pengguna untuk lebih sering merating film yang mereka sukai atau setidaknya tidak terlalu buruk.
    *   *Insight*: Model rekomendasi perlu mempertimbangkan bahwa pengguna mungkin lebih vokal tentang film yang mereka nikmati.

    ![Distribusi Rating](https://github.com/miezlearning/Coding-Camp-2025/blob/master/Machine%20Learning%20Terapan/Proyek%20Besar/Assets/dis_movrating.png?raw=true)
2.  **Jumlah Rating per Pengguna**:
    *   Distribusi jumlah rating per pengguna sangat skewed ke kanan.
    *   Sebagian besar pengguna (lebih dari 50%) memberikan kurang dari ~70 rating.
    *   Beberapa pengguna sangat aktif, dengan satu pengguna memberikan hingga ~2700 rating.
    *   *Insight*: Ada variasi besar dalam aktivitas pengguna. Model Collaborative Filtering mungkin lebih akurat untuk pengguna dengan banyak rating, dan perlu strategi khusus (seperti Content-Based) untuk pengguna dengan sedikit histori (cold-start).

    ![Distribusi Rating per User](https://github.com/miezlearning/Coding-Camp-2025/blob/master/Machine%20Learning%20Terapan/Proyek%20Besar/Assets/dis_rat_user.png?raw=true)

3.  **Jumlah Rating per Film**:
    *   Distribusi ini juga skewed. Banyak film yang hanya mendapatkan sedikit rating (lebih dari 50% film memiliki kurang dari ~3 rating).
    *   Beberapa film sangat populer dan mendapatkan ratusan rating.
    *   *Insight*: Film dengan sedikit rating mungkin sulit direkomendasikan oleh Collaborative Filtering karena kurangnya data. Rekomendasi mungkin cenderung bias ke film populer.

    ![Distribusi Rating per Film](https://github.com/miezlearning/Coding-Camp-2025/blob/master/Machine%20Learning%20Terapan/Proyek%20Besar/Assets/dis_rating_mov.png?raw=true)

4.  **Genre Film Terpopuler**:
    *   Genre Drama dan Comedy adalah yang paling dominan dalam dataset, diikuti oleh Thriller dan Action.
    *   Genre seperti Documentary, IMAX, atau Film-Noir lebih jarang muncul.
    *   *Insight*: Model Content-Based akan memiliki banyak data untuk genre populer, tetapi perlu diperhatikan agar genre minoritas juga mendapat kesempatan direkomendasikan untuk menjaga keberagaman.

    ![Genre Terpopuler](https://github.com/miezlearning/Coding-Camp-2025/blob/master/Machine%20Learning%20Terapan/Proyek%20Besar/Assets/download%20(1).png?raw=true) 

## Data Preparation
Tahap Data Preparation bertujuan untuk membersihkan dan mentransformasi data agar siap digunakan oleh model sistem rekomendasi. Proses ini krusial karena kualitas data input sangat mempengaruhi kualitas output model.

Langkah-langkah Data Preparation yang dilakukan:

1.  **Agregasi Tag per Film**:
    *   **Proses**: Data `tags.csv` berisi multiple tag per film dari pengguna yang berbeda. Tag-tag ini diagregasi berdasarkan `movieId`, di mana semua tag untuk satu film digabungkan menjadi satu string tunggal, dipisahkan oleh spasi.
    *   **Alasan**: Untuk membuat representasi teks yang komprehensif dari tag yang diasosiasikan dengan setiap film, yang akan digunakan sebagai fitur konten.

2.  **Penggabungan Data Film dan Tag**:
    *   **Proses**: DataFrame `movies_df` (berisi `movieId`, `title`, `genres`) digabungkan dengan hasil agregasi tag menggunakan `movieId` sebagai kunci. Penggabungan dilakukan dengan `left join` untuk memastikan semua film dari `movies_df` tetap ada.
    *   **Alasan**: Untuk mengkonsolidasikan semua informasi konten (genre dan tag) ke dalam satu DataFrame.

3.  **Penanganan Missing Values pada Tag**:
    *   **Proses**: Setelah penggabungan, film yang tidak memiliki tag akan memiliki nilai `NaN` pada kolom tag. Nilai `NaN` ini diisi dengan string kosong (`''`).
    *   **Alasan**: Untuk menghindari error saat pemrosesan teks dan memastikan semua film dapat di-vektorisasi, meskipun beberapa mungkin tidak memiliki kontribusi dari tag.

4.  **Pembersihan Fitur Teks (Genres dan Tags)**:
    *   **Proses**:
        *   Untuk kolom `genres`, karakter `|` (yang memisahkan genre) diganti dengan spasi.
        *   Semua teks pada kolom `genres` dan `tag` dikonversi menjadi huruf kecil (lowercase).
    *   **Alasan**: Standardisasi format teks penting agar kata yang sama (misalnya, "Action" dan "action") dianggap identik oleh algoritma pemrosesan teks (seperti TF-IDF), sehingga meningkatkan konsistensi dan akurasi perhitungan kemiripan.

5.  **Pembuatan Fitur Gabungan (`soup`)**:
    *   **Proses**: Kolom `genres` yang sudah dibersihkan dan kolom `tag` yang sudah dibersihkan digabungkan menjadi satu kolom baru bernama `soup`. Spasi berlebih yang mungkin muncul dari penggabungan juga dibersihkan.
    *   **Alasan**: Fitur `soup` ini menyediakan representasi teks tunggal yang kaya akan informasi konten (genre dan tag) untuk setiap film. Ini akan menjadi input utama untuk model Content-Based Filtering berbasis TF-IDF.

6.  **Persiapan Data untuk Collaborative Filtering (Surprise)**:
    *   **Proses**: DataFrame `ratings_df` (kolom `userId`, `movieId`, `rating`) dipilih. Library `Surprise` memerlukan data dalam format spesifik, yang diatur menggunakan `Reader` (untuk menentukan skala rating) dan `Dataset.load_from_df()`.
    *   **Alasan**: Untuk menyesuaikan data rating dengan format yang dibutuhkan oleh algoritma SVD dalam library `Surprise`.

## Modeling

Dua pendekatan model sistem rekomendasi diimplementasikan:

### 1. Content-Based Filtering
Model ini merekomendasikan film berdasarkan kemiripan kontennya (genre dan tag).

*   **Proses**:
    1.  Fitur teks `soup` dari setiap film diubah menjadi representasi vektor numerik menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)**. TF-IDF memberi bobot pada kata-kata berdasarkan frekuensinya dalam satu dokumen (film) dan invers frekuensinya di seluruh koleksi dokumen (semua film). Ini membantu mengidentifikasi kata kunci yang paling deskriptif untuk setiap film.
    2.  Matriks TF-IDF yang dihasilkan memiliki dimensi (jumlah film x jumlah term unik). Dari output, matriks ini berukuran (9742, 727).
    3.  **Cosine Similarity** dihitung antara semua pasangan vektor TF-IDF film untuk mendapatkan matriks kemiripan (film x film). Matriks ini berukuran (9742, 9742).
    4.  Sebuah fungsi dikembangkan untuk mengambil judul film sebagai input, mencari skor kemiripan film tersebut dengan semua film lain, dan mengembalikan N film teratas dengan skor kemiripan tertinggi.

*   **Contoh Rekomendasi Content-Based**:
    *   Untuk film 'Toy Story (1995)':
        1.  Bug's Life, A (1998) (Skor: 0.8622)
        2.  Toy Story 2 (1999) (Skor: 0.6440)
        3.  Up (2009) (Skor: 0.3879)
    *   Untuk film 'Jumanji (1995)':
        1.  Tomb Raider (2018) (Skor: 0.5719)
        2.  Night at the Museum (2006) (Skor: 0.3786)
        3.  Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001) (Skor: 0.3449)

*   **Kelebihan**:
    *   Tidak memerlukan data dari pengguna lain (mengatasi user cold-start).
    *   Mampu merekomendasikan item baru jika item tersebut memiliki deskripsi konten yang cukup.
    *   Rekomendasi dapat dijelaskan berdasarkan fitur konten yang mirip.
*   **Kekurangan**:
    *   Terbatas pada fitur yang digunakan; jika fitur tidak lengkap atau kurang deskriptif, kualitas rekomendasi menurun.
    *   Cenderung menghasilkan rekomendasi yang mirip secara konten (kurang serendipity).
    *   Dapat menciptakan "filter bubble" di mana pengguna hanya terekspos pada jenis konten yang sudah ia sukai.

### 2. Collaborative Filtering (SVD)
Model ini merekomendasikan film berdasarkan pola rating dari pengguna lain yang memiliki selera serupa.

*   **Proses**:
    1.  Menggunakan algoritma **Singular Value Decomposition (SVD)** dari library `Surprise`.
    2.  Data rating (`userId`, `movieId`, `rating`) dibagi menjadi data latih (80%) dan data uji (20%).
    3.  Model SVD dilatih pada data latih. Parameter yang digunakan adalah `n_factors=100`, `n_epochs=30`, `lr_all=0.005`, dan `reg_all=0.1`.
    4.  Model yang telah dilatih digunakan untuk memprediksi rating film yang belum ditonton oleh pengguna target.
    5.  Sebuah fungsi dikembangkan untuk mengambil `userId` sebagai input dan mengembalikan N film teratas dengan prediksi rating tertinggi yang belum pernah ditonton oleh pengguna tersebut.

*   **Contoh Rekomendasi Collaborative Filtering (SVD)**:
    *   Untuk User ID 1 (Top 5):
        1.  Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964) (Prediksi Rating: 5.0)
        2.  Streetcar Named Desire, A (1951) (Prediksi Rating: 5.0)
        3.  Paths of Glory (1957) (Prediksi Rating: 5.0)
        4.  Lawrence of Arabia (1962) (Prediksi Rating: 5.0)
        5.  Three Billboards Outside Ebbing, Missouri (2017) (Prediksi Rating: 5.0)
    *   Untuk User ID 68 (Top 5):
        1.  Paths of Glory (1957) (Prediksi Rating: 4.0401)
        2.  Cinema Paradiso (Nuovo cinema Paradiso) (1989) (Prediksi Rating: 4.0136)
        3.  Streetcar Named Desire, A (1951) (Prediksi Rating: 3.9943)

*   **Kelebihan**:
    *   Mampu menemukan hubungan dan preferensi implisit yang tidak terlihat dari fitur konten saja.
    *   Potensi untuk menghasilkan rekomendasi yang lebih beragam dan mengejutkan (serendipity).
    *   Kualitas rekomendasi dapat meningkat seiring bertambahnya data interaksi pengguna.
*   **Kekurangan**:
    *   **Cold-start problem**: Sulit memberikan rekomendasi untuk pengguna baru (tanpa histori rating) atau item baru (belum ada yang merating).
    *   **Data sparsity**: Kinerja dapat menurun jika matriks user-item sangat sparse.
    *   Cenderung memiliki bias terhadap item-item populer yang sudah banyak dirating.

## Evaluation

Evaluasi model sistem rekomendasi penting untuk mengukur kinerjanya dan memahami seberapa baik model tersebut dalam memenuhi tujuannya.

### Metrik Evaluasi yang Digunakan
1.  **RMSE (Root Mean Squared Error)**:
    *   **Formula**: `sqrt(mean((actual_rating - predicted_rating)^2))`
    *   **Cara Kerja**: RMSE mengukur rata-rata besarnya selisih kuadrat antara rating aktual yang diberikan pengguna dan rating yang diprediksi oleh model. Karena adanya pengkuadratan, RMSE memberikan bobot yang lebih besar pada kesalahan prediksi yang besar. Nilai RMSE yang lebih kecil menunjukkan bahwa model lebih akurat dalam memprediksi rating.
    *   **Konteks**: Metrik ini sangat relevan untuk model Collaborative Filtering berbasis prediksi rating seperti SVD, di mana tujuan utamanya adalah memprediksi seberapa besar pengguna akan menyukai suatu item.

2.  **Precision@k dan Recall@k**:
    *   **Precision@k Formula**: `(Jumlah item relevan di top-k rekomendasi) / k`
        *   *Cara Kerja Precision@k*: Mengukur proporsi item yang direkomendasikan dalam k item teratas yang ternyata memang relevan bagi pengguna. Misalnya, jika k=10 dan ada 3 item relevan di antara 10 rekomendasi, maka Precision@10 = 0.3. Ini penting karena pengguna biasanya hanya memperhatikan beberapa rekomendasi awal.
    *   **Recall@k Formula**: `(Jumlah item relevan di top-k rekomendasi) / (Total item relevan untuk pengguna di test set)`
        *   *Cara Kerja Recall@k*: Mengukur proporsi item yang relevan bagi pengguna (di seluruh test set) yang berhasil ditangkap dan direkomendasikan oleh sistem dalam k item teratas. Misalnya, jika ada total 5 item relevan untuk pengguna dan 2 di antaranya masuk dalam 10 rekomendasi teratas, maka Recall@10 = 0.4.
    *   **Konteks**: Kedua metrik ini digunakan untuk mengevaluasi kualitas ranking dari Top-N rekomendasi. Item dianggap "relevan" jika rating aktualnya pada data uji (test set) berada di atas suatu threshold tertentu (dalam proyek ini, threshold = 4.0).
    *   **Pemilihan Metrik**: Precision@k dan Recall@k dipilih karena sangat sesuai dengan tujuan praktis sistem rekomendasi, yaitu menyajikan daftar pendek film yang paling mungkin disukai pengguna. Kualitas rekomendasi di posisi teratas sangat krusial.

### Hasil Proyek Berdasarkan Metrik Evaluasi

**Evaluasi Content-Based Filtering**:
Evaluasi untuk model Content-Based Filtering yang dikembangkan dalam proyek ini bersifat kualitatif, dengan menganalisis relevansi konten dari film yang direkomendasikan terhadap film input.
-   Misalnya, untuk film **'Toy Story (1995)'** (genre Adventure|Animation|Children|Comedy|Fantasy), rekomendasi teratas yang dihasilkan adalah:
    1.  'Bug's Life, A (1998)' (genre Adventure|Animation|Children|Comedy) dengan skor kemiripan **0.8622**.
    2.  'Toy Story 2 (1999)' (genre Adventure|Animation|Children|Comedy|Fantasy) dengan skor kemiripan **0.6440**.
    3.  'Up (2009)' (genre Adventure|Animation|Children|Drama) dengan skor kemiripan **0.3879**.
    Rekomendasi ini sangat masuk akal karena film-film tersebut memiliki genre dan kemungkinan tag yang serupa, menunjukkan bahwa model berhasil menangkap kemiripan berdasarkan konten.
-   Untuk film **'Jumanji (1995)'** (genre Adventure|Children|Fantasy), rekomendasi seperti 'Tomb Raider (2018)' (skor 0.5719) dan 'Night at the Museum (2006)' (skor 0.3786) juga menunjukkan kemiripan genre Adventure dan Fantasy.

**Hasil Metrik Evaluasi untuk Collaborative Filtering (SVD)**:
-   Model SVD menghasilkan nilai **RMSE sebesar 0.8718** pada test set. Ini menunjukkan bahwa rata-rata kesalahan prediksi rating oleh model adalah sekitar 0.87 poin pada skala rating 0.5 hingga 5.0. Nilai ini cukup baik, mengindikasikan model memiliki kemampuan prediktif yang layak.
-   Dengan menggunakan threshold relevansi rating ≥ 4.0 dan k=10 untuk Top-N rekomendasi:
    -   **Average Precision@10 adalah 0.6489**. Ini berarti, secara rata-rata, sekitar 64.89% dari 10 film teratas yang direkomendasikan kepada pengguna memang dianggap relevan (disukai) oleh pengguna tersebut berdasarkan data rating aktual di test set.
    -   **Average Recall@10 adalah 0.6668**. Ini berarti, secara rata-rata, model berhasil merekomendasikan sekitar 66.68% dari total film yang relevan bagi pengguna (yang ada di test set) dalam 10 rekomendasi teratasnya.
    Kedua nilai ini menunjukkan bahwa model SVD cukup efektif dalam menyajikan rekomendasi yang akurat di posisi-posisi teratas.

## Kesimpulan

Proyek ini berhasil mengembangkan dan mengevaluasi dua jenis sistem rekomendasi film menggunakan dataset MovieLens: Content-Based Filtering dan Collaborative Filtering (dengan algoritma SVD).

**Content-Based Filtering**:
-   Model ini mampu memberikan rekomendasi berdasarkan kemiripan fitur konten film, yaitu genre dan tag yang diberikan pengguna.
-   Hasil kualitatif menunjukkan bahwa rekomendasi yang dihasilkan relevan secara tematik. Contohnya, untuk "Toy Story (1995)", film-film animasi seperti "Bug's Life, A (1998)" (skor kemiripan 0.8622) dan "Toy Story 2 (1999)" (skor 0.6440) berhasil direkomendasikan.
-   **Kelebihan**: Tidak memerlukan data histori rating pengguna lain, dapat menjelaskan mengapa suatu item direkomendasikan, dan efektif untuk item baru dengan deskripsi yang kaya.
-   **Kekurangan**: Cenderung merekomendasikan item yang sangat mirip (kurang serendipity), dan sangat bergantung pada kualitas serta kelengkapan fitur konten yang diekstraksi.

**Collaborative Filtering (SVD)**:
-   Model SVD dilatih menggunakan data rating pengguna dan menunjukkan kinerja yang cukup baik dalam memprediksi rating, dengan **RMSE sebesar 0.8718** pada test set.
-   Dalam hal kualitas Top-N rekomendasi, model ini mencapai **Average Precision@10 sebesar 0.6489** dan **Average Recall@10 sebesar 0.6668** (dengan threshold relevansi rating ≥ 4.0). Ini mengindikasikan bahwa model cukup baik dalam menyajikan film-film yang kemungkinan besar akan disukai pengguna di antara rekomendasi teratasnya.
-   Contohnya, untuk User ID 1, film-film klasik dan berating tinggi seperti "Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)" dan "Streetcar Named Desire, A (1951)" direkomendasikan dengan prediksi rating 5.0.
-   **Kelebihan**: Mampu menangkap preferensi implisit pengguna dan menghasilkan rekomendasi yang lebih beragam atau serendipitous, tidak bergantung pada fitur konten item.
-   **Kekurangan**: Mengalami masalah cold-start untuk pengguna atau item baru, performanya menurun jika data rating sangat sparse, dan bisa memiliki bias terhadap item populer.

**Secara Keseluruhan**:
Kedua pendekatan sistem rekomendasi ini menawarkan solusi yang berbeda dengan kekuatan dan kelemahan masing-masing. Content-Based Filtering cocok untuk memulai rekomendasi ketika data interaksi pengguna masih sedikit atau untuk menjelaskan rekomendasi. Collaborative Filtering, terutama model-based seperti SVD, unggul dalam mempersonalisasi rekomendasi berdasarkan perilaku kolektif pengguna ketika data rating yang cukup tersedia. Untuk sistem yang lebih robust dan mampu mengatasi kekurangan masing-masing pendekatan, pengembangan model hybrid yang menggabungkan kedua metode ini seringkali menjadi langkah selanjutnya yang menjanjikan hasil terbaik. Proyek ini telah memberikan dasar pemahaman yang baik tentang kedua teknik fundamental dalam sistem rekomendasi.
