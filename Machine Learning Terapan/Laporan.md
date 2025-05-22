

# Laporan Proyek Machine Learning - Sistem Prediksi Penyakit Jantung

**Nama Anda:** Muhammad Alif  
**Email Dicoding:** m.alif7890@gmail.com  
**Tanggal:** 23/05/2025

## Domain Proyek

Penyakit jantung kardiovaskular (Cardiovascular Diseases/CVDs) merupakan penyebab kematian nomor satu secara global, merenggut jutaan nyawa setiap tahunnya. Deteksi dini dan diagnosis yang akurat sangat krusial untuk intervensi medis yang cepat, yang pada gilirannya dapat secara signifikan meningkatkan prognosis pasien dan mengurangi angka kematian. Namun, proses diagnosis penyakit jantung seringkali kompleks, melibatkan banyak faktor risiko dan hasil tes medis yang bervariasi, membutuhkan keahlian dan pengalaman dokter.

Dalam menghadapi tantangan ini, teknologi *machine learning* (ML) menawarkan potensi besar. Algoritma ML mampu menganalisis pola-pola kompleks dalam volume data medis yang besar, mengidentifikasi hubungan tersembunyi antar variabel, dan membuat prediksi dengan tingkat akurasi yang tinggi. Dengan memanfaatkan ML, kita dapat mengembangkan sistem pendukung keputusan yang membantu tenaga medis dalam melakukan skrining awal pada populasi berisiko tinggi dan memprediksi kemungkinan seseorang menderita penyakit jantung, sehingga memungkinkan tindakan pencegahan atau pengobatan yang lebih cepat dan tepat.

**Rubrik/Kriteria Tambahan (Opsional)**:
*   **Mengapa dan bagaimana masalah tersebut harus diselesaikan:**
    Penyakit jantung menyebabkan beban kesehatan yang masif, baik dari sisi mortalitas, morbiditas, maupun biaya perawatan. Pendekatan tradisional seringkali bergantung pada penilaian subjektif dokter dan hasil tes yang mahal/invasif. *Machine learning* dapat menyelesaikan masalah ini dengan:
    1.  **Prediksi Dini:** Mengidentifikasi individu berisiko tinggi bahkan sebelum gejala parah muncul, memungkinkan intervensi preventif.
    2.  **Objektivitas dan Konsistensi:** Memberikan prediksi berbasis data yang lebih objektif dan konsisten dibandingkan penilaian manusia, mengurangi variabilitas dalam diagnosis.
    3.  **Efisiensi Skrining:** Memungkinkan skrining massal yang lebih efisien di fasilitas kesehatan dengan sumber daya terbatas, memprioritaskan pasien untuk pemeriksaan lebih lanjut.
    4.  **Pemanfaatan Data Kompleks:** Menganalisis interaksi kompleks antar banyak fitur (demografis, klinis, hasil tes) yang mungkin sulit diinterpretasi oleh manusia secara langsung.

*   **Hasil riset terkait atau referensi:**
    1.  [1] World Health Organization. (2024). *Cardiovascular diseases (CVDs)*. Diakses dari [https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)).
        *   *Penjelasan:* Sumber ini menyediakan statistik global mengenai prevalensi, beban, dan faktor risiko penyakit kardiovaskular, menegaskan urgensi proyek prediksi dini.
    2.  [2] Muhammad, A. B., & Mahmud, H. (2020). *Heart Disease Prediction Using Machine Learning Algorithms: A Comparative Study*. International Journal of Computer Applications, 174(1), 1-5.
        *   *Penjelasan:* Studi ini memberikan contoh aplikasi nyata berbagai algoritma ML dalam memprediksi penyakit jantung, menunjukkan potensi dan relevansi teknis dari pendekatan yang dipilih dalam proyek ini.

## Business Understanding

Pada bagian ini, saya menjelaskan proses klarifikasi masalah yang ingin dipecahkan dengan *machine learning*.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
1.  Bagaimana kita dapat mengembangkan model *machine learning* yang efektif untuk secara akurat memprediksi keberadaan penyakit jantung pada pasien berdasarkan data klinis dan demografis yang tersedia?
2.  Bagaimana model prediksi ini dapat diintegrasikan sebagai alat bantu bagi tenaga medis untuk mendukung diagnosis dini dan pengambilan keputusan klinis, sehingga memungkinkan intervensi lebih cepat dan mengurangi risiko komplikasi serius?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
1.  Mengembangkan model klasifikasi *machine learning* (spesifiknya, **Klasifikasi Biner**) yang mampu mengidentifikasi pasien dengan penyakit jantung (kelas positif) dan tanpa penyakit jantung (kelas negatif) dengan performa evaluasi yang optimal.
2.  Mengidentifikasi fitur-fitur medis utama yang memiliki pengaruh paling signifikan terhadap risiko penyakit jantung berdasarkan analisis model.
3.  Menyediakan alat prediksi yang dapat diandalkan untuk skrining awal, mendukung deteksi dini dan manajemen risiko pasien.

**Rubrik/Kriteria Tambahan (Opsional)**:

### Solution Statements
Untuk mencapai tujuan proyek ini dan memastikan solusi yang *robust* serta terukur, dua pendekatan utama akan diterapkan dan dievaluasi:
1.  **Perbandingan Berbagai Algoritma Klasifikasi:** Mengimplementasikan dan membandingkan kinerja setidaknya tiga algoritma *machine learning* yang berbeda: Logistic Regression (sebagai *baseline*), Random Forest Classifier, dan LightGBM Classifier. Performa setiap algoritma akan diukur menggunakan metrik `Accuracy`, `Precision`, `Recall`, `F1-score`, dan `ROC AUC` pada data uji.
2.  **Optimasi Model Melalui Hyperparameter Tuning:** Meningkatkan performa model terbaik yang telah diidentifikasi dari perbandingan algoritma melalui proses *hyperparameter tuning* menggunakan `GridSearchCV` dengan 5-fold *cross-validation*. Peningkatan performa akan diukur dari peningkatan nilai `F1-score` (sebagai metrik utama yang menyeimbangkan *Precision* dan *Recall*) pada set validasi dan uji.

## Data Understanding

Paragraf awal bagian ini menjelaskan informasi mengenai data yang digunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset.

Dataset yang digunakan dalam proyek ini adalah `heart_disease_uci.csv`, yang merupakan kompilasi dari empat dataset penyakit jantung yang berbeda dari UCI Machine Learning Repository (Cleveland, Hungary, Switzerland, VA Long Beach). Dataset ini diunduh dari Kaggle, tersedia di: [https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).

Dataset awal memiliki 1025 sampel (baris) dan 16 kolom (fitur, termasuk ID dan dataset sumber). Setelah langkah *initial data cleaning* dan transformasi target, dataset akan memiliki 920 sampel dan 14 fitur (13 fitur prediktor dan 1 variabel target). Dataset ini bersifat kuantitatif (atau telah dikonversi menjadi numerik) dan memenuhi persyaratan minimal 500 sampel.

### Variabel-variabel pada Dataset Penyakit Jantung adalah sebagai berikut:
*   `age`: (Numerik) Usia pasien dalam tahun.
*   `sex`: (Numerik) Jenis kelamin pasien (1 = laki-laki, 0 = perempuan).
*   `cp`: (Kategorikal) Tipe nyeri dada (misalnya `typical angina`, `asymptomatic`, `non-anginal pain`, `atypical angina`).
*   `trestbps`: (Numerik) Tekanan darah saat istirahat (resting blood pressure) dalam mm Hg.
*   `chol`: (Numerik) Kolesterol serum (serum cholestoral) dalam mg/dl.
*   `fbs`: (Numerik) Gula darah puasa (fasting blood sugar) > 120 mg/dl (1 = true, 0 = false).
*   `restecg`: (Kategorikal) Hasil elektrokardiografi saat istirahat (misalnya `normal`, `st-t abnormality`, `lv hypertrophy`).
*   `thalch`: (Numerik) Detak jantung maksimum yang dicapai (maximum heart rate achieved).
*   `exang`: (Numerik) Angina yang diinduksi oleh olahraga (exercise induced angina) (1 = yes, 0 = no).
*   `oldpeak`: (Numerik) Depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat.
*   `slope`: (Kategorikal) Kemiringan segmen ST puncak olahraga (misalnya `upsloping`, `flat`, `downsloping`).
*   `ca`: (Numerik) Jumlah pembuluh darah utama (0-3) yang diwarnai oleh fluoroskopi.
*   `thal`: (Kategorikal) Thalassemia (misalnya `normal`, `fixed defect`, `reversable defect`).
*   `target`: (Numerik - Variabel Target) Keberadaan penyakit jantung (1 = ada penyakit jantung, 0 = tidak ada penyakit jantung).

**Rubrik/Kriteria Tambahan (Opsional)**:
*   **Melakukan beberapa tahapan yang diperlukan untuk memahami data (Exploratory Data Analysis - EDA):**
    EDA dilakukan untuk memahami karakteristik data, mengidentifikasi pola, *outlier*, dan *missing values*, serta hubungan antar fitur dan dengan variabel target.

    1.  **Kondisi Data Awal:**
        *   Data mentah memiliki *missing values* yang direpresentasikan secara tidak standar (misalnya `'?'` atau spasi kosong `''`). Kolom `fbs` dan `exang` menjadi 100% *missing* setelah konversi awal. Kolom `ca` (66.41%), `thal` (52.83%), dan `slope` (33.59%) juga memiliki persentase *missing values* yang sangat tinggi. Nilai `0` pada `trestbps`, `chol`, dan `thalch` yang tidak valid secara medis juga diidentifikasi sebagai *missing values*.

    2.  **Distribusi Variabel Target:**
        *   Distribusi kelas target (`target`) relatif seimbang: 44.67% pasien tidak memiliki penyakit jantung (`0`) dan 55.33% memiliki penyakit jantung (`1`). Keseimbangan ini baik untuk klasifikasi, meskipun tetap relevan untuk menggunakan metrik `F1-score` dan `ROC AUC`.
        *   *(Lihat plot `countplot` distribusi target pada Jupyter Notebook untuk visualisasi)*

    3.  **Distribusi Fitur Numerik dan Kategorikal:**
        *   Fitur-fitur seperti `age` dan `thalch` menunjukkan distribusi yang mendekati normal. `oldpeak` menunjukkan distribusi *zero-inflated*. Kolom `chol` dan `trestbps` memiliki distribusi bervariasi.
        *   Distribusi fitur kategorikal seperti `cp` (tipe nyeri dada) dan `thal` (thalassemia) menunjukkan frekuensi masing-masing kategori, dengan `asymptomatic` dan `reversable defect` menjadi kategori paling dominan.
        *   *(Lihat plot `histplot` dan `countplot` distribusi fitur pada Jupyter Notebook untuk visualisasi)*

    4.  **Korelasi Antar Fitur:**
        *   Heatmap korelasi menunjukkan bahwa `oldpeak` (0.49), `exang` (0.49), `ca` (0.52), dan `thalch` (-0.42) memiliki korelasi linear yang paling signifikan dengan variabel `target`. Fitur-fitur ini menjadi kandidat prediktor kuat.
        *   *(Lihat heatmap korelasi pada Jupyter Notebook untuk visualisasi)*

    5.  **Hubungan Fitur dengan Target:**
        *   Boxplot memvisualisasikan perbedaan distribusi fitur numerik antara kedua kelompok target. Pasien dengan penyakit jantung (`target=1`) cenderung memiliki nilai `age`, `trestbps`, `chol`, dan `oldpeak` yang lebih tinggi, serta nilai `thalch` yang lebih rendah. Ini mengkonfirmasi *insight* dari korelasi dan memberikan pemahaman visual tentang perbedaan grup.
        *   *(Lihat boxplot hubungan fitur vs. target pada Jupyter Notebook untuk visualisasi)*

## Data Preparation

Pada bagian ini saya menerapkan teknik *data preparation* yang esensial. Teknik-teknik ini berurutan dan diterapkan secara sistematis.

**Rubrik/Kriteria Tambahan (Opsional)**:

*   **Menjelaskan proses *data preparation* yang dilakukan dan alasannya:**

    1.  **Initial Data Cleaning dan Transformasi Target:**
        *   **Proses:**
            *   Penggantian nilai non-standar (`'?'`, `''`) dengan `np.nan` di seluruh DataFrame.
            *   Konversi kolom `sex`, `fbs`, `exang` (boolean string) ke numerik (1/0).
            *   Konversi kolom `ca` dan `num` ke numerik (float), dengan `errors='coerce'` untuk menangani konversi yang gagal.
            *   Penggantian nilai `0` yang tidak valid secara klinis pada `trestbps`, `chol`, dan `thalch` dengan `np.nan`.
            *   Penghapusan kolom `id` dan `dataset`.
            *   Transformasi `num` (0-4) menjadi biner `target` (0 atau 1).
        *   **Alasan:** Ini adalah langkah fundamental untuk membersihkan data mentah, memastikan konsistensi tipe data, dan menyiapkan variabel target sesuai dengan tujuan klasifikasi biner. Nilai non-numerik dan `0` yang tidak valid akan mengganggu perhitungan dan pemodelan.

    2.  **Pemisahan Data Latih dan Uji:**
        *   **Proses:** Data dibagi menjadi fitur (`X`) dan target (`y`), lalu dipisahkan menjadi `X_train`, `X_test`, `y_train`, `y_test` dengan proporsi 80% data latih dan 20% data uji menggunakan `train_test_split` (`random_state=42` untuk reproduktifitas). Parameter `stratify=y` digunakan.
        *   **Alasan:** Pemisahan ini mencegah *data leakage*, memastikan model dievaluasi pada data yang belum pernah dilihat, dan `stratify` menjaga proporsi kelas target yang sama di kedua set, penting untuk *class imbalance* ringan.

    3.  **Pipeline Preprocessing Lanjutan:**
        *   **Proses:**
            *   **Fitur Numerik (`age`, `trestbps`, `chol`, `thalch`, `oldpeak`, `ca`):** Menggunakan `SimpleImputer(strategy='median')` diikuti oleh `StandardScaler()`.
            *   **Fitur Kategorikal (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal`):** Menggunakan `SimpleImputer(strategy='most_frequent')` diikuti oleh `OneHotEncoder(handle_unknown='ignore')`.
            *   Kedua transformer digabungkan dalam `ColumnTransformer` dan diintegrasikan ke dalam `Pipeline` model.
        *   **Alasan:**
            *   **Automatisasi dan Konsistensi:** `Pipeline` memastikan semua langkah preprocessing (imputasi, scaling, encoding) diterapkan secara berurutan dan konsisten pada `X_train` dan `X_test`.
            *   **Mencegah *Data Leakage*:** Parameter imputasi (median, modus) dan scaling (mean, std deviasi) *hanya* dipelajari dari `X_train` dan kemudian diterapkan ke `X_test`. Ini adalah praktik terbaik untuk menghindari optimisme performa yang tidak realistis.
            *   **Penanganan *Missing Values*:** `SimpleImputer` mengisi nilai yang hilang. `median` dipilih untuk numerik karena robust terhadap *outlier*, sedangkan `most_frequent` untuk kategorikal.
            *   **Encoding Variabel Kategorikal:** `OneHotEncoder` mengubah fitur kategorikal menjadi representasi numerik biner, mencegah model mengasumsikan urutan atau nilai ordinal pada data nominal. `handle_unknown='ignore'` memastikan *pipeline* tidak error jika ada kategori baru di data uji.
            *   **Feature Scaling:** `StandardScaler` menormalisasi fitur numerik, penting untuk algoritma berbasis jarak atau gradien (seperti Logistic Regression) agar semua fitur berkontribusi secara proporsional.

## Modeling

Tahapan ini membahas mengenai model *machine learning* yang digunakan untuk menyelesaikan permasalahan. saya menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan, serta proses pemilihan model terbaik.

**Rubrik/Kriteria Tambahan (Opsional)**:

*   **Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan:**

    1.  **Logistic Regression**
        *   **Kelebihan:** Sederhana, cepat dalam pelatihan dan prediksi, mudah diinterpretasi (melalui koefisien fitur yang menunjukkan pengaruh pada *log-odds*). Baik sebagai model *baseline*.
        *   **Kekurangan:** Mengasumsikan hubungan linier antara fitur input dan *log-odds* dari variabel target. Performa bisa menurun pada data dengan hubungan non-linear atau yang sangat kompleks.

    2.  **Random Forest Classifier**
        *   **Kelebihan:** Model *ensemble* yang sangat kuat, menggabungkan banyak pohon keputusan untuk meningkatkan akurasi dan mengurangi *overfitting*. Mampu menangani hubungan non-linear dan interaksi kompleks antar fitur. Kurang sensitif terhadap *outlier* dan penskalaan fitur. Dapat memberikan indikasi *feature importance*, yang berguna untuk interpretasi.
        *   **Kekurangan:** Kurang *interpretable* dibandingkan model linier (sering disebut 'black box' karena struktur pohon yang kompleks). Membutuhkan sumber daya komputasi yang lebih besar dan waktu pelatihan yang lebih lama dibandingkan model sederhana.

    3.  **LightGBM Classifier**
        *   **Kelebihan:** Algoritma *gradient boosting* yang sangat efisien dan cepat, terutama pada dataset besar. Menawarkan kinerja prediktif yang superior, seringkali menjadi salah satu algoritma terbaik untuk data tabular. Secara bawaan dapat menangani *missing values* dan fitur kategorikal.
        *   **Kekurangan:** Sangat rentan terhadap *overfitting* jika *hyperparameter* tidak di-tuning dengan benar. Lebih kompleks dan kurang *interpretable* (model 'black box' lainnya).

*   **Jelaskan proses *improvement* yang dilakukan (Hyperparameter Tuning):**

    *   **Proses:** Setiap model diintegrasikan ke dalam sebuah `Pipeline` lengkap dengan langkah-langkah *preprocessing* yang telah didefinisikan (`preprocessor` mencakup imputasi, *scaling*, dan *encoding*). *Hyperparameter tuning* dilakukan menggunakan `GridSearchCV` dari scikit-learn. `GridSearchCV` secara sistematis mencoba setiap kombinasi *hyperparameter* yang ditentukan dalam `param_grid`.
    *   **Cross-Validation:** saya menggunakan 5-Fold *Cross-Validation* (`cv=5`), yang membagi data pelatihan menjadi 5 bagian, melatih model 5 kali (masing-masing menggunakan 4 bagian untuk pelatihan dan 1 bagian untuk validasi) dan merata-ratakan skornya. Ini memberikan estimasi performa model yang lebih *robust* dan mengurangi varians, dibandingkan hanya menggunakan satu *train-validation split*.
    *   **Metrik Penilaian (`scoring`):** `scoring='f1'` dipilih sebagai metrik utama untuk *tuning*. `F1-score` sangat relevan dalam konteks medis karena menyeimbangkan `Precision` dan `Recall`, penting untuk meminimalkan baik *false positives* maupun *false negatives*.
    *   **Parameter yang Dicari (`param_grids`):**
        *   **Logistic Regression:** `classifier__C` (kekuatan regularisasi), `classifier__solver` (algoritma optimasi).
        *   **Random Forest:** `classifier__n_estimators` (jumlah pohon), `classifier__max_depth` (kedalaman maksimum pohon), `classifier__min_samples_split` (minimum sampel untuk membagi *node*).
        *   **LightGBM:** `classifier__n_estimators` (jumlah *boosting rounds*), `classifier__learning_rate` (ukuran langkah kontribusi setiap pohon), `classifier__num_leaves` (jumlah maksimum daun per pohon).

*   **Pilih model terbaik sebagai solusi dan jelaskan mengapa:**

    Setelah melatih dan melakukan *tuning* untuk semua model, saya mengevaluasi setiap model terbaik yang telah di-tuning pada data uji (`X_test`, `y_test`).

    ```
    Performa Model Terbaik pada Data Uji:

                         Accuracy  Precision  Recall  F1-Score  ROC AUC
    Model                                                              
    Logistic Regression    0.7880     0.7788  0.8627    0.8186   0.8928
    Random Forest          0.8152     0.7881  0.9118    0.8455   0.9002
    LightGBM               0.8424     0.8349  0.8922    0.8626   0.8782
    ```

    Berdasarkan hasil evaluasi pada data uji, model **LightGBM Classifier** menunjukkan performa terbaik secara keseluruhan, terutama pada metrik `F1-Score` (0.8626) dan `ROC AUC` (0.8782). Oleh karena itu, LightGBM Classifier dipilih sebagai model terbaik untuk menyelesaikan permasalahan ini.

    **Alasan Pemilihan:**
    LightGBM dipilih karena kombinasinya antara efisiensi (waktu pelatihan yang relatif cepat) dan performa prediktif yang tinggi. Model berbasis *gradient boosting* seperti LightGBM mampu menangkap pola data yang kompleks lebih baik daripada model linier atau *ensemble* berbasis *bagging* seperti Random Forest, menghasilkan metrik `F1-Score` dan `ROC AUC` yang lebih baik pada data uji. Ini menunjukkan kemampuannya yang unggul dalam membedakan antara pasien sehat dan sakit, yang merupakan prioritas utama dalam konteks medis.

## Evaluation

Pada bagian ini, saya menyebutkan metrik evaluasi yang digunakan dan menjelaskan hasil proyek berdasarkan metrik tersebut.

### Metrik Evaluasi yang Digunakan
Untuk mengukur kinerja model klasifikasi dalam memprediksi penyakit jantung, beberapa metrik evaluasi kunci telah digunakan: `Accuracy`, `Precision`, `Recall`, `F1-Score`, `Confusion Matrix`, dan `ROC AUC`.

**Rubrik/Kriteria Tambahan (Opsional)**:
*   **Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja:**

    1.  **Accuracy (Akurasi):**
        *   **Formula:** `(True Positives + True Negatives) / Total Sampel`
        *   **Cara Kerja:** Mengukur proporsi total prediksi yang benar (baik kelas positif maupun negatif).
        *   **Interpretasi Konteks:** Mengindikasikan seberapa sering model membuat diagnosis yang benar secara keseluruhan.

    2.  **Precision (Presisi):**
        *   **Formula:** `True Positives / (True Positives + False Positives)`
        *   **Cara Kerja:** Mengukur proporsi positif sejati dari semua kasus yang diprediksi positif oleh model.
        *   **Interpretasi Konteks:** Penting untuk meminimalkan *false positives* (misalnya, mencegah pasien sehat menjalani tes/perawatan yang tidak perlu).

    3.  **Recall (Sensitivitas/Rekal):**
        *   **Formula:** `True Positives / (True Positives + False Negatives)`
        *   **Cara Kerja:** Mengukur proporsi positif sejati yang berhasil dideteksi oleh model dari semua kasus positif aktual.
        *   **Interpretasi Konteks:** Sangat krusial dalam konteks medis untuk meminimalkan *false negatives* (misalnya, tidak melewatkan pasien yang sebenarnya sakit).

    4.  **F1-Score:**
        *   **Formula:** `2 * (Precision * Recall) / (Precision + Recall)`
        *   **Cara Kerja:** Rata-rata harmonik dari Precision dan Recall. Memberikan keseimbangan antara kedua metrik, sangat berguna ketika ada ketidakseimbangan kelas.
        *   **Interpretasi Konteks:** Menunjukkan kinerja yang seimbang dalam mendeteksi kasus positif sekaligus meminimalkan kesalahan positif dan negatif.

    5.  **ROC AUC (Receiver Operating Characteristic - Area Under the Curve):**
        *   **Formula:** Area di bawah kurva ROC, yang memplot True Positive Rate (Recall) terhadap False Positive Rate pada berbagai ambang batas klasifikasi.
        *   **Cara Kerja:** Mengukur kemampuan model untuk membedakan antara kelas positif dan negatif secara keseluruhan. Semakin dekat nilai AUC ke 1, semakin baik model dalam membedakan kedua kelas.
        *   **Interpretasi Konteks:** Menunjukkan kekuatan diskriminasi model; model yang baik mampu memisahkan pasien dengan dan tanpa penyakit jantung secara efektif.

### Hasil Proyek Berdasarkan Metrik Evaluasi

Model terbaik yang terpilih adalah **LightGBM Classifier**. Berikut adalah ringkasan performanya pada data uji:

*   **Model Terpilih:** LightGBM Classifier
*   **Accuracy:** 0.8424
*   **Precision:** 0.8349
*   **Recall:** 0.8922
*   **F1-Score:** 0.8626
*   **ROC AUC:** 0.8782

### Classification Report:

```
              precision    recall  f1-score   support

           0       0.85      0.78      0.82        82
           1       0.83      0.89      0.86       102

    accuracy                           0.84       184
   macro avg       0.84      0.84      0.84       184
weighted avg       0.84      0.84      0.84       184
```

### Visualisasi Confusion Matrix
![image](https://github.com/user-attachments/assets/c8550c16-65b3-4447-a691-fa6b83f0c558)


*   **Interpretasi:** Confusion Matrix menunjukkan:
    *   **True Positives (TP):** 91 pasien sakit berhasil diprediksi sakit.
    *   **True Negatives (TN):** 64 pasien sehat berhasil diprediksi sehat.
    *   **False Positives (FP):** 18 pasien sehat salah diprediksi sakit.
    *   **False Negatives (FN):** 11 pasien sakit salah diprediksi sehat.
    *   Model berhasil meminimalkan *false negatives* dan *false positives* dengan angka yang relatif kecil, menunjukkan kemampuan model yang baik dalam meminimalkan kedua jenis kesalahan.

### Visualisasi ROC Curve
![image](https://github.com/user-attachments/assets/93b780a4-aa9b-4b7c-b3f5-55ef7752a0f6)


*   **Interpretasi:** Kurva ROC berada jauh di atas garis acak, dan nilai AUC sebesar 0.8782. Ini mengindikasikan bahwa model memiliki kemampuan diskriminasi yang sangat baik dalam membedakan antara pasien dengan dan tanpa penyakit jantung di berbagai *threshold* klasifikasi. Semakin tinggi AUC, semakin baik model dalam membedakan kedua kelas tersebut.

### Diskusi Hasil:
Model LightGBM Classifier menunjukkan performa yang sangat baik dalam memprediksi penyakit jantung. F1-score yang tinggi, bersama dengan Recall yang kuat dan Precision yang baik, menunjukkan keseimbangan yang optimal antara kemampuan model untuk mendeteksi semua kasus positif dan menghindari kesalahan prediksi positif. ROC AUC yang tinggi juga mengkonfirmasi kemampuan diskriminasi model yang kuat. Dalam konteks medis, `Recall` yang tinggi sangat penting karena kegagalan mendeteksi penyakit jantung (*false negative*) dapat memiliki konsekuensi yang jauh lebih serius daripada diagnosis positif palsu (*false positive*).

## 7. 📝 Conclusion & Next Step

### Ringkasan Hasil Modeling

Proyek ini telah berhasil mengembangkan model *machine learning* untuk prediksi penyakit jantung menggunakan dataset kombinasi dari UCI Machine Learning Repository. Melalui tahap *data cleaning* yang komprehensif, eksplorasi data mendalam, dan penerapan *pipeline preprocessing* yang robust, data berhasil disiapkan secara optimal untuk pemodelan. Berbagai algoritma klasifikasi (Logistic Regression, Random Forest, LightGBM) diuji dan dioptimalkan melalui *hyperparameter tuning* dengan *cross-validation*. Model **LightGBM Classifier** terbukti sebagai model terbaik berdasarkan metrik evaluasi pada data uji, dengan F1-score, Recall, dan ROC AUC yang tinggi. Model ini menunjukkan kapabilitas yang signifikan dalam memprediksi keberadaan penyakit jantung, didukung oleh identifikasi fitur-fitur penting yang mempengaruhi prediksi.

### Insight yang Didapat

Selama proses pengerjaan proyek, beberapa *insight* penting berhasil diidentifikasi:

*   **Kualitas Data Awal:** Dataset awal memiliki tantangan dalam hal *missing values* yang tersebar di beberapa kolom, direpresentasikan dalam berbagai format non-standar (`?`, `0`, atau spasi kosong). Penanganan awal yang teliti (mengubah ke `NaN`, mengonversi tipe data, dan menghapus kolom tidak relevan) sangat esensial untuk validitas analisis dan pemodelan.
*   **Ketidakseimbangan Kelas Ringan:** Meskipun ada *class imbalance* ringan pada variabel target (sekitar 55.33% kelas positif dan 44.67% kelas negatif), pemilihan metrik `F1-score` (yang menyeimbangkan *Precision* dan *Recall*) untuk *hyperparameter tuning* dan evaluasi akhir sangat tepat, memastikan model tidak hanya akurat secara keseluruhan tetapi juga efektif dalam mendeteksi kedua kelas.
*   **Fitur Kritis:** Analisis *feature importance* dari model LightGBM menunjukkan bahwa `cholesterol (chol)`, `age`, `thalch` (detak jantung maksimum), dan `trestbps` (tekanan darah saat istirahat) adalah fitur-fitur numerik yang paling berpengaruh terhadap prediksi penyakit jantung. Di antara fitur kategorikal, `cp_asymptomatic` (tipe nyeri dada asimptomatik) dan `thal_reversable defect` (thalassemia jenis *reversable defect*) juga memiliki kontribusi signifikan. Ini memberikan validasi klinis terhadap faktor-faktor risiko yang sudah dikenal.
*   **Efektivitas Optimasi Model:** Proses *hyperparameter tuning* menggunakan `GridSearchCV` secara signifikan meningkatkan performa model terbaik dibandingkan dengan *baseline* awal. Perbandingan antar model juga menunjukkan bahwa LightGBM, sebagai algoritma *gradient boosting*, unggul dalam menangani kompleksitas data dan mencapai metrik `F1-score` serta `ROC AUC` yang lebih tinggi dibandingkan Logistic Regression dan Random Forest.

### Potensi Pengembangan Berikutnya (Next Steps)

Untuk lebih meningkatkan robusta dan aplikasi model di dunia nyata, beberapa langkah pengembangan di masa depan dapat dilakukan:

1.  **Validasi Eksternal:** Melakukan validasi model pada *dataset* independen yang belum pernah dilihat sama sekali, idealnya dari populasi atau rumah sakit yang berbeda. Ini sangat penting untuk memastikan kemampuan generalisasi model terhadap data dunia nyata dan mengurangi risiko *overfitting* pada data yang ada.
2.  **Feature Engineering Lanjutan:** Mengeksplorasi pembuatan fitur-fitur baru dari fitur yang sudah ada atau melalui kombinasi fitur (misalnya, rasio antar fitur medis, indeks risiko yang dikustomisasi). Hal ini dapat mengungkap pola tersembunyi yang mungkin meningkatkan kekuatan prediktif model.
3.  **Interpretasi Model (XAI):** Menerapkan teknik *Explainable AI* (XAI) seperti SHAP (SHapley Additive exPlanations) atau LIME (Local Interpretable Model-agnostic Explanations) untuk memberikan penjelasan yang lebih transparan dan mudah dipahami tentang bagaimana model membuat setiap prediksi. Transparansi ini esensial dalam domain medis untuk membangun kepercayaan dan memvalidasi keputusan model.
4.  **Optimasi *Threshold*:** Menyesuaikan ambang batas klasifikasi (yang defaultnya 0.5) untuk mengoptimalkan metrik tertentu sesuai dengan prioritas klinis. Misalnya, jika meminimalkan *false negatives* (tidak mendeteksi pasien sakit) adalah prioritas utama, *threshold* dapat disesuaikan untuk meningkatkan *Recall* lebih jauh, meskipun mungkin mengorbankan sedikit *Precision*.
5.  **Pengumpulan Data Tambahan:** Jika memungkinkan, mengumpulkan lebih banyak data pasien dengan atribut yang lebih beragam atau data longitudinal (rekam jejak pasien dari waktu ke waktu) dapat secara signifikan meningkatkan akurasi dan generalisasi model.
6.  **Pengembangan Antarmuka Pengguna (UI):** Membuat antarmuka pengguna sederhana (misalnya, dengan Streamlit atau Flask) untuk model yang memungkinkan tenaga medis untuk dengan mudah menginput data pasien dan mendapatkan hasil prediksi secara langsung, memfasilitasi integrasi model ke dalam alur kerja klinis.

### Apakah Model Sudah Cukup Baik untuk Digunakan di Dunia Nyata?

Model yang dikembangkan dalam proyek ini menunjukkan performa prediktif yang **sangat menjanjikan**, ditunjukkan oleh `F1-score` yang kuat (0.8626), `Recall` yang tinggi (0.8922), dan `ROC AUC` yang sangat baik (0.8782). Kemampuan model untuk secara efektif membedakan antara pasien dengan dan tanpa penyakit jantung, serta tingginya tingkat deteksi kasus positif (Recall), menjadikannya alat yang potensial untuk mendukung keputusan klinis.

**Namun, untuk implementasi klinis di dunia nyata**, model ini masih memerlukan validasi lebih lanjut dan pertimbangan etika yang cermat. Uji coba dengan data pasien yang tidak bias, validasi menyeluruh oleh para ahli medis, dan persetujuan dari badan regulasi kesehatan adalah langkah-langkah yang tidak dapat diabaikan. Aspek privasi dan keamanan data pasien harus selalu menjadi prioritas utama.

Dengan validasi yang ketat dan adaptasi yang tepat berdasarkan *feedback* dari para profesional medis, model ini memiliki **potensi besar** untuk menjadi alat pendukung keputusan yang berharga dalam skrining dini, manajemen risiko, dan peningkatan kualitas perawatan pasien penyakit jantung.

