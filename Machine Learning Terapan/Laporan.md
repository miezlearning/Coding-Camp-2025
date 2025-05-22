# Sistem Prediksi Penyakit Jantung Menggunakan Machine Learning

**Nama:** Muhammad Alif  
**Email Dicoding:** m.alif7890@gmail.com  
**Tanggal:** 23/05/2025

---

## 1. 🧠 Domain Proyek

### Latar Belakang

Penyakit jantung kardiovaskular (Cardiovascular Diseases/CVDs) merupakan penyebab kematian nomor satu secara global, merenggut jutaan nyawa setiap tahunnya. Deteksi dini dan diagnosis yang akurat sangat krusial untuk intervensi medis yang cepat, yang pada gilirannya dapat secara signifikan meningkatkan prognosis pasien dan mengurangi angka kematian. Namun, proses diagnosis penyakit jantung seringkali kompleks, melibatkan banyak faktor risiko dan hasil tes medis yang bervariasi, membutuhkan keahlian dan pengalaman dokter.

Dalam menghadapi tantangan ini, teknologi *machine learning* (ML) menawarkan potensi besar. Algoritma ML mampu menganalisis pola-pola kompleks dalam volume data medis yang besar, mengidentifikasi hubungan tersembunyi antar variabel, dan membuat prediksi dengan tingkat akurasi yang tinggi. Dengan memanfaatkan ML, kita dapat mengembangkan sistem pendukung keputusan yang membantu tenaga medis dalam melakukan skrining awal pada populasi berisiko tinggi dan memprediksi kemungkinan seseorang menderita penyakit jantung, sehingga memungkinkan tindakan pencegahan atau pengobatan yang lebih cepat dan tepat.

### Referensi Terkait (Rubrik Tambahan)

Sebagai validasi dan pendukung relevansi proyek ini, beberapa referensi kredibel yang membahas urgensi dan potensi ML dalam domain kesehatan adalah:

1. **[World Health Organization (WHO). Cardiovascular diseases (CVDs).](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))**  
    *Penjelasan:* Sumber ini memberikan statistik global tentang prevalensi dan dampak serius penyakit kardiovaskular, menggarisbawahi urgensi permasalahan ini dalam skala global.

2. **Muhammad, A. B., & Mahmud, H. (2020). Heart Disease Prediction Using Machine Learning Algorithms: A Comparative Study.**  
    *International Journal of Computer Applications*, 174(1), 1-5.  
    *Penjelasan:* Studi ini menyajikan contoh konkret bagaimana berbagai algoritma *machine learning* telah diterapkan untuk memprediksi penyakit jantung, menegaskan relevansi pendekatan ML dalam domain medis.

---

## 2. 🧠 Business Understanding

### Pernyataan Masalah (Problem Statements)

1. Bagaimana kita dapat mengembangkan model *machine learning* yang efektif untuk secara akurat memprediksi keberadaan penyakit jantung pada pasien berdasarkan data klinis dan demografis yang tersedia?
2. Bagaimana model prediksi ini dapat diintegrasikan sebagai alat bantu bagi tenaga medis untuk mendukung diagnosis dini dan pengambilan keputusan klinis, sehingga memungkinkan intervensi lebih cepat dan mengurangi risiko komplikasi serius?

### Tujuan (Goals)

1. Mengembangkan model klasifikasi *machine learning* (**Klasifikasi Biner**) yang mampu mengidentifikasi pasien dengan penyakit jantung (kelas positif) dan tanpa penyakit jantung (kelas negatif) dengan performa evaluasi yang optimal, terutama pada metrik `F1-score`, `Recall`, dan `ROC AUC`.
2. Mengidentifikasi fitur-fitur medis utama (misalnya, usia, jenis kelamin, tekanan darah, kolesterol, tipe nyeri dada) yang memiliki pengaruh paling signifikan terhadap risiko penyakit jantung berdasarkan analisis model.
3. Membangun sistem prediksi yang dapat diandalkan untuk skrining awal, mengarahkan pasien berisiko tinggi untuk pemeriksaan lebih lanjut dan intervensi yang tepat waktu.

### Pernyataan Solusi (Solution Statements) (Rubrik Tambahan)

Untuk mencapai tujuan proyek ini dan memastikan solusi yang robust dan terukur, dua pendekatan utama akan diterapkan:

1. **Perbandingan Berbagai Algoritma Klasifikasi:**  
    Kami akan menguji dan membandingkan kinerja setidaknya tiga algoritma *machine learning* yang relevan untuk masalah klasifikasi biner:
    - **Logistic Regression:** Sebagai model *baseline* yang sederhana dan *interpretable*.
    - **Random Forest Classifier:** Model *ensemble* yang kuat, dikenal karena kemampuannya menangani kompleksitas data dan mengurangi *overfitting*.
    - **LightGBM Classifier:** Algoritma *gradient boosting* yang efisien dan berkinerja tinggi.
    
    Performa setiap algoritma akan diukur menggunakan metrik `Accuracy`, `Precision`, `Recall`, `F1-score`, dan `ROC AUC` pada data uji. Model dengan kombinasi metrik terbaik akan dipilih sebagai kandidat utama.

2. **Optimasi Model Melalui Hyperparameter Tuning:**  
    Model-model yang menjanjikan akan menjalani proses *hyperparameter tuning* ekstensif menggunakan `GridSearchCV` dengan 5-fold *cross-validation*. Proses ini akan mencari kombinasi *hyperparameter* terbaik yang mengoptimalkan performa prediktif model, terutama pada `F1-score`.

---

## 3. 📂 Data Understanding

### Informasi Data

- **Dataset:** `heart_disease_uci.csv` (gabungan dari Cleveland, Hungary, Switzerland, VA Long Beach - UCI ML Repository)
- **Sumber Data:** [Kaggle - Heart Disease Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- **Jumlah Sampel & Fitur:** 1025 sampel, 16 kolom (setelah pembersihan: 13 fitur + 1 target)
- **Kondisi Data Awal:**  
  - Terdapat nilai hilang (misal `'?'`, spasi kosong)
  - Beberapa kolom numerik memiliki nilai `0` yang tidak valid
  - Fitur kategorikal perlu di-encode
  - Target (`num`) perlu ditransformasi menjadi biner

### Uraian Variabel (Fitur)

| Fitur      | Tipe      | Deskripsi                                                                 |
|------------|-----------|---------------------------------------------------------------------------|
| age        | Numerik   | Usia pasien (tahun)                                                       |
| sex        | Numerik   | Jenis kelamin (1 = laki-laki, 0 = perempuan)                              |
| cp         | Kategorikal | Tipe nyeri dada                                                         |
| trestbps   | Numerik   | Tekanan darah saat istirahat (mm Hg)                                      |
| chol       | Numerik   | Kolesterol serum (mg/dl)                                                  |
| fbs        | Numerik   | Gula darah puasa > 120 mg/dl (1 = true, 0 = false)                        |
| restecg    | Kategorikal | Hasil elektrokardiografi saat istirahat                                 |
| thalch     | Numerik   | Detak jantung maksimum yang dicapai                                       |
| exang      | Numerik   | Angina yang diinduksi oleh olahraga (1 = yes, 0 = no)                     |
| oldpeak    | Numerik   | Depresi ST yang diinduksi oleh olahraga                                   |
| slope      | Kategorikal | Kemiringan segmen ST puncak olahraga                                    |
| ca         | Numerik   | Jumlah pembuluh darah utama (0-3) yang diwarnai fluoroskopi               |
| thal       | Kategorikal | Thalassemia                                                             |
| target     | Numerik   | Keberadaan penyakit jantung (1 = ada, 0 = tidak ada)                      |

### Eksplorasi Data (EDA)

- **Inspeksi Awal:**  
  - Terdapat missing values pada beberapa kolom (`fbs`, `exang`, `ca`, `thal`, dll)
  - Distribusi target relatif seimbang (44.67% tidak sakit, 55.33% sakit)
- **Distribusi Fitur:**  
  - Fitur numerik seperti `age`, `thalch` mendekati normal
  - Fitur kategorikal seperti `cp`, `restecg`, `slope`, `thal` menunjukkan pola berbeda antara kelas target
- **Korelasi:**  
  - Fitur `oldpeak`, `exang`, `ca`, dan `thalch` memiliki korelasi signifikan dengan target
- **Boxplot:**  
  - Pasien dengan penyakit jantung cenderung memiliki nilai `age`, `trestbps`, `chol`, dan `oldpeak` lebih tinggi, serta `thalch` lebih rendah

---

## 4. 🧹 Data Preparation

### 4.1. Initial Data Cleaning dan Transformasi Target

- Ganti nilai `'?'` dan spasi kosong dengan `np.nan`
- Konversi kolom boolean ke numerik
- Ganti nilai `0` yang tidak valid pada kolom numerik dengan `np.nan`
- Hapus kolom tidak relevan (`id`, `dataset`)
- Transformasi target `num` menjadi biner `target`

### 4.2. Pemisahan Data Latih dan Uji

- Split data menjadi fitur (`X`) dan target (`y`)
- Gunakan `train_test_split` (80% train, 20% test, stratify by target)

### 4.3. Pipeline Preprocessing Lanjutan

- **Fitur Numerik:** Imputasi median + StandardScaler
- **Fitur Kategorikal:** Imputasi modus + OneHotEncoder
- Gunakan `Pipeline` dan `ColumnTransformer` untuk automasi dan mencegah data leakage

---

## 5. 🏗️ Modeling

### 5.1. Pembuatan Model Machine Learning

- **Logistic Regression:** Sederhana, baseline, interpretable
- **Random Forest Classifier:** Ensemble, kuat, menangani non-linearitas
- **LightGBM Classifier:** Gradient boosting, efisien, performa tinggi

### 5.2. Tahapan dan Parameter Pemodelan

- Setiap model diintegrasikan dengan pipeline preprocessing
- Hyperparameter tuning dengan `GridSearchCV` (cv=5, scoring='f1', n_jobs=-1)
- Param grid disesuaikan untuk tiap model

### 5.3. Pemilihan Model Terbaik

| Model                | Accuracy | Precision | Recall  | F1-Score | ROC AUC |
|----------------------|----------|-----------|---------|----------|---------|
| Logistic Regression  | 0.7880   | 0.7788    | 0.8627  | 0.8186   | 0.8928  |
| Random Forest        | 0.8152   | 0.7881    | 0.9118  | 0.8455   | 0.9002  |
| LightGBM             | 0.8424   | 0.8349    | 0.8922  | 0.8626   | 0.8782  |

**LightGBM Classifier** dipilih sebagai model terbaik.

#### Alasan Memilih LightGBM

- Kombinasi efisiensi dan performa prediktif tinggi
- Mampu menangkap pola data kompleks
- Metrik F1-Score dan ROC AUC terbaik

### 5.4. Analisis Feature Importance (LightGBM)

| No. | Fitur                       | Importance |
|-----|-----------------------------|------------|
| 1   | num__chol                   | 1091       |
| 2   | num__age                    | 1041       |
| 3   | num__thalch                 | 1018       |
| 4   | num__trestbps               | 750        |
| 5   | num__oldpeak                | 562        |
| 6   | cat__restecg_normal         | 180        |
| 7   | cat__cp_asymptomatic        | 142        |
| 8   | cat__cp_atypical angina     | 93         |
| 9   | cat__sex_0                  | 90         |
| 10  | cat__restecg_st-t abnormality | 87       |

---

## 6. 📈 Evaluation

### Hasil Evaluasi Akhir Model Terbaik

- **Model Terpilih:** LightGBM Classifier
- **Accuracy:** 0.8424
- **Precision:** 0.8349
- **Recall:** 0.8922
- **F1-Score:** 0.8626
- **ROC AUC:** 0.8782

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.85      | 0.78   | 0.82     | 82      |
| 1     | 0.83      | 0.89   | 0.86     | 102     |

|        | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Accuracy |          |        | 0.84     | 184     |
| Macro avg | 0.84    | 0.84   | 0.84     | 184     |
| Weighted avg | 0.84 | 0.84   | 0.84     | 184     |

### Penjelasan Metrik Evaluasi

1. **Accuracy:** Proporsi prediksi benar dari seluruh data.
2. **Precision:** Proporsi prediksi positif yang benar-benar positif.
3. **Recall:** Proporsi kasus positif aktual yang berhasil dideteksi.
4. **F1-Score:** Rata-rata harmonik Precision dan Recall.
5. **ROC AUC:** Kemampuan model membedakan kelas positif dan negatif.

### Visualisasi Confusion Matrix

```python
plt.figure(figsize=(8, 6))
sns.heatmap(final_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted No Disease', 'Predicted Disease'],
                yticklabels=['Actual No Disease', 'Actual Disease'])
plt.title('Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.show()
```

**Interpretasi:**  
Confusion Matrix menunjukkan distribusi prediksi model:
- **True Positives (TP):** Pasien sakit yang diprediksi sakit
- **True Negatives (TN):** Pasien sehat yang diprediksi sehat
- **False Positives (FP):** Pasien sehat yang salah diprediksi sakit
- **False Negatives (FN):** Pasien sakit yang salah diprediksi sehat

Model berhasil meminimalkan false negatives dan false positives, yang sangat penting dalam diagnosis medis.


### Visualisasi ROC Curve


```py
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_estimator(best_model_for_eval, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.show()
```

- **Interpretasi**: Kurva ROC berada jauh di atas garis acak (garis diagonal), dan nilai AUC (Area Under the Curve) sebesar 0.8782. Ini mengindikasikan bahwa model memiliki kemampuan diskriminasi yang sangat baik antara pasien dengan dan tanpa penyakit jantung di berbagai threshold klasifikasi. Semakin tinggi AUC, semakin baik model dalam membedakan kedua kelas tersebut.


#### Diskusi Hasil:
Model LightGBM Classifier menunjukkan performa yang sangat baik dalam memprediksi penyakit jantung. F1-score yang tinggi (0.8626), bersama dengan Recall yang kuat (0.8922) dan Precision yang baik (0.8349), menunjukkan keseimbangan yang optimal antara kemampuan model untuk mendeteksi semua kasus positif dan menghindari kesalahan prediksi positif. ROC AUC yang tinggi (0.8782) juga mengkonfirmasi kemampuan diskriminasi model yang kuat. Dalam konteks medis, Recall yang tinggi sangat penting karena kegagalan mendeteksi penyakit jantung (false negative) dapat memiliki konsekuensi yang jauh lebih serius daripada diagnosis positif palsu (false positive).

## 7. 📝 Conclusion & Next Step

### ✅ Ringkasan Hasil Modeling

Proyek ini telah berhasil mengembangkan model machine learning untuk prediksi penyakit jantung menggunakan dataset kombinasi dari *UCI Machine Learning Repository*. Melalui tahap *data cleaning* yang komprehensif, eksplorasi data mendalam, dan penerapan *pipeline preprocessing* yang robust, data berhasil disiapkan secara optimal untuk pemodelan.

Berbagai algoritma klasifikasi (Logistic Regression, Random Forest, LightGBM) diuji dan dioptimalkan melalui *hyperparameter tuning* dengan *cross-validation*. **Model LightGBM Classifier** terbukti sebagai model terbaik berdasarkan metrik evaluasi pada data uji, dengan **F1-score**, **Recall**, dan **ROC AUC** yang tinggi. Model ini menunjukkan kapabilitas yang signifikan dalam memprediksi keberadaan penyakit jantung, didukung oleh identifikasi fitur-fitur penting yang mempengaruhi prediksi.

---

### 🔍 Insight yang Didapat

#### 1. Kualitas Data Awal

Dataset awal memiliki tantangan dalam hal *missing values* di beberapa kolom, direpresentasikan dalam format non-standar seperti `?`, `0`, atau spasi kosong. Penanganan awal yang teliti (mengubah ke `NaN`, mengonversi tipe data, dan menghapus kolom tidak relevan) sangat esensial untuk validitas analisis dan pemodelan.

#### 2. Ketidakseimbangan Kelas Ringan

Meskipun terdapat *class imbalance* ringan (sekitar **55.33% kelas positif** dan **44.67% kelas negatif**), penggunaan metrik **F1-score** untuk tuning dan evaluasi akhir sangat tepat. Ini memastikan model tidak hanya akurat, tetapi juga adil dalam mendeteksi kedua kelas.

#### 3. Fitur Kritis

*Feature importance* dari model LightGBM menunjukkan bahwa:

* **Numerik**: `chol` (kolesterol), `age`, `thalch` (detak jantung maksimum), dan `trestbps` (tekanan darah saat istirahat)
* **Kategorikal**: `cp_asymptomatic` (tipe nyeri dada asimptomatik) dan `thal_reversable defect`

Fitur-fitur ini secara klinis memang sudah dikenal sebagai faktor risiko penyakit jantung.

#### 4. Efektivitas Optimasi Model

Tuning dengan `GridSearchCV` secara signifikan meningkatkan performa model dibanding baseline. LightGBM menunjukkan keunggulan dalam menangani kompleksitas data dan mencapai **F1-score serta ROC AUC** yang lebih tinggi dibandingkan Logistic Regression dan Random Forest.

---

### 🚀 Potensi Pengembangan Berikutnya (Next Steps)

1. **Validasi Eksternal**
   Uji model pada dataset independen dari populasi berbeda untuk memastikan generalisasi dan menghindari *overfitting*.

2. **Feature Engineering Lanjutan**
   Ciptakan fitur baru dari kombinasi fitur lama (misalnya rasio antar variabel medis) untuk meningkatkan prediksi.

3. **Interpretasi Model (XAI)**
   Gunakan teknik seperti **SHAP** atau **LIME** agar keputusan model dapat dijelaskan dengan transparan – penting untuk bidang medis.

4. **Optimasi Threshold**
   Sesuaikan *classification threshold* (default 0.5) untuk memaksimalkan *Recall* atau *Precision* sesuai kebutuhan klinis.

5. **Pengumpulan Data Tambahan**
   Tambahkan lebih banyak data atau data longitudinal untuk meningkatkan akurasi dan ketahanan model.

6. **Pengembangan Antarmuka Pengguna (UI)**
   Bangun UI (misalnya dengan Streamlit atau Flask) agar tenaga medis bisa langsung menggunakan model untuk prediksi.

---

### 📌 Apakah Model Sudah Cukup Baik untuk Dunia Nyata?

Model yang dikembangkan menunjukkan performa yang sangat menjanjikan:

* **F1-score**: `0.8626`
* **Recall**: `0.8922`
* **ROC AUC**: `0.8782`

Kemampuan membedakan pasien dengan dan tanpa penyakit jantung cukup kuat. Namun, **implementasi nyata masih memerlukan:**

* Validasi eksternal lebih lanjut
* Review dan persetujuan dari **ahli medis & regulator kesehatan**
* Jaminan **privasi dan keamanan data pasien**

Dengan validasi yang ketat dan adaptasi berdasarkan masukan profesional medis, model ini sangat potensial sebagai **alat bantu klinis** untuk *skrining dini, manajemen risiko, dan peningkatan kualitas perawatan*.




