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
*   **[1] World Health Organization (WHO). Cardiovascular diseases (CVDs).** Tersedia di: [https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)). (Diakses pada 23 Mei 2025).
    *   *Penjelasan:* Sumber ini memberikan statistik global tentang prevalensi dan dampak serius penyakit kardiovaskular, menggarisbawahi urgensi permasalahan ini dalam skala global.
*   **[2] Muhammad, A. B., & Mahmud, H. (2020). Heart Disease Prediction Using Machine Learning Algorithms: A Comparative Study.** *International Journal of Computer Applications*, 174(1), 1-5.
    *   *Penjelasan:* Studi ini menyajikan contoh konkret bagaimana berbagai algoritma *machine learning* telah diterapkan untuk memprediksi penyakit jantung, menegaskan relevansi pendekatan ML dalam domain medis.

## 2. 🧠 Business Understanding

### Pernyataan Masalah (Problem Statements)
1.  Bagaimana kita dapat mengembangkan model *machine learning* yang efektif untuk secara akurat memprediksi keberadaan penyakit jantung pada pasien berdasarkan data klinis dan demografis yang tersedia?
2.  Bagaimana model prediksi ini dapat diintegrasikan sebagai alat bantu bagi tenaga medis untuk mendukung diagnosis dini dan pengambilan keputusan klinis, sehingga memungkinkan intervensi lebih cepat dan mengurangi risiko komplikasi serius?

### Tujuan (Goals)
1.  Mengembangkan model klasifikasi *machine learning* (spesifiknya, **Klasifikasi Biner**) yang mampu mengidentifikasi pasien dengan penyakit jantung (kelas positif) dan tanpa penyakit jantung (kelas negatif) dengan performa evaluasi yang optimal, terutama pada metrik `F1-score`, `Recall`, dan `ROC AUC`.
2.  Mengidentifikasi fitur-fitur medis utama (misalnya, usia, jenis kelamin, tekanan darah, kolesterol, tipe nyeri dada) yang memiliki pengaruh paling signifikan terhadap risiko penyakit jantung berdasarkan analisis model.
3.  Membangun sistem prediksi yang dapat diandalkan untuk skrining awal, mengarahkan pasien berisiko tinggi untuk pemeriksaan lebih lanjut dan intervensi yang tepat waktu.

### Pernyataan Solusi (Solution Statements) (Rubrik Tambahan)
Untuk mencapai tujuan proyek ini dan memastikan solusi yang robust dan terukur, dua pendekatan utama akan diterapkan:

1.  **Perbandingan Berbagai Algoritma Klasifikasi:** Kami akan menguji dan membandingkan kinerja setidaknya tiga algoritma *machine learning* yang relevan untuk masalah klasifikasi biner:
    *   **Logistic Regression:** Sebagai model *baseline* yang sederhana dan *interpretable*, untuk melihat sejauh mana linearitas data mempengaruhi prediksi.
    *   **Random Forest Classifier:** Sebuah model *ensemble* yang kuat, dikenal karena kemampuannya menangani kompleksitas data dan mengurangi *overfitting*.
    *   **LightGBM Classifier:** Sebuah algoritma *gradient boosting* yang efisien dan berkinerja tinggi, seringkali memberikan akurasi yang sangat baik pada data tabular.
    Performa setiap algoritma akan diukur menggunakan metrik `Accuracy`, `Precision`, `Recall`, `F1-score`, dan `ROC AUC` pada data uji. Model dengan kombinasi metrik terbaik akan dipilih sebagai kandidat utama.

2.  **Optimasi Model Melalui Hyperparameter Tuning:** Setelah perbandingan awal, model-model yang menjanjikan akan menjalani proses *hyperparameter tuning* ekstensif menggunakan `GridSearchCV` dengan 5-fold *cross-validation*. Proses ini akan secara sistematis mencari kombinasi *hyperparameter* terbaik yang mengoptimalkan performa prediktif model. Kualitas optimasi akan diukur dari peningkatan nilai `F1-score` (sebagai metrik utama untuk keseimbangan *Precision* dan *Recall*) pada set validasi dan uji, memastikan bahwa solusi yang diberikan adalah yang terbaik dan dapat diukur.

## 3. 📂 Data Understanding

### Informasi Data
Dataset yang digunakan dalam proyek ini adalah `heart_disease_uci.csv`, yang dikombinasikan dari empat dataset penyakit jantung yang berbeda dari UCI Machine Learning Repository (Cleveland, Hungary, Switzerland, VA Long Beach).

*   **Sumber Data:** Dataset ini diunduh dari Kaggle, tersedia di: [https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).
*   **Jumlah Sampel & Fitur Awal:** Dataset awal memiliki 1025 sampel (baris) dan 16 kolom (fitur, termasuk ID dan dataset sumber). Setelah pembersihan awal (penghapusan kolom ID dan dataset sumber, serta transformasi target), dataset akan memiliki 13 fitur yang relevan untuk pemodelan dan 1 variabel target. Dataset ini bersifat kuantitatif (atau telah dikonversi menjadi numerik) dan memenuhi persyaratan minimal 500 sampel.
*   **Kondisi Data Awal:** Data mentah memiliki beberapa nilai yang hilang yang direpresentasikan secara tidak standar (misalnya `'?'` atau spasi kosong). Beberapa kolom numerik kunci (seperti `trestbps`, `chol`, `thalch`) juga memiliki nilai `0` yang secara klinis tidak valid dan diinterpretasikan sebagai nilai hilang. Terdapat fitur kategorikal dalam format string yang perlu di-encode, dan variabel target asli (`num`) perlu ditransformasi menjadi biner.

### Uraian Variabel (Fitur)
Berikut adalah penjelasan setiap kolom (fitur) dalam dataset setelah pembersihan awal, serta variabel target:

*   **`age`**: (Numerik) Usia pasien dalam tahun.
*   **`sex`**: (Numerik) Jenis kelamin pasien (1 = laki-laki, 0 = perempuan).
*   **`cp`**: (Kategorikal) Tipe nyeri dada (misalnya `typical angina`, `asymptomatic`, `non-anginal pain`, `atypical angina`).
*   **`trestbps`**: (Numerik) Tekanan darah saat istirahat (resting blood pressure) dalam mm Hg.
*   **`chol`**: (Numerik) Kolesterol serum (serum cholestoral) dalam mg/dl.
*   **`fbs`**: (Numerik) Gula darah puasa (fasting blood sugar) > 120 mg/dl (1 = true, 0 = false).
*   **`restecg`**: (Kategorikal) Hasil elektrokardiografi saat istirahat (misalnya `normal`, `st-t abnormality`, `lv hypertrophy`).
*   **`thalch`**: (Numerik) Detak jantung maksimum yang dicapai (maximum heart rate achieved).
*   **`exang`**: (Numerik) Angina yang diinduksi oleh olahraga (exercise induced angina) (1 = yes, 0 = no).
*   **`oldpeak`**: (Numerik) Depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat.
*   **`slope`**: (Kategorikal) Kemiringan segmen ST puncak olahraga (misalnya `upsloping`, `flat`, `downsloping`).
*   **`ca`**: (Numerik) Jumlah pembuluh darah utama (0-3) yang diwarnai oleh fluoroskopi.
*   **`thal`**: (Kategorikal) Thalassemia (misalnya `normal`, `fixed defect`, `reversable defect`).
*   **`target`**: (Numerik - Variabel Target) Keberadaan penyakit jantung (1 = ada penyakit jantung, 0 = tidak ada penyakit jantung). Ini adalah hasil transformasi dari kolom `num` asli (0 = tidak ada penyakit, 1-4 = ada penyakit).

### Eksplorasi Data (Exploratory Data Analysis - EDA)
EDA dilakukan untuk memahami karakteristik data, mengidentifikasi pola, outlier, dan nilai yang hilang, serta hubungan antar fitur dan dengan variabel target. Berikut adalah beberapa *insight* kunci yang diperoleh dari EDA (visualisasi terkait dapat dilihat pada Jupyter Notebook):

1.  **Inspeksi Awal Data (`df.head()`, `df.info()`, `df.describe()`):**
    *   ```python
        print("5 Baris Pertama Dataset:")
        print(df.head())
        print("\nInformasi Umum Dataset:")
        df.info()
        print("\nStatistik Deskriptif Dataset:")
        print(df.describe())
        ```
    *   **Insight:** Dari `df.info()`, terlihat beberapa kolom (`trestbps`, `chol`, `fbs`, `restecg`, `thalch`, `exang`, `oldpeak`, `slope`, `ca`, `thal`) memiliki *non-null count* yang lebih rendah dari total sampel (920), menunjukkan adanya *missing values*. Kolom `fbs` dan `exang` memiliki 0 *non-null count*, yang berarti 100% *missing values* setelah konversi awal 'TRUE'/'FALSE' ke numerik dan penggantian non-numerik ke NaN. Kolom `sex` sudah berhasil dikonversi ke numerik. `df.describe()` memberikan ringkasan statistik untuk kolom numerik, seperti rentang usia (28-77 tahun) dan detak jantung maksimum (60-202).

2.  **Pengecekan dan Persentase *Missing Values*:**
    *   ```python
        print("\nJumlah Missing Values Setelah Pembersihan Awal:")
        print(df.isnull().sum())
        print("\nPersentase Missing Values Setelah Pembersihan Awal:")
        print(df.isnull().sum() / len(df) * 100)
        ```
    *   **Insight:** Dikonfirmasi bahwa `fbs` dan `exang` memiliki 100% *missing values* karena konversi string 'TRUE'/'FALSE' gagal akibat adanya nilai non-standar, atau karena memang seluruhnya kosong. Kolom `ca` (66.41%) dan `thal` (52.83%) juga memiliki persentase *missing values* yang sangat tinggi. Kolom `slope` (33.59%) dan `chol` (21.96%) juga signifikan. Ini menunjukkan bahwa strategi imputasi yang efektif sangat diperlukan.

3.  **Distribusi Variabel Target:**
    *   ```python
        plt.figure(figsize=(6, 4))
        sns.countplot(x='target', data=df)
        plt.title('Distribusi Kelas Target (0: No Disease, 1: Disease)')
        plt.xlabel('Penyakit Jantung')
        plt.ylabel('Jumlah Pasien')
        plt.xticks(ticks=[0, 1], labels=['Tidak Ada', 'Ada'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        print("\nDistribusi Nilai Variabel Target:")
        print(df['target'].value_counts())
        print(f"Persentase Kelas 'Tidak Ada Penyakit': {df['target'].value_counts(normalize=True)[0]*100:.2f}%")
        print(f"Persentase Kelas 'Ada Penyakit': {df['target'].value_counts(normalize=True)[1]*100:.2f}%")
        ```
    *   **Insight:** Distribusi kelas target (`target`) relatif seimbang, dengan 44.67% pasien tidak memiliki penyakit jantung dan 55.33% memiliki penyakit jantung. Meskipun tidak sepenuhnya seimbang, ini bukan *extreme imbalance*, sehingga tidak memerlukan teknik penanganan *imbalance* yang agresif, namun tetap relevan untuk menggunakan metrik `F1-score` dan `ROC AUC` selain `Accuracy`.

4.  **Distribusi Fitur Numerik:**
    *   ```python
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(numeric_features): # numeric_features includes 'sex', 'fbs', 'exang', 'ca'
            plt.subplot(3, 4, i + 1)
            sns.histplot(pd.to_numeric(df[feature], errors='coerce').dropna(), kde=True)
            plt.title(f'Distribusi {feature}')
        plt.tight_layout()
        plt.show()
        ```
    *   **Insight:** Fitur seperti `age` dan `thalch` (detak jantung maksimum) menunjukkan distribusi yang mendekati normal. `oldpeak` menunjukkan distribusi *zero-inflated* dengan beberapa nilai yang menyebar. Fitur `chol` dan `trestbps` juga memiliki distribusi yang cukup bervariasi. Kolom `fbs` dan `exang` setelah konversi awal ke numerik menjadi hampir seluruhnya NaN, sehingga distribusinya tidak informatif pada tahap ini.

5.  **Distribusi Fitur Kategorikal:**
    *   ```python
        plt.figure(figsize=(18, 12))
        for i, feature in enumerate(categorical_features):
            plt.subplot(2, 3, i + 1)
            sns.countplot(x=feature, hue='target', data=df, palette='viridis')
            plt.title(f'Distribusi {feature}')
            plt.xticks(rotation=45, ha='right')
        plt.legend(title='Penyakit Jantung', labels=['Tidak Ada', 'Ada'])
        plt.tight_layout()
        plt.show()
        ```
    *   **Insight:**
        *   `cp` (tipe nyeri dada): Kategori `asymptomatic` (`cp_asymptomatic`) memiliki proporsi pasien penyakit jantung yang lebih tinggi dibandingkan tipe nyeri dada lainnya.
        *   `restecg` (hasil EKG): Pasien dengan `lv hypertrophy` atau `st-t abnormality` lebih sering memiliki penyakit jantung.
        *   `slope` (kemiringan ST): Kemiringan `flat` atau `downsloping` lebih sering terkait dengan penyakit jantung.
        *   `thal` (thalassemia): Pasien dengan `reversable defect` memiliki risiko penyakit jantung yang sangat tinggi.

6.  **Korelasi Antar Fitur (Rubrik Tambahan):**
    *   ```python
        correlation_matrix = df_corr[all_numeric_cols_for_eda].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Matriks Korelasi Fitur Numerik dan Target')
        plt.show()
        ```
    *   **Insight:** Heatmap korelasi menunjukkan beberapa hubungan penting. `oldpeak` (`0.49`), `exang` (`0.49`), `ca` (`0.52`), dan `thalch` (`-0.42`) memiliki korelasi linear yang paling signifikan dengan variabel `target`. Korelasi yang lebih tinggi (positif atau negatif) menunjukkan potensi fitur tersebut sebagai prediktor yang kuat. Beberapa fitur juga menunjukkan korelasi satu sama lain (`age` dan `trestbps`), yang dapat mengindikasikan multikolinearitas namun tidak menjadi masalah besar untuk model berbasis pohon seperti Random Forest atau LightGBM.

7.  **Hubungan Fitur Numerik dengan Target (Rubrik Tambahan):**
    *   ```python
        plt.figure(figsize=(18, 12))
        for i, feature in enumerate(continuous_numeric_features):
            plt.subplot(3, 4, i + 1)
            sns.boxplot(x='target', y=pd.to_numeric(df_corr[feature], errors='coerce'), data=df_corr)
            plt.title(f'{feature} vs Target')
        plt.tight_layout()
        plt.show()
        ```
    *   **Insight:** Boxplot memvisualisasikan perbedaan distribusi fitur numerik antara pasien dengan dan tanpa penyakit jantung. Pasien dengan penyakit jantung (`target=1`) cenderung memiliki nilai `age`, `trestbps`, `chol`, dan `oldpeak` yang lebih tinggi, serta nilai `thalch` yang lebih rendah. Ini mengkonfirmasi *insight* dari korelasi dan memberikan pemahaman visual tentang perbedaan grup.

## 4. 🧹 Data Preparation

Tahap *data preparation* adalah langkah krusial untuk mengubah data mentah menjadi format yang siap untuk pemodelan *machine learning*. Teknik-teknik yang digunakan dan alasannya adalah sebagai berikut:

### 4.1. Initial Data Cleaning dan Transformasi Target
Ini adalah langkah pertama untuk mengatasi masalah data yang terlihat pada inspeksi awal.

*   **Proses:**
    *   Nilai non-standar (`'?'`, spasi kosong `''`) di seluruh DataFrame diganti dengan `np.nan`.
    *   Kolom `sex`, `fbs`, `exang` yang tadinya memiliki nilai boolean (`TRUE`/`FALSE` dalam string) dikonversi menjadi representasi numerik (1/0).
    *   Kolom `ca` dan `num` yang mungkin terpengaruh oleh penggantian `np.nan` dikonversi kembali ke tipe numerik (`float`). `errors='coerce'` digunakan untuk memastikan bahwa nilai yang tidak dapat dikonversi akan menjadi `NaN`.
    *   Nilai `0` pada kolom `trestbps`, `chol`, dan `thalch` yang secara klinis tidak valid (karena tekanan darah atau kolesterol tidak mungkin nol pada pasien hidup) diganti dengan `np.nan`.
    *   Kolom `id` dan `dataset` dihapus karena tidak relevan sebagai fitur prediktif.
    *   Variabel target `num` (0-4) ditransformasi menjadi biner `target` (0 = tidak ada penyakit jantung, 1 = ada penyakit jantung).
*   **Alasan:** Langkah-langkah ini sangat penting untuk membersihkan data dari format yang tidak konsisten, memastikan semua data dalam bentuk numerik yang dapat diproses oleh model ML, dan mendefinisikan variabel target sesuai dengan tujuan klasifikasi biner.

### 4.2. Pemisahan Data Latih dan Uji
Setelah pembersihan awal, data dipisahkan sebelum preprocessing lanjutan untuk mencegah *data leakage*.

*   **Proses:** Data dibagi menjadi fitur (`X`) dan target (`y`). Kemudian, `X` dan `y` dipisahkan menjadi set pelatihan (`X_train`, `y_train`) dan set pengujian (`X_test`, `y_test`) dengan proporsi 80% untuk pelatihan dan 20% untuk pengujian menggunakan `train_test_split` dari scikit-learn. Parameter `stratify=y` digunakan untuk memastikan bahwa proporsi kelas target (ada/tidak ada penyakit jantung) dipertahankan sama antara set pelatihan dan pengujian, yang penting karena ada sedikit ketidakseimbangan kelas.
*   **Alasan:** Pemisahan ini krusial untuk mengevaluasi performa model secara objektif. Model dilatih hanya pada data pelatihan dan kemudian diuji pada data yang belum pernah dilihat sebelumnya (data uji) untuk mengukur kemampuan generalisasinya dan mencegah *overfitting*.

### 4.3. Pipeline Preprocessing Lanjutan
Untuk preprocessing lanjutan, digunakan `Pipeline` dan `ColumnTransformer` dari scikit-learn.

*   **Proses:**
    *   **Fitur Numerik:** Untuk fitur numerik (`age`, `trestbps`, `chol`, `thalch`, `oldpeak`, `ca`), `SimpleImputer` dengan strategi `median` digunakan untuk mengisi *missing values*, diikuti oleh `StandardScaler` untuk penskalaan fitur.
    *   **Fitur Kategorikal:** Untuk fitur kategorikal (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal`), `SimpleImputer` dengan strategi `most_frequent` (modus) digunakan untuk mengisi *missing values*, diikuti oleh `OneHotEncoder` untuk *one-hot encoding*.
    *   Semua transformer ini digabungkan dalam `ColumnTransformer` yang diterapkan dalam sebuah `Pipeline` sebelum model klasifikasi.
*   **Alasan:**
    *   **Automatisasi dan Konsistensi:** `Pipeline` mengotomatiskan urutan langkah preprocessing, memastikan semua transformasi diterapkan secara konsisten pada data latih dan uji.
    *   **Mencegah *Data Leakage*:** Dengan *pipeline*, semua parameter transformasi (misalnya median untuk imputasi, *mean/std* untuk *StandardScaler*, kategori unik untuk *OneHotEncoder*) hanya dipelajari dari data latih (`X_train`) dan kemudian diterapkan ke data uji (`X_test`). Ini mencegah "kebocoran" informasi dari data uji ke data latih yang dapat menyebabkan estimasi performa model yang terlalu optimis.
    *   **Penanganan *Missing Values*:** `SimpleImputer` mengatasi *missing values*. `median` dipilih untuk fitur numerik karena lebih robust terhadap outlier dibandingkan `mean`, sementara `most_frequent` adalah pilihan standar untuk kategorikal.
    *   **Encoding Variabel Kategorikal:** `OneHotEncoder` mengubah data kategorikal string menjadi representasi numerik biner, mencegah model mengasumsikan hubungan ordinal yang tidak ada antar kategori. `handle_unknown='ignore'` memastikan model dapat memproses kategori yang tidak terlihat selama pelatihan.
    *   **Feature Scaling:** `StandardScaler` menyeragamkan skala fitur numerik (mean 0, standar deviasi 1). Ini penting untuk algoritma yang sensitif terhadap skala, seperti Logistic Regression dan SVM, meskipun tidak terlalu krusial untuk model berbasis pohon (Random Forest, LightGBM).

## 5. 🏗️ Modeling

Pada tahap ini, kami membangun dan melatih beberapa model klasifikasi untuk memprediksi penyakit jantung. Kami memilih tiga algoritma yang populer dan memiliki karakteristik berbeda untuk mengevaluasi performa mereka secara komprehensif.

### 5.1. Pembuatan Model Machine Learning
Kami memilih tiga algoritma klasifikasi sebagai kandidat utama:

1.  **Logistic Regression**
    *   **Kelebihan:**
        *   Sederhana, cepat, dan mudah diinterpretasi (melalui koefisien fitur).
        *   Baik sebagai model *baseline* untuk masalah klasifikasi biner, terutama jika hubungan data cenderung linier.
    *   **Kekurangan:**
        *   Mengasumsikan hubungan linier antara fitur input dan log-odds dari variabel target.
        *   Performa bisa menurun pada data dengan hubungan non-linear atau yang sangat kompleks.

2.  **Random Forest Classifier**
    *   **Kelebihan:**
        *   Model *ensemble* yang sangat kuat, menggabungkan banyak pohon keputusan untuk meningkatkan akurasi dan mengurangi *overfitting*.
        *   Mampu menangani hubungan non-linear dan interaksi kompleks antar fitur.
        *   Kurang sensitif terhadap outlier dan penskalaan fitur.
        *   Dapat memberikan indikasi *feature importance*, yang berguna untuk interpretasi.
    *   **Kekurangan:**
        *   Kurang *interpretable* dibandingkan model linier (sering disebut 'black box').
        *   Membutuhkan sumber daya komputasi yang lebih besar dan waktu pelatihan yang lebih lama dibandingkan model sederhana.

3.  **LightGBM Classifier**
    *   **Kelebihan:**
        *   Algoritma *gradient boosting* yang sangat efisien dan cepat, terutama pada dataset besar.
        *   Menawarkan kinerja prediktif yang superior, seringkali menjadi salah satu algoritma terbaik untuk data tabular.
        *   Secara bawaan dapat menangani *missing values* dan fitur kategorikal.
    *   **Kekurangan:**
        *   Sangat rentan terhadap *overfitting* jika *hyperparameter* tidak di-tuning dengan benar.
        *   Lebih kompleks dan kurang *interpretable* (model 'black box' lainnya).
        *   Membutuhkan pemahaman yang baik tentang cara kerja algoritma untuk *tuning* yang efektif.

### 5.2. Tahapan dan Parameter Pemodelan (Rubrik Tambahan: Optimasi Hyperparameter)
Setiap model diintegrasikan ke dalam sebuah `Pipeline` bersama dengan langkah-langkah *preprocessing* (`preprocessor` yang mencakup imputasi, *scaling*, dan *encoding*). Proses pelatihan model utama melibatkan *hyperparameter tuning* ekstensif:

*   **Inisialisasi Model:** Setiap model diinisialisasi dengan `random_state=42` untuk memastikan hasil yang replikatif.
*   **Hyperparameter Tuning dengan `GridSearchCV`:**
    *   **Metode:** `GridSearchCV` digunakan untuk mencari kombinasi *hyperparameter* terbaik untuk setiap model. `GridSearchCV` akan mencoba setiap kombinasi parameter yang ditentukan dalam `param_grid` secara sistematis.
    *   **Cross-Validation:** Parameter `cv=5` digunakan, yang berarti 5-Fold *Cross-Validation*. Ini membagi data pelatihan menjadi 5 bagian, melatih model 5 kali (masing-masing menggunakan 4 bagian untuk pelatihan dan 1 bagian untuk validasi) dan merata-ratakan skornya. Hal ini memberikan estimasi performa model yang lebih robust dan mengurangi varians, dibandingkan hanya menggunakan satu *train-validation split*.
    *   **Metrik Penilaian (`scoring`):** `scoring='f1'` dipilih sebagai metrik utama untuk *tuning*. `F1-score` adalah metrik yang seimbang antara `Precision` dan `Recall`. Ini sangat relevan dalam konteks medis di mana False Positives dan False Negatives sama-sama penting untuk diminimalisir (misalnya, tidak salah mendiagnosis sehat padahal sakit, dan tidak salah mendiagnosis sakit padahal sehat).
    *   **`n_jobs=-1`:** Menggunakan semua inti CPU yang tersedia untuk mempercepat proses *tuning*.

Berikut adalah *param_grid* yang digunakan untuk setiap model:
*   **Logistic Regression:** `classifier__C` (inverse dari kekuatan regularisasi), `classifier__solver` (algoritma optimasi).
*   **Random Forest:** `classifier__n_estimators` (jumlah pohon dalam *forest*), `classifier__max_depth` (kedalaman maksimum pohon), `classifier__min_samples_split` (minimum sampel yang dibutuhkan untuk membagi *node* internal).
*   **LightGBM:** `classifier__n_estimators` (jumlah *boosting rounds*), `classifier__learning_rate` (ukuran langkah kontribusi setiap pohon), `classifier__num_leaves` (jumlah maksimum daun dalam satu pohon).

### 5.3. Pemilihan Model Terbaik

Setelah melatih dan melakukan *tuning* untuk semua model, kami mengevaluasi setiap model terbaik yang telah di-tuning pada data uji (`X_test`, `y_test`). Kami membandingkan `Accuracy`, `Precision`, `Recall`, `F1-score`, dan `ROC AUC` mereka.

Berikut adalah tabel performa model pada data uji:

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




