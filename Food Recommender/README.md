# Recommendation System Project Report - Anggun Sulis Setyawan ✨

---
## List of Contents
  - [Background](#Background)
  - [Business Understanding](#Business-Understanding)
    - [Problem Statments](#Problem-Statments)
    - [Goals](#Goals)
    - [Solution Statements](#Solution-Statements)
  - [Data Understanding](#Data-Understanding)
    - [Dataset](#Dataset)
    - [Overview](#Overview)
    - [Column Description](#Column-Description)
    - [Explore](#Explore)
  - [Data Preparation](#Data-Preparation#)
    - [Features Engineering](#Feature-Engineering)
    - [Data Normalization](#Data-Normalization)
    - [Data Split](#Data-Split)
  - [Modeling](#Modeling)
    - [Model Design](#Model-Design)
    - [Training](#Training)
  - [Evaluation](#Evaluation)
    - [Metrics Description](#Metrics-Description)
    - [Model Comparison](#Model-Comparison)
  - [Inference](#Inference)
  - [Development Opportunity](#Development-Opportunity)
  - [References](#References)

---
## Background
  Indonesia memiliki cita-cita besar untuk menjadi negara maju. Oleh karena itu, peningkatan kualitas Sumber Daya Manusia (SDM) menjadi pilar penting yang harus diperbaiki, terutama dalam hal peningkatan kesehatan, pemenuhan gizi, dan pencegahan stunting (Kemenko PMK RI, 2024). Pemerintah Indonesia telah merencanakan untuk melaksanakan program makan bergizi gratis sesuai arahan Presiden RI Prabowo Subianto. Pemerintah berharap program tersebut dapat menurunkan jumlah kasus stunting bahkan mencegah kasus stunting baru di masa depan. Fokus awal dari program ini adalah anak-anak sekolah dan kelompok rentan lainnya. Anak sekolah yang dimaksud antara lain pelajar PAUD, SD, SMP, dan SMA (detik.com, 2024). 

  Anak sekolah menjadi sasaran karena status gizi dan  stunting sangat mempengaruhi kecerdasan anak. Status gizi memberikan kontribusi terhadap kesulitan belajar sebesar 32,83%. Anak yang kekurangan nutrisi cenderung memiliki kelemahan pada sistem saraf hingga dapat menyebabkan kelainan motorik dan kognitif (Dewi, dkk, 2021). Sebagai calon penerus bangsa, maka pertumbuhan dan perkembangan anak sekolah perlu diperhatikan dengan baik agar menghasilkan potensi sumber daya manusia dengan kualitas maksimal. Hal ini dapat dicapai dengan salah satu cara yaitu pemenuhan kebutuhan nutrisi harian melalui program makanan bergizi gratis.

  Angka Kecukupan Gizi (AKG) menurut Kementerian Kesehatan Republik Indonesia adalah kecukupan rata-rata gizi harian yang dianjurkan untuk sekelompok orang setiap harinya. Kebutuhan gizi ideal anak yang harus terpenuhi dalam sehari terbagi menjadi dua kelompok, yaitu zat gizi makro dan mikro. Zat gizi makro adalah semua jenis zat gizi yang dibutuhkan anak dalam jumlah banyak, seperti energi (kalori), protein, lemak, dan karbohidrat. Sementara zat gizi mikro adalah nutrisi yang dibutuhkan dalam jumlah sedikit, seperti vitamin dan mineral (Damar Upahita, 2021). Penentuan nilai gizi disesuaikan dengan jenis kelamin, kelompok umur, tinggi badan, berat badan, serta aktivitas fisik (Kemenkes RI, 2019).

  Seluruh program makanan bergizi gratis harus melibatkan kolaborasi pemangku kepentingan terkait untuk dikonvergensikan sehingga bisa komprehensif dan terintegrasi. Salah satunya adalah penyediaan makanan yang efektif dan efisien. Salah satu perusahaan penyedia layanan catering dan bento, Olagizi ingin mengambil peran penting dalam penyediaan paket makanan bergizi bagi siswa SMP dan SMA. Olagizi ingin memberikan layanan dengan optimal. Oleh karena itu, Olagizi ingin membuat sebuah sistem yang dapat memberikan rekomendasi tentang makanan bergizi yang dipersonalisasi sesuai kebutuhan gizi dan selera para siswa. Di sisi lain, Olagizi juga ingin rekomendasi tersebut memberikan pilihan makanan yang dapat dimasak dalam waktu yang tidak terlalu lama agar makanan dapat disiapkan tepat pada waktu, khususnya makanan untuk sesi sarapan. Untuk pengembangan tahap awal, Olagizi ingin membuat model sistem rekomendasi makanan berdasarkan kemiripan jumlah kalori yang terkandung serta berdasarkan hasil ulasan rating makanan.

---
## Business Understanding
### Problem Statements
  1. Bagaimana sistem rekomendasi dapat memberikan pilihan makanan dengan kandungan jumlah kalori yang mirip?
  2. Bagaimana sistem rekomendasi dapat memberikan berbagai pilihan makanan yang mungkin disukai oleh target pelanggan?
### Goals
  1. Menghasilkan 10 rekomendasi makanan yang memiliki nilai kalori yang mirip dan dapat dimasak dalam waktu kurang dari 2 jam.
  2. Menghasilkan 10 rekomendasi makanan yang mungkin disukai oleh target pelanggan dan dapat dimasak dalam waktu kurang dari 2 jam.
### Solution Statements
  1. Menerapkan pendekatan _content-based filtering_ menggunakan algoritma _cosine similarity_.
  2. Menerapkan pendekatan _collaborative filtering_ menggunakan algoritma _deep learning_.

---
## Data Understanding
### Overview
  Dataset ini berasal dari platform Kaggle salah satu pengembang sistem rekomendasi makanan diet dengan nama akun @SOUMEDHIK yang dapat diakses pada link di bawah. Ukuran dataset begitu besar sehingga dataset yang digunakan hanya sebagian sampel saja. Pada proyek ini, hanya akan menggunakan 2 file dataset dalam format csv, yaitu recipes data dan interactions data.
_Download raw dataset_:
[Diet Recommender Dataset](https://www.kaggle.com/code/soumedhik/diet-recommender/input)

### Recipes Data
`recipes_sample_df` memiliki total 2316 baris data dan 12 kolom.

| No | Kolom | Tipe Data | Deskripsi |
|----|-------|-----------|-----------|
| 1 | Name | `object` | Nama resep. |
| 2 | id | `integer` | ID resep. |
| 3 | minutes | `integer` | Waktu yang diperlukan untuk memasak (dalam menit). |
| 4 | contribution_id | `integer` | ID pengguna yang berkontribusi mengunggah resep. |
| 5 | submitted | `object` | Tanggal resep diunggah. |
| 6 | tags | `object` | Kategori atau tag resep. |
| 7 | nutrition | `object`| Informasi nutrisi (kalori, lemak/Total Fat (g), gula (g), sodium (mg), protein (g), lemak jenuh/saturated fat (g), dan karbohidrat (g)). |
| 8 | n_steps | `integer` | Jumlah langkah yang diperlukan untuk memasak. |
| 9 | description | `object` | Deskripsi singkat mengenai resep. |
| 10 | ingredients | `object` | Daftar bahan-bahan yang digunakan dalam resep. |
| 11 | n_ingredients | `integer` | Jumlah bahan yang digunakan dalam resep. |

### Interactions Data
`interactions_sample_df` memiliki total 11324 baris data dan 5 kolom.

| No | Kolom | Tipe Data | Deskripsi |
|----|-------|-----------|-----------|
| 1 | user_id | `integer` | ID pengguna yang memberikan ulasan. |
| 2 | recipe_id | `integer` | ID resep yang diberi ulasan. |
| 3 | date | `object` | Tanggal ulasan diberikan. |
| 4 | rating | `integer` | Penilaian yang diberikan (dalam skala tertentu). |
| 5 | review | `object` | Isi ulasan yang diberikan pengguna. |

### Explore Data
  Eksplorasi data dilakukan untuk mengetahui lebih banyak terkait karakteristik dataset yang akan digunakan, mulai dari kelengkapan data, format data, dan statistik data. Berdasarkan hasil eksplorasi data, terdapat beberapa temuan antara lain:
* Pada dataset `recipes_sample_df` ditemukan 52 _missing value_ untuk kolom _description_ dan terdapat format data yang belum sesuai pada kolom _submitted_ dan _nutrition_. Menurut hasil deskripsi statistik rata-rata makanan pada data membutuhkan waktu memasak sekitar 130 menit atau sekitar 9 tahapan.
* Pada dataset `interactions_sample_df` ditemukan 1 _missing value_ untuk kolom _review_ dan terdapat format data yang belum sesuai pada kolom _date_. Diketahui rata-rata nilai ulasan atau _rating_ yang diberikan orang-orang adalah 4,4.


### Univariate Analysis
#### recipes_sample_df
![minutes](https://github.com/user-attachments/assets/743048bc-8aab-455f-81b9-cab385b348b6)
Diketahui distribusi data kolom _minutes_ pada `recipes_sample_df` sangat _skewed_ karena data memiliki _outlier_ yang cukup banyak dan rentang nilai yang ekstrem. 

![n_steps](https://github.com/user-attachments/assets/ab2a868b-9885-40cc-97c0-380132ce11f0)
Sementara itu, data kolom _n_steps_ juga terdistribusi _skewed_, tetapi memiliki lebih sedikit outlier dengan rentang yang tidak terlalu ekstrem.

#### interactions_sample_df
![rating](https://github.com/user-attachments/assets/03c5e40b-31d7-4834-b77a-c827bc9b0ce5)
Data ulasan yang ada menunjukkan dominasi rating yang diberikan berkisar 3 - 5. Hal ini menandakan bahwa banyak pelanggan atau pengguna yang merasa puas.

### Multivariate Analysis


---
## Data Preparation

#### _Data Cleaning_
  Salah satu tahap terpenting dalam _Data Preparation_ yaitu _Data Cleaning_. Proses ini dilakukan untuk memastikan bahwa data yang akan digunakan untuk melatih model merupakan data yang bersih, rapi, dan berkualitas. Misalnya memastikan format data sudah tepat sesuai dengan representasi data, perlakuan terhadap data yang hilang (_missing value_) maupun pencilan data (_outlier_), dll. Dengan begitu, proses persiapan data setelahnya dapat dilakukan dengan lebih mudah.

#### _Filter Data 
##### recipes_sample_df
Berdasarkan _goals_ yang telah ditetapkan, Olagizi ingin sistem memberikan rekomendasi makanan yang dapat dimasak kurang dari 2 jam. Oleh karena itu, data sebaiknya disaring terlebih dahulu.

#### _Features Engineering_
1. Ekstrak data pada fitur _nutrition_ dalam `recipes_sample_df`:
   Diketahui data pada kolom _nutrition_ memiliki format `object` yang berisi sebuah `list`. Data tersebut mengandung informasi jumlah kalori, lemak, gula, sodium, protein, lemak jenuh, dan karbohidrat. Untuk memudahkan analisis, maka data harus diekstrak menjadi kolom-kolom tersendiri. Ini dilakukan agar hubungan antar variabel data dapat dianalisis dengan lebih mudah pada proses selanjutnya.

2. Sesuaikan data *ingredients* pada `recipes_sample_df`:
   Diketahui data pada kolom *ingredients* memiliki format `object` yang berisi `string` sehingga perlu dipisahkan untuk setiap frasa bahan agar dapat terdeteksi sebagai satu bahan baku dengan benar saat proses vektoriasi. Caranya adalah dengan split frasa-frasa tersebut dengan menggunakan koma.

#### _Content Based Filtering_
##### Features Selection
  Selanjutnya, pemilihan data-data yang relevan bagi sistem rekomendasi berbasis _Content Based Filtering_ dari `recipes_sample_df`. Fitur-fitur yang terpilih untuk proses analisis selanjutnya adalah sebagai berikut.
| Kolom Terpilih | Dataframe Asal |
|-----------|----------------|
| id | recipes_sample_df |
| name | recipes_sample_df | 
| ingredients | recipes_sample_df |
| calories | recipes_sample_df |
| minutes | recipes_sample_df | 
| steps | recipes_sample_df | 

##### Features Extraction


#### Collaborative Filtering
##### Features Selection
  Sementara itu, pemilihan data-data yang relevan bagi sistem rekomendasi berbasis _collaborative filtering_ dari `interactions_sample_df` juga `recipes_sample_df`. Fitur-fitur yang terpilih untuk proses analisis selanjutnya adalah sebagai berikut.
| Kolom Terpilih | Dataframe Asal |
|-----------|----------------|
| name | recipes_sample_df | 
| recipe_id | interactions_sample_df |
| user_id | interactions_sample_df | 
| rating | interactions_sample_df | 

##### Features Encoding
  Diketahui data *recipe_id* dan *user_id* memiliki format data `int64` dengan variasi yang berbeda sehingga perlu dilakukan encoding ke dalam indeks integer agar memiliki persebaran data yang seragam. Dengan demikian, data dapat digunakan untuk proses pelatihan model dengan lebih baik dan model dapat menemukan pola dari data dengan lebih mudah.

##### Data Normalization
  Data *rating*, yang merupakan hasil ulasan dari pelanggan, akan digunakan sebagai data terget pada pelatihan model yang akan merepresentasikan bahwa pelanggan suka atau tidak suka terhadap makanan yang diulas. Oleh karena itu, untuk memudahkan proses pengenalan pola oleh model, data *rating* dinormalisasi nilainya ke dalam rentang 0 - 1 menggunakan metode `MinMaxScaler()`. 

##### Data Split
  Langkah terakhir sebelum memasuki tahap pelatihan model adalah pembagian data menjadi data training dan data validasi. Pembagian data dilakukan dengan rasio 80 : 20 untuk data training dan data validasi. Data training digunakan untuk melatih model, sedangkan data validasi digunakan untuk mengevaluasi model yang telah dilatih bahwa model dapat memberikan performa yang baik terhadap data yang belum perah dilihat sebelumnya.

---
## Modeling
### Cosine Similarity
  Pendekatan algoritma cosine similarity digunakan untuk membuat model sistem rekomendasi dengan metode *content-based filtering*. Cosine similarity bekerja dengan cara mengukur kesamaan arah antara dua vektor dari representasi data. Algoritma ini menghitung besaran sudut cosinus antara vektor satu dengan lainnya. Semakin kecil derajat sudut, maka semakin besar nilai cosine similarity, artinya kedua data semakin mirip. Cosine similarity antara dua vektor **A** dan **B** dapat dihitung dengan formula berikut:

$$\[
\text{Cosine Similarity} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \cdot \sqrt{\sum_{i=1}^{n} B_i^2}}
\]$$

Di mana:
- $$\( A_i \)$$ dan $$\( B_i \)$$ adalah komponen dari vektor **A** dan **B** pada dimensi $$\( i \)$$.
- $$\( n \)$$ adalah jumlah dimensi vektor.

### Penjelasan
1. **Pembilang**: Hasil kali dot product antara dua vektor.
2. **Penyebut**: Perkalian dari magnitudo kedua vektor.

Cosine similarity menghasilkan nilai antara -1 hingga 1:
- $$\( 1 \)$$: Vektor memiliki arah yang sama.
- $$\( 0 \)$$: Vektor saling tegak lurus (tidak memiliki hubungan).
- $$\( -1 \)$$: Vektor memiliki arah yang berlawanan.

### Neural Collaborative Filtering


---
## Evaluation
Since the prediction model is a regression model, it used 3 evaluation metrics as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared. By using those metrics, the best prediction model can be developed.

### Metrics Description
| Metrics | Description | Formula |
|--------|-------------|---------|
| Mean Absolute Error (MAE) | MAE measures the average absolute error between the model's predictions and the actual values. A smaller MAE value indicates better model performance. |  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$, where the errors are considered in absolute terms. |
| Mean Squared Error (MSE) | MSE calculates the average of the squared errors between predictions and actual values, making it more sensitive to larger errors (outliers) than MAE | $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$ |
| R-squared | R-squared represents the proportion of the data variability that the model can explain. R-squared values range from 0 to 1, with values closer to 1 indicating that the model explains the data well. | $$R^2 = 1 - { Σ (yᵢ - ŷᵢ)² / Σ (yᵢ - ȳ)² }$$ |

This is the result of training on Model 1:

![evaluation model 1](https://github.com/user-attachments/assets/555012f1-ead8-449d-9e4e-e167c39ce1f8)

This is the result of training on Model 2:

![evaluation model 2](https://github.com/user-attachments/assets/126d990f-fee5-4252-826f-536a2efcdc49)

---
### Model Comparison
| Metrics | Model 1 | Model 2 | Notes |
|---------|---------|---------|------|
| MAE | 0.0505 | 0.0741 | Model 1 has a lower MAE compared to Model 2, meaning that Model 1's predictions, on average, are closer to the actual values than those of Model 2. |
| MSE | 0.0042 | 0.0114 | Model 1 has a smaller MSE than Model 2. This suggests that Model 1 is not only more accurate but also has more consistent small errors. Model 1 either handles outliers better or large errors occur less frequently compared to Model 2. |
| R-squared | 0.9917 | 0.9768 | Model 1 explains the relationship between input features and target values better than Model 2. |

That means Model 1 has a better architecture and design than Model 2. It has shown lower error rates in predicting nutrition density. Proper parameter configuration (as mentioned in solution statement number 3) helps to build a highly accurate prediction model. In addition, the model met the criteria we have set in the goals statements.

These are comparisons of the model prediction to the actual data.

![Prediction Model](https://github.com/user-attachments/assets/6dea88cd-38e3-4f06-97ed-7cc049e456ca)

---
## Inference
Nutrition density prediction is done using dummy data containing the values ​​of various nutrients contained in food like the original data.

The predictions were saved into a `.csv` file as follows:
[Prediction_of_dummy_data](https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/predictions_of_dummy_data.csv)

![Predicted Nutrition Density](https://github.com/user-attachments/assets/f06ba568-c713-4245-906a-cd8a7276b5bd)

---
## Conclusion
1. The model is a regression model because it aims to predict a continuous outcome (Nutrition Density Values) based on various input features (nutrient values). The prediction is estimating nutrient density values, which inherently involves predicting numerical quantities.

2. All data have skewed distribution and outliers, whereas the neural network model is more suitable for normally distributed data. Hence, the data needs to be transformed to logarithmic values and then normalized. By pre-processing the data this way, we can enhance the model's ability to learn meaningful patterns, leading to better performance and generalization on unseen data.

3. Based on the model evaluation, the architecture design and training scheme (splitting data into 90% `data_train`, 5% `data_validation`, and 5% `data_test`) of Model 1 is better than Model 2. This is drawn from comparing metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared values between the two models. The choice of architecture such as the number of layers and nodes, loss functions, and optimization techniques used in Model 1, likely contributed to its performance. In addition, there are recommendations to develop a model with superior performance as follows:

    * Further refinement of feature selection may enhance model performance. Identifying key features that significantly impact nutrition density could lead to improved predictive accuracy.
    * Explore different architectures and hyperparameter settings then identify the optimal configuration for the model.
    * Implementing regularization techniques such as dropout, L1/L2 regularization, or batch normalization can help prevent overfitting in complex models.

---
## Development Opportunity
__1. Meal Recommendation System__.
By integrating this dataset with broader dietary data, machine learning models can recommend dietary adjustments to individuals. For instance, a recommendation system could suggest lower-calorie or lower-sugar food alternatives to users looking to reduce their calorie intake but who still want to get high nutrition. Machine learning models can integrate food consumption data into broader dietary tracking tools used in fitness and health apps, providing users with insights into their dietary meal plans.

__2. Predictive Modeling for Health Impacts__.
With sufficient data linking meal consumption to health outcomes, predictive models could forecast health impacts based on meal consumption patterns. This could be particularly useful for public health.

---
## References
1. Alfarisi, B. I., et al. (2023). *Mengungkap Kesehatan Melalui Angka: Prediksi Malnutrisi Melalui Penilaian Status Gizi dan Asupan Makronutrien.* Prosiding SNPPM-5, 299-311.
2. Bouma, S. (2017). *Diagnosing Pediatric Malnutrition: Paradigm Shifts of Etiology-Related Definitions and Appraisal of the Indicators.* Nutrition in Clinical Practice, 32(1), 52–67.
3. Cakrawala. (2024). *Apa itu Neural Network? Ini Pengertian, Konsep, dan Contohnya.* Retrieved October 31, 2024, from [https://www.cakrawala.ac.id/berita/neural-network-adalah](https://www.cakrawala.ac.id/berita/neural-network-adalah).
4. Cederholm, T., et al. (2019). *GLIM criteria for the diagnosis of malnutrition – A consensus report from the global clinical nutrition community.* Journal of Cachexia, Sarcopenia and Muscle, 10(1), 207–217.
5. European Food Information Council (EUFIC). (2021). *What is nutrient density?* Retrieved October 31, 2024, from [https://www.eufic.org/en/understanding-science/article/what-is-nutrient-density](https://www.eufic.org/en/understanding-science/article/what-is-nutrient-density).
6. Khan, D. S. A., et al. (2022). *Nutritional Status and Dietary Intake of School-Age Children and Early Adolescents: Systematic Review in a Developing Country and Lessons for the Global Perspective.* Frontiers in Nutrition, 8(February).
7. Ministry of Health of the Republic of Indonesia. (2018). Situasi Balita Pendek (Stunting) di Indonesia.
8. RevoU. (2024). *Apa itu Neural Network.* Retrieved October 31, 2024, from [https://revou.co/kosakata/neural-network](https://revou.co/kosakata/neural-network).
9. Rinninella, E., et al. (2017). *Clinical tools to assess nutritional risk and malnutrition in hospitalized children and adolescents.* European Review for Medical and Pharmacological Sciences, 21(11), 2690–2701.
10. Simbolon, D. (2013). *Model Prediksi Indeks Massa Tubuh Remaja Berdasarkan Riwayat Lahir dan Status Gizi Anak.* Kesmas, 8(1), 19–27.
11. Yamantri, A. B., & Rifa’i, A. A. (2024). *Penerapan Algoritma C4.5 Untuk Prediksi Faktor Risiko Obesitas Pada Penduduk Dewasa.* Jurnal Komputer Antartika, 2(3), 118–125.
12. Zianka, I. D., et al. (2024). *The Design Android Application Nutrition Calculation to Prevent Stunting with CNN Method in Jakarta.* MALCOM: Indonesian Journal of Machine Learning and Computer Science, 4, 99–107.

> **Ini adalah bagian akhir laporan**
