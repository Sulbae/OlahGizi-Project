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

  Seluruh program makanan bergizi gratis harus melibatkan kolaborasi pemangku kepentingan terkait untuk dikonvergensikan sehingga bisa komprehensif dan terintegrasi. Salah satunya adalah penyediaan makanan yang efektif dan efisien. Salah satu perusahaan penyedia layanan catering dan bento, Olagizi ingin mengambil peran penting dalam penyediaan paket makanan bergizi bagi siswa SMP dan SMA. Olagizi ingin memberikan layanan dengan optimal. Oleh karena itu, Olagizi ingin membuat sebuah sistem yang dapat memberikan rekomendasi tentang makanan bergizi yang dipersonalisasi sesuai kebutuhan gizi dan selera para siswa. Di sisi lain, Olagizi juga ingin rekomendasi tersebut memberikan pilihan makanan yang dapat dimasak dalam waktu yang tidak terlalu lama agar makanan dapat disiapkan tepat pada waktu, khususnya makanan untuk sesi sarapan.

---
## Business Understanding
### Problem Statements
  1. Bagaimana sistem rekomendasi dapat memberikan pilihan makanan yang dipersonalisasi sesuai kebutuhan kalori seseorang?
  2. Bagaimana sistem rekomendasi dapat memberikan berbagai pilihan makanan yang mungkin sesuai dengan selera seseorang?
### Goals
  1. Menghasilkan 10 rekomendasi makanan yang memiliki nilai kalori sesuai dengan kebutuhan seseorang dan dapat dimasak dalam waktu kurang dari 2 jam.
  2. Menghasilkan 10 rekomendasi makanan yang mungkin sesuai dengan selera seseorang dan dapat dimasak dalam waktu kurang dari 2 jam.
### Solution Statements
  1. Menerapkan pendekatan _collaborative filtering_ menggunakan algoritma _deep learning_.
  2. Menerapkan pendekatan _content-based filtering_ menggunakan algoritma _cosine similarity_.
---
## Data Understanding
### Overview
  Dataset ini berasal dari platform Kaggle salah satu pengembang sistem rekomendasi makanan diet dengan nama akun @SOUMEDHIK yang dapat diakses pada link di bawah. Ukuran dataset begitu besar sehingga dataset yang digunakan hanya sebagian sampel saja. Pada proyek ini, hanya akan menggunakan 3 file dataset dalam format csv, yaitu recipes data, interactions data, dan people profile data.

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

### People Profile Data
`people_profile_df` memiliki total 10726 baris data dan 11 kolom.

| No | Kolom | Tipe Data | Deskripsi |
|----|-------|-----------|-----------|
| 1 | Unnamed:0 | `integer` | - |
| 2 | age | `integer` | Usia individu (dalam tahun). |
| 3 | weight (kg) | `float` | Berat badan individu (dalam kilogram). |
| 4 | height (m) | `float` | Tinggi badan individu (dalam meter). |
| 5 | gender | `object` | Jenis kelamin (`F` untuk perempuan, `M` untuk laki-laki.) |
| 6 | BMI | `float` | Body Mass Index. Rasio ideal antara berat dan tinggi badan. |
| 7 | BMR (kcal/hari) | `float` | Basal Metabolic Rate, kebutuhan kalori dasar individu. |
| 8 | activity_level | `float` | Tingkat aktivitas individu. |
| 9 | calories_to_maintain_weigth | `float` | Kalori yang dibutuhkan untuk mempertahankan berat badan. |
| 10 | BMI_tags | `integer` | Kategori BMI berdasarkan nilai (misalnya underweight, normal, dan overweight.)
| 11 | Label | `integer` | Kategori atau kelas. |

### Explore Data
  Eksplorasi data dilakukan untuk mengetahui lebih banyak terkait karakteristik dataset yang akan digunakan, mulai dari kelengkapan data, format data, dan statistik data. Berdasarkan hasil eksplorasi data, terdapat beberapa temuan antara lain:
* Pada dataset `recipes_sample_df` ditemukan 52 _missing value_ untuk kolom _description_ dan terdapat format data yang belum sesuai pada kolom _submitted_ dan _nutrition_. Menurut hasil deskripsi statistik rata-rata makanan pada data membutuhkan waktu memasak sekitar 130 menit atau sekitar 9 tahapan.
* Pada dataset `interactions_sample_df` ditemukan 1 _missing value_ untuk kolom _review_ dan terdapat format data yang belum sesuai pada kolom _date_. Diketahui rata-rata nilai ulasan atau _rating_ yang diberikan orang-orang adalah 4,4.
* Representasi data dari kolom _Unnamed:0_ pada dataset `people_profile_df` sulit untuk diidentifikasi. Kemudian, diketahui rentang usia pada data meliputi balita (< 5 tahun), anak-anak (6-10) tahun, remaja (12 - 17 tahun), dewasa (18 - 65 tahun), hingga lanjut usia (> 65 tahun).

### Univariate Analysis
#### recipes_sample_df
![minutes](https://github.com/user-attachments/assets/743048bc-8aab-455f-81b9-cab385b348b6)
Diketahui distribusi data kolom _minutes_ pada `recipes_sample_df` sangat _skewed_ karena data memiliki _outlier_ yang cukup banyak dan rentang nilai yang ekstrem. 

![n_steps](https://github.com/user-attachments/assets/ab2a868b-9885-40cc-97c0-380132ce11f0)
Sementara itu, data kolom _n_steps_ juga terdistribusi _skewed_, tetapi memiliki lebih sedikit outlier dengan rentang yang tidak terlalu ekstrem.

#### interactions_sample_df
![rating](https://github.com/user-attachments/assets/03c5e40b-31d7-4834-b77a-c827bc9b0ce5)
Data ulasan yang ada menunjukkan dominasi rating yang diberikan berkisar 3 - 5. Hal ini menandakan bahwa banyak pelanggan atau pengguna yang merasa puas.

#### people_profile_df
![age](https://github.com/user-attachments/assets/2bbe20a5-3a95-4892-93f8-e5aef1618190)
![weight](https://github.com/user-attachments/assets/8a58d8f6-6c66-4b6d-94c6-bcff611d27d6)
![height](https://github.com/user-attachments/assets/04837a9e-e1b0-4a80-b6bd-e88bf3ef012d)
Data didominasi oleh orang dengan usia berkisar antara belasan hingga tiga puluhan tahun dengan berat badan berkisar antara 45 - 85 kg dan tinggi badan lebih dari 1,5 meter.

![BMI](https://github.com/user-attachments/assets/cc2d78e2-586c-4e63-8267-8b0345052912)
![BMR](https://github.com/user-attachments/assets/11e964d2-b900-4cdb-a366-6509534f12da)
![calories_to_maintain_weight](https://github.com/user-attachments/assets/076ce107-e5fd-4149-a888-11fd126bb02d)
Mayoritas orang terklasifikasi memiliki Body Mass Index (BMI) 20 - 30 dengan Basal Metabolic Rate (BMR) 1300 - 1600 kcal/hari, sedangkan mereka membutuhkan kalori untuk mempertahankan berat badan nya minimal sekitar 1800 - 2400 kcal/hari.

### Multivariate Analysis
Untuk mengetahui berbagai variabel yang berpegaruh terhadap kebutuhan kalori seseorang, maka dilakukan analisis terhadap korelasi antar variabel menggunakan _heatmap correlation matrix_.
![correlation with calorie](https://github.com/user-attachments/assets/68594496-53bf-465e-8115-800f83bd98ab)
Diketahui hampir seluruh variabel memiliki korelasi positif yang cukup untuk memengaruhi kebutuhan kalori setiap individu.

---
## Data Preparation

#### _Data Cleaning_
  Salah satu tahap terpenting dalam _Data Preparation_ yaitu _Data Cleaning_. Proses ini dilakukan untuk memastikan bahwa data yang akan digunakan untuk melatih model merupakan data yang bersih, rapi, dan berkualitas. Misalnya memastikan format data sudah tepat sesuai dengan representasi data, perlakuan terhadap data yang hilang (_missing value_) maupun pencilan data (_outlier_), dll. Dengan begitu, proses persiapan data setelahnya dapat dilakukan dengan lebih mudah.

#### _Filter Data 
##### recipes_sample_df
Berdasarkan _goals_ yang telah ditetapkan, Olagizi ingin sistem memberikan rekomendasi makanan yang dapat dimasak kurang dari 2 jam. Oleh karena itu, data sebaiknya disaring terlebih dahulu.
##### people_profile_df
Berdasarkan latar belakang project, sasaran yang dituju Olagizi adalah siswa SMP dan SMA. Oleh karena itu, data yang digunakan cukup data *people_profile* dengan rentang usia 12 - 18 tahun. Hasilnya, terdapat 1920 data yang memiliki rentang nilai _age_ 12 - 18 tahun.

#### _Features Engineering_
1. Ekstrak data pada fitur _nutrition_ dalam `recipes_sample_df`:
   Diketahui data pada kolom _nutrition_ memiliki format `object` yang berisi sebuah `list`. Data tersebut mengandung informasi jumlah kalori, lemak, gula, sodium, protein, lemak jenuh, dan karbohidrat. Untuk memudahkan analisis, maka data harus diekstrak menjadi kolom-kolom tersendiri. Ini dilakukan agar hubungan antar variabel data dapat dianalisis dengan lebih mudah pada proses selanjutnya.

2. Menambahkan data *people_id* pada `people_profile_df`:
   Menambahkan *people_id* pada `people_profile_df` dengan mengambil data dari user_id pada `interactions_sample_df`. Data *user_id* pada `interactions_sample_df` diacak terlebih dahulu, kemudian ditambahkan ke dalam kolom *people_id* pada `people_profile_df`. Hal ini dilakukan untuk memanipulasi data agar data dapat gabungkan dengan mudah pada proses selanjutnya.

#### _Data Merging_
  Penggabungan dataframe antara `recipes_sample_df` dan `interaction_sample_df` dilakukan agar dapat lebih mudah menganalisis korelasi antara variabel rating dengan variabel lainnya. Penggabungan data tersebut memanfaatkan key value yaitu _recipe_id_. Lalu, gabungkan juga `people_profile_df` agar dapat menemukan insight yang lebih lengkap.

### Features Selection
  Selanjutnya, pemilihan data-data yang relevan dilakukan agar meringankan beban komputasi pada proses analisis berikutnya. Fitur-fitur yang terpilih untuk proses analisis selanjutnya adalah sebagai berikut.
| Kolom Terpilih | Dataframe Asal |
|-----------|----------------|
| recipe_id | interactions_sample_df |
| name | recipes_sample_df | 
| description | recipes_sample_df | 
| minutes | recipes_sample_df | 
| nutrition | recipes_sample_df | 
| n_steps | recipes_sample_df | 
| calories | recipes_sample_df | 
| fat | recipes_sample_df | 
| sugar | recipes_sample_df | 
| sodium | recipes_sample_df | 
| protein | recipes_sample_df | 
| saturated fat | recipes_sample_df | 
| carbohydrates | recipes_sample_df | 
| user_id | interactions_sample_df | 
| rating | interactions_sample_df | 
| review | interactions_sample_df | 
| age | people_profile_df | 
| weight(kg) | people_profile_df | 
| height(m) | people_profile_df | 
| gender | people_profile_df | 
| BMI | people_profile_df | 
| BMR | people_profile_df | 
| activity_level | people_profile_df | 
| calories_to_maintain_weight | people_profile_df | 


### Content-Based Filtering


### Data Normalization
As we know almost all of the data have a skewed distribution, different value ranges, and outliers. 
#### _Log Transform_
First, transform the data into logarithmic value using `np.log()`. It will handle the outliers and reduce the skewness. The logarithmic transformation helps stabilize variance and reduce the effect of outliers by compressing the range of the data. For example, the Nutrient Density values ​​change to the following:

![Log Transform Caloric Value](https://github.com/user-attachments/assets/6dedeeff-10b1-494f-9a5b-8754546971a1)

![Log Transform Nutrition Density](https://github.com/user-attachments/assets/515892cd-6030-406c-b046-5acec1697256)

#### _Normalization_
Second, normalize the data using `RobustScaler()` to scale the data by using the median and Interquartile Range (IQR). This way helps to adjust the feature scales, which is crucial for neural networks as they are sensitive to the scale of input features. 

![Normalized Caloric Value](https://github.com/user-attachments/assets/38802f3d-1625-4b9b-8859-c6963871048a)

![Normalized Nutrition Density](https://github.com/user-attachments/assets/3866fedd-f1ed-4afe-802c-e24478deacad)

### Data Split
Data was separated into features (X) and target (y). All nutrient columns are the features. Then, The nutrition Density column would be the target.

Two data-splitting schemes were used for two different models as follows:
1. `TEST_SIZE_1 = 0.1`. The data will be split into 90% `data_train`, 5% `data_validation`, and 5% `data_test`. 
2. `TEST_SIZE_2 = 0.2`. The data will be split into 80% `data_train`, 10% `data_validation`, and 10% `data_test`.

---
## Modeling
### Model Design
The parameters for each model are set as follows:
| Parameter | Model 1 | Model 2 |
|-----------|---------|---------|
| Optimizer | RMSprop | Adam |
| Loss | MSE | Huber |
| Output Activation | Linear | Linear |
| Batch Normalization | 1 | 1 |
| Dropout | 0.3 | 0.3 |
| Dense Layers | 4 | 3 |
| Epochs | 150 | 300 |
| Batch | 128 | 128 |

The model design can be seen in the following diagram:

![MODEL STRUCTURE](https://github.com/user-attachments/assets/c1423c3a-e06c-4539-bdab-acce70558162)

### Training
The model training stage consists of the following steps:
1. __Forward Pass__
   * _Input Data_: Nutritional Data (features) will be fed into the `input layer`.
   * _Prediction_: The model calculates the `weight` and then produces Nutrition Density as a prediction by calculating the `activation function` in every neuron.
2. __Loss__
  *  _Loss Caculation_: The model will calculate the `loss function` value to assess how well the model's predictions compare to the actual values.
  *  The loss function for regression can be expressed by the following formula:

$$
\text{Loss} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:
- yi is the actual value,
- ŷi is the model prediction,
- n is the number of data points.

3. __Backward Pass__
  *  _Gradient Descent_: The `gradient` will be calculated from the loss function against the model weights. This process is carried out using the chain rule to calculate how much each weight contributes to the output error.
  *  _Weights Updating_: The `weights` are refined using `optimization` algorithms, such as `RMSProp()` and `Adam()`, to minimize the loss.
  *  The weight update at each iteration can be expressed by the following formula:

$$
w \gets w - \eta \frac{\partial \text{Loss}}{\partial w}
$$

Where:
- w is the weight,
- η (eta) is the learning rate,
- ∂Loss/∂w is the gradient of the loss function to the weight.

4. __Training Evaluation__
   *  _Validation_: At each `epoch` or every few epochs, the model is evaluated on the validation set. The loss function on the validation set is calculated to see how the model generalizes beyond the training data.

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
