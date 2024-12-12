# Recommendation System Project Report - Anggun Sulis Setyawan ✨
---
## Background
  Indonesia memiliki cita-cita besar untuk menjadi negara maju. Oleh karena itu, peningkatan kualitas Sumber Daya Manusia (SDM) menjadi pilar penting yang harus diperbaiki, terutama dalam hal peningkatan kesehatan, pemenuhan gizi, dan pencegahan stunting (Kemenko PMK RI, 2024). Pemerintah Indonesia telah merencanakan untuk melaksanakan program makan bergizi gratis sesuai arahan Presiden RI Prabowo Subianto. Pemerintah berharap program tersebut dapat menurunkan jumlah kasus stunting bahkan mencegah kasus stunting baru di masa depan. Fokus awal dari program ini adalah anak-anak sekolah dan kelompok rentan lainnya. Anak sekolah yang dimaksud antara lain pelajar PAUD, SD, SMP, dan SMA (detik.com, 2024). 

  Anak sekolah menjadi sasaran karena status gizi dan  stunting sangat mempengaruhi kecerdasan anak. Status gizi memberikan kontribusi terhadap kesulitan belajar sebesar 32,83%. Anak yang kekurangan nutrisi cenderung memiliki kelemahan pada sistem saraf hingga dapat menyebabkan kelainan motorik dan kognitif (Dewi, dkk, 2021). Sebagai calon penerus bangsa, maka pertumbuhan dan perkembangan anak sekolah perlu diperhatikan dengan baik agar menghasilkan potensi sumber daya manusia dengan kualitas maksimal. Hal ini dapat dicapai dengan salah satu cara yaitu pemenuhan kebutuhan nutrisi harian melalui program makanan bergizi gratis.

  Angka Kecukupan Gizi (AKG) menurut Kementerian Kesehatan Republik Indonesia adalah kecukupan rata-rata gizi harian yang dianjurkan untuk sekelompok orang setiap harinya. Kebutuhan gizi ideal anak yang harus terpenuhi dalam sehari terbagi menjadi dua kelompok, yaitu zat gizi makro dan mikro. Zat gizi makro adalah semua jenis zat gizi yang dibutuhkan anak dalam jumlah banyak, seperti energi (kalori), protein, lemak, dan karbohidrat. Sementara zat gizi mikro adalah nutrisi yang dibutuhkan dalam jumlah sedikit, seperti vitamin dan mineral (Damar Upahita, 2021). Penentuan nilai gizi disesuaikan dengan jenis kelamin, kelompok umur, tinggi badan, berat badan, serta aktivitas fisik (Kemenkes RI, 2019).

  Seluruh program makanan bergizi gratis harus melibatkan kolaborasi pemangku kepentingan terkait untuk dikonvergensikan sehingga bisa komprehensif dan terintegrasi. Salah satunya adalah penyediaan makanan yang efektif dan efisien. Salah satu perusahaan penyedia layanan catering dan bento, Olagizi ingin mengambil peran penting dalam penyediaan paket makanan bergizi bagi siswa SMP dan SMA. Olagizi ingin memberikan layanan dengan optimal. Oleh karena itu, Olagizi ingin membuat sebuah sistem yang dapat memberikan rekomendasi tentang makanan bergizi yang dipersonalisasi sesuai kebutuhan gizi dan selera para siswa. Di sisi lain, Olagizi juga ingin rekomendasi tersebut memberikan pilihan makanan yang dapat dimasak dalam waktu yang tidak terlalu lama agar makanan dapat disiapkan tepat pada waktu, khususnya makanan untuk sesi sarapan. Untuk pengembangan tahap awal, Olagizi ingin membuat model sistem rekomendasi makanan berdasarkan kemiripan jumlah kalori yang terkandung serta berdasarkan hasil ulasan rating makanan.

---
## Business Understanding
### Problem Statements
  1. Bagaimana sistem rekomendasi dapat memberikan pilihan makanan dengan bahan baku utama yang sama?
  2. Bagaimana sistem rekomendasi dapat memberikan berbagai pilihan makanan yang mungkin disukai oleh target pelanggan?
### Goals
  1. Menghasilkan 10 rekomendasi makanan yang memiliki bahan baku utama yang sama.
  2. Menghasilkan 10 rekomendasi makanan yang mungkin disukai oleh target pelanggan.
### Solution Approach
  1. Menerapkan pendekatan _content-based filtering_ menggunakan algoritma _cosine similarity_ untuk menghitung kemiripan bahan baku yang digunakan diurutkan berdasarkan nilai _similarity_ terbesar.
  2. Menerapkan pendekatan _collaborative filtering_ menggunakan algoritma _deep learning_ untuk menemukan pola pemberian rating oleh user.

---
## Data Understanding
### Overview
Dataset ini berasal dari platform Kaggle salah satu pengembang sistem rekomendasi makanan dengan nama akun "GRACE HEPHZIBAH M" yang dapat diakses pada link di bawah. Pada proyek ini, akan menggunakan 2 file dataset dalam format csv, yaitu food data dan rating data.

_Download raw dataset_:
[Food Recommendation System](https://www.kaggle.com/code/gracehephzibahm/food-recommendation-system-easy-comprehensive/input)

### Food Data
| No | Kolom | Tipe Data | Deskripsi |
|----|-------|-----------|-----------|
| 1 | Name | `object` | Nama makanan. |
| 2 | Food_ID | `integer` | ID makanan. |
| 3 | C_Type | `object` | Kategori makanan. |
| 4 | Veg_Non | `object` | Keterangan apakah makanan mengandung bahan baku hewani atau tidak |
| 5 | Describe | `object` | Keterangan bahan-bahan yang digunakan pada makanan tersebut. |

### Rating Data
| No | Kolom | Tipe Data | Deskripsi |
|----|-------|-----------|-----------|
| 1 | User_ID | `integer` | ID pengguna yang memberikan ulasan. |
| 2 | Food_ID | `integer` | ID resep yang diberi ulasan. |
| 3 | Rating | `integer` | Penilaian yang diberikan (dalam skala 1 - 10). |

### Explore Data
Eksplorasi data dilakukan untuk mengenali dan memahami data dengan lebih detail dan menyeluruh. Eksplorasi data ini dilakukan terhadap kedua dataset yang akan digunakan. Hasil eksplorasi menunjukkan dataset `food_df` memiliki total 400 baris data dan 5 kolom. `food_df` tidak memiliki duplikasi data maupun nilai null. Ini artinya data terdiri dari 400 makanan yang berbeda. Makanan pada dataset didominasi oleh makanan dengan kategori makanan India, Healthy Food, dan Dessert (makanan penutup) seperti yang ditampilkan pada chart berikut.

![kategori makanan](https://github.com/user-attachments/assets/2922e0a9-1e66-45d6-814e-a2721aafaeb4)

Sebagai tambahan, enam puluh persen makanan pada dataset merupakan makanan yang tidak mengandung bahan baku hewani, sementara sisanya menggunakan bahan baku hewani.

![Veg_Non](https://github.com/user-attachments/assets/df0c3cf5-527a-4254-ba35-f89555089dee)


Di sisi lain, dataset `rating_df` memiliki total 512 baris data dan 3 kolom. Hasil pengecekan menunjukkan dataset `rating_df` memiliki 3 data yang mengandung _missing value_. Karena rasionya sangat kecil dibandingkan jumlah total data, maka sebaiknya data tersebut dihapus. Diketahui bahwa distribusi data _rating_ cukup seimbang karena memiliki nilai `mean = 5.4` dan `median = 5.0`. Sementara rentang nilai rating berkisar dari 1 hingga 10. Hal ini didukung oleh hasil visualisasi dari distribusi data berikut.

![Distribusi Rating](https://github.com/user-attachments/assets/e10eeb2c-914c-4e79-8bd0-96a8bc7f0039)


---
## Data Preparation

#### _Data Cleaning_
  Salah satu tahap terpenting dalam _Data Preparation_ yaitu _Data Cleaning_. Proses ini dilakukan untuk memastikan bahwa data yang akan digunakan untuk melatih model merupakan data yang bersih, rapi, dan berkualitas. Misalnya memastikan format data sudah tepat sesuai dengan representasi data, perlakuan terhadap data yang hilang (_missing value_) maupun pencilan data (_outlier_), dll. Dengan begitu, proses persiapan data setelahnya dapat dilakukan dengan lebih mudah.

#### _Content Based Filtering_
##### Features Extraction
  Data bahan baku yang berupa teks harus diubah ke dalam data numerik dengan cara mengekstraksi fitur teks menjadi vektor menggunakan `TfidfVectorizer`. Hasil vektorisasi tersebut akan digunakan untuk menghitung kemiripan (_similarity_) bahan baku antar makanan pada dataset yang tersedia. Dengan demikian, ini dapat mempermudah model untuk memberikan rekomendasi berdasarkan bahan baku makanan.

#### Collaborative Filtering
##### Features Encoding
  Diketahui data *Food_ID* dan *User_ID* memiliki format data `float64` dengan variasi yang berbeda sehingga perlu dilakukan encoding ke dalam indeks integer agar memiliki persebaran data yang seragam. Dengan demikian, data dapat digunakan untuk proses pelatihan model dengan lebih baik dan model dapat menemukan pola dari data dengan lebih mudah.

##### Data Normalization
  Data *rating*, yang merupakan hasil ulasan dari pelanggan, akan digunakan sebagai data terget pada pelatihan model yang akan merepresentasikan bahwa pelanggan suka atau tidak suka terhadap makanan yang diulas. Oleh karena itu, untuk memudahkan proses pengenalan pola oleh model, data *rating* dinormalisasi nilainya ke dalam rentang 0 - 1 menggunakan metode `MinMaxScaler()`. 

##### Data Split
  Langkah terakhir sebelum memasuki tahap pelatihan model adalah pembagian data menjadi data training dan data validasi. Pembagian data dilakukan dengan rasio `80 : 20` untuk data training dan data validasi. Data training digunakan untuk melatih model, sedangkan data validasi digunakan untuk mengevaluasi model yang telah dilatih bahwa model dapat memberikan performa yang baik terhadap data yang belum perah dilihat sebelumnya.

---
## Modeling
### Cosine Similarity
#### Desain Model 
  Model akan didesain agar dapat memberikan rekomendasi dari makanan yang sebelumnya pernah dipilih berdasarkan kemiripan bahan baku yang digunakan. Pendekatan algoritma cosine similarity digunakan untuk membuat model sistem rekomendasi dengan metode *content-based filtering*. Cosine similarity bekerja dengan cara mengukur kesamaan arah antara dua vektor dari representasi data. Algoritma ini menghitung besaran sudut cosinus antara vektor satu dengan lainnya. Semakin kecil derajat sudut, maka semakin besar nilai cosine similarity, artinya kedua data semakin mirip. Cosine similarity antara dua vektor **A** dan **B** dapat dihitung dengan formula berikut:

$$\text{Cosine Similarity} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \cdot \sqrt{\sum_{i=1}^{n} B_i^2}}$$

Keterangan:
- $$A_i$$ dan $$\( B_i \)$$ adalah komponen dari vektor **A** dan **B** pada dimensi $$i$$.
- $$\n\$$ adalah jumlah dimensi vektor.

**_Penjelasan_**
1. **Pembilang**: Hasil kali dot product antara dua vektor.
2. **Penyebut**: Perkalian dari magnitudo kedua vektor.

Cosine similarity menghasilkan nilai antara -1 hingga 1:
- $$\( 1 \)$$: Dua vektor sangat mirip atau identik (vektor memiliki arah yang sama dalam ruang vektor).
- $$\( 0 \)$$: Dua vektor tidak memiliki kesamaan sama sekali (vektor saling tegak lurus atau tidak memiliki hubungan linear).
- $$\( -1 \)$$: Dua vektor sangat berbeda (vektor memiliki arah yang berlawanan dalam ruang vektor).

  Penerapan cosine similarity pada sistem rekomendasi, khususnya dengan metode vektoriasi TF-IDF memiliki kelebihan dan kekurangan yang perlu diperhatikan sebagai berikut:
**_Kelebihan_**
1. TF-IDF sering menghasilkan vektor yang _sparse_ (banyak elemen nol). Cosine similarity dapat bekerja sangat baik pada representasi seperti ini tanpa memerlukan normalisasi tambahan.
2. Cosine similarity mengukur sudut antara dua vektor, sehingga tidak terpengaruh oleh skala besar atau kecilnya nilai fitur. Hal ini bermanfaat ketika nilai TF-IDF memiliki variasi yang besar.
3. Cosine similarity efektif dalam mengukur kemiripan semantik antar item, yang sesuai dengan pendekatan berbasis TF-IDF.
4. Nilai cosine similarity berkisar antara 0 - 1 (non negatif) sehingga mudah untuk interpretasi derajat kemiripan.
5. Algoritma cosine similarity sederhana, cepat dihitung, dan mudah diimplementasikan dengan pustaka `scikit-learn`.

**_Kekurangan_**
1. Cosine similarity hanya mengukur kemiripan numerik dari vektor, tanpa memahami konteks sebenarnya dari kata-kata dalam teks.
2. Kombinasi cosine similarity dan TF-IDF bekerja pada asumsi representasi linier dari data sehingga tidak dapat menangkap hubungan non-linear yang lebih kompleks.
3. Jika jumlah fitur dalam TF-IDF sangat besar (terdapat banyak kata unik), perhitungan cosine similarity bisa menjadi lebih lambat, terutama jika diterapkan pada dataset besar.
4. Cosine similarity hanya membandingkan vektor, makna sebenarnya dari teks mungkin tidak tercermin dengan baik karena TF-IDF mengabaikan sinonim dan polisemi (satu kata dengan banyak makna).

#### Hasil Rekomendasi
  Setelah model dibuat, selajutnya model akan digunakan untuk mendapatkan 10 rekomendasi makanan lain berdasarkan makanan yang pernah dipilih pelanggan. Hasilnya diperoleh seperti berkut:
`rekomendasi_makanan("chicken minced salad)`
| No | Name | 
|----|------|
| 1 | chilli chicken |
| 2 | veg hakka noodles | 
| 3 | veg fried rice |
| 4 | prawn fried rice |
| 5 | chilli fish |
| 6 | garlic soya chicken |
| 7 | Thai Spareribs |
| 8 | Spicy Korean Steak |
| 9 | egg and garlic fried rice |
| 10 | almond and chicken momos (without shell) |

### Neural Collaborative Filtering
#### Desain Model 
  Pada sistem rekomendasi dengan _collaborative filtering_ akan digunakan algoritma _deep learning_, khususnya dengan menggunakan _embedding layer_. Model akan dilatih untuk dapat menemukan pola antara beragam data makanan yang disukai pelanggan. Model diharapkan dapat memberikan rekomendasi kepada pelanggan makanan populer yang belum pernah dipilih sebelumnya. _Embedding layer_ digunakan untk merepresentasikan data kategorikal seperti pelanggan dan makanan ke dalam ruang vektor berdimensi rendah yang dapat dilatih. Terdapat 2 _embedding layer_ (satu untuk pelanggan dan satu untuk makanan) yang kemudian digabungkan dengan operasi **dot product** untuk menghasilkan skor kesesuaian. Setelah embedding, hasilnya akan dilewatkan melalui lapisan neural network (dense) dan output dengan fungsi `sigmoid` untuk mendapatkan prediksi skor yang menunjukkan tingkat preferensi pengguna terhadap makanan tertentu.
  Pendekatan ini tentunya memiliki kelebihan dan kekurangan pada penerapannya seperti berikut:
**_Kelebihan_**
1. Embedding layer mengurangi dimensi data dengan merepresentasikan setiap data pelanggan dan makanan sebagai vektor berdimensi rendah tetapi tetap informatif.
2. Embedding dapat digabungkan dengan berbagai algoritma deep learning lainnya untuk menangkap pola tambahan. 
3. Embedding dapat menggeneralisasi ke data baru selama memiliki data yang cukup.
4. Embedding mampu mengatasi _data sparsity_ (sedikit data) untuk menemukan pola tersembunyi.
5. Neural network setelah embedding memungkinkan model menangkap pola kompleks antar data.

**_Kekurangan_**
1. Training embedding layer membutuhkan dataset besar agar representasi vektor cukup bermakna.
2. Jika embedding terlalu kompleks atau data terbatas, model dapat overfit pada data training.
3. Representasi embedding bersifat numerik dan sulit diinterpretasikan dibandingkan pendekatan statistik tradisional.
4. Training embedding dan neural network memerlukan waktu lebih lama, terutama pada dataset besar.

#### Hasil Rekomendasi
  Setelah model dibuat, selajutnya model akan digunakan untuk mendapatkan 10 rekomendasi makanan populer lain yang belum pernah dipilih pelanggan. Hasilnya diperoleh seperti berkut:
`Makanan kesukaan user sebelumnya: christmas cake, corn and raw mango salad`
| No | Name | 
|----|------|
| 1 | summer squash salad |
| 2 | roasted spring chicken with root veggies | 
| 3 | chicken dong style |
| 4 | andhra crab meat masala |
| 5 | malabari fish curry |
| 6 | malabar fish curry |
| 7 | surmai curry with lobster butter rice |
| 8 | instant rava dosa |
| 9 | wok tossed asparagus in mild garlic sauce |
| 10 | banana chips |

---
## Evaluation
### Content Based Filtering
  Untuk memudahkan evaluasi, perbandingan dilakukan terhadap relevansi makanan yang direkomendasikan terhadap makanan sebelumnya berdasarkan tipe bahan baku yang digunakan, yaitu Vegan atau Non-Vegan. Metrik evaluasi yang digunakan pada pendekatan ini adalah `Mean Cosine Similarity`. Metrik ini dapat mengukur rata-rata kemiripan antara makanan yang direkomendasikan berdasarkan perhitungan _cosine similarity_ dari bahan baku yang digunakan setiap makanan. Formula Mean Cosine Similarity adalah sebagai berikut.
  
$$\
\text{Mean Cosine Similarity} = \frac{1}{N} \sum_{i=1}^{N} \text{cosine similarity}(A, B_i)
\$$

Keterangan:
- $$\(N\)$$ adalah jumlah item yang relevan atau direkomendasikan.
- $$\(A\)$$ adalah item yang dipilih atau disukai oleh pengguna.
- $$\(B_i\)$$ adalah item-item yang direkomendasikan untuk item $$\(A\)$$.
- $$\(\text{cosine similarity}(A, B_i)\)$$ adalah nilai kemiripan cosine antara item $$\(A\)$$ dan $$\(B_i\)$$.

Terdapat kriteria umum untuk menilai rata-rata _cosine similarity_ antara lain:
1. **Nilai Tinggi (mendekati 1)**: ini menunjukkan kemiripan yang sangat kuat. Biasanya nilai > 0,8 dianggap sangat relevan.
2. **Nilai Tinggi (mendekati 1)**: ini menunjukkan kemiripan yang sangat kuat. Biasanya nilai antara 0,6 - 8 dianggap memiliki relevansi sedang.
3. **Nilai Tinggi (mendekati 1)**: ini menunjukkan kemiripan yang sangat kuat. Biasanya nilai antara 0,4 - 6 dianggap kurang relevan.
4. **Nilai Tinggi (mendekati 1)**: ini menunjukkan kemiripan yang sangat kuat. Biasanya nilai < 0,4 dianggap tidak relevan.

  Hasil evaluasi model menunjukkan nilai Mean Cosine Similarity sebesar 0,859 (sangat relevan) dengan rincian sebagai berikut:
`Makanan sebelumnya --> chicken minced salad merupakan tipe non-veg`.
| No | name | Tipe | Skor Relevansi |
|----|------|------|----------------|
| 1 | chilli chicken | non-veg | 1.0 |
| 2 | veg hakka noodles | non-veg | 0.48 |
| 3 | veg fried rice | non-veg | 0.48 |
| 4 | prawn fried rice | non-veg | 0.48 |
| 5 | chilli fish | non-veg | 1.0 |
| 6 | garlic soya chicken | non-veg | 1.0 |
| 7 | Thai Spareribs | non-veg | 1.0 |
| 8 | Spicy Korean Steak | non-veg | 1.0 |
| 9 | egg and garlic fried rice | non-veg | 1.0 |
| 10 | almond and chicken momos (without shell) | non-veg | 1.0 |


### Neural Collaborative Filtering
Root Mean Squared Error (RMSE) mengukur rata-rata _error_ antara nilai prediksi yang dihasilkan oleh model dan nilai sebenarnya (_ground thruth_). Error dihitung dengan mengambil selisih antara nilai prediksi dan nilai sebenarnya, kemudian dikuadratkan untuk memastikan semua error bernilai positif. Rata-rata dari error kuadrat akan dihitung dan dikonversi ke skala asli data dengan kalkulasi nilai akar kuadrat. Berikut formula lengkapnya:

$$\
\text{RMSE} = sqrt((1/n) * Σ (y_i - ŷ_i)²
\$$

Keterangan:
* $$\(n\)$$: Jumlah observasi (data point yang dievaluasi).
* $$\(y_i)\$$: Nilai sebenarnya (ground truth) untuk data ke-𝑖.
* $$\(ŷ_i)\$$: Nilai prediksi.
* $$\(y_i - ŷ_i)\$$: Error antara nilai sebenarnya dan prediksi.

![coll_train](https://github.com/user-attachments/assets/ed71ff51-364b-4151-9381-fd2be5a3eb7c)

Berdasarkan hasil pelatihan model dapat disimpulkan bahwa:
* Grafik _history_ pelatihan model menunjukkan nilai RMSE pada data training terus menurun secara konsisten, artinya bahwa model semakin baik dalam mempelajari pola pada data training.
* Nilai RMSE pada data validasi juga menurun, meski lebih lambat dibandingkan dengan data training. Ini menunjukkan bahwa model cukup baik dalam melakukan generalisasi pada data yang tidak terlihat sebelumnya.
* Terlihat tidak ada kenaikan signifikan pada nilai RMSE validasi yang menandakan bahwa overfitting belum terjadi.

---
## Conclusion
1. Sistem rekomendasi dapat memberikan pilihan makanan yang memiliki kemiripan dari segi bahan baku yang digunakan dengan menggunakan model rekomendasi yang dikembangkan melalui pendekatan _content-based filtering_. Model dapat memberikan rekomendasi berbagai pilihan makanan berdasarkan makanan yang pernah dipilih oleh pelanggan. Model memanfaatkan dataset yang berisi data nama dan daftar bahan baku makanan yang tersedia. Model akan menghitung _cosine similarity_ antara bahan baku dari nama makanan yang dipilih, kemudian menghasilkan rekomendasi nama makanan yang memiliki kemiripan bahan baku dengan mengurutkannya berdasarkan nilai _similarity_ terbesar. 

2. Sistem rekomendasi dapat memberikan pilihan makanan yang mungkin disukai oleh pelanggan dengan menggunakan model rekomendasi yang dikembangkan melalui pendekatan _collaborative filtering_. Model dapat memberikan rekomendasi berbagai pilihan makanan yang belum pernah dicoba dan mungkin akan disukai berdasarkan makanan yang sebelumnya pernah disukai oleh pelanggan. Model dikembangkan dengan memanfaatkan dataset interaksi pelanggan terhadap makanan melalui ulasan _rating_ kepuasan pelanggan. Model dilatih dengan algoritma embedding yang dikombinasikan dengan neural network sehingga dapat memprediksi makanan yang mungkin akan disukai oleh pelanggan.

---
## References
1. Damar Upahita. 2021. *Panduan Mencukupi Kebutuhan Gizi Harian Untuk Anak Usia Sekolah (6 - 9 Tahun).* https://hellosehat.com/parenting/anak-6-sampai-9-tahun/gizi-anak/kebutuhan-asupan-gizi-anak/?amp=1. 

2. Dewi, dkk. 2021. *Pentingnya Pemenuhan Gizi Terhadap Kecerdasan Anak*. SENAPADMA:Seminar Nasional Pendidikan Dasar dan Menengah, Vol.1, pp. 16-21. Sukabumi: Universitas Nusa Putra.

1. KA, Mutirasari. 2024. *Program Makan Bergizi Gratis: Jadwal Berlaku, Sasaran hingga Aturan Pembagian.* Diakses pada 6 Desember 2024, dari https://news.detik.com/berita/d-7617806/program-makan-bergizi-gratis-jadwal-berlaku-sasaran-hingga-aturan-pembagian.

2. Kementerian Koordinator Bidang Pembangunan Manusia dan Kebudayaan Republik Indonesia. 2024. *Program Makan Bergizi Gratis untuk Tingkatkan Kualitas SDM Indonesia.* Diakses pada 6 Desember 2024, dari https://www.kemenkopmk.go.id/program-makan-bergizi-gratis-untuk-tingkatkan-kualitas-sdm-indonesia.

3. Kementerian Kesehatan Republik Indonesia. 2019. *Peraturan Menteri Kesehatan Republik Indonesia Nomor 28 Tahun 2019 Tentang Angka Kecukupan Gizi Yang Dianjurkan Untuk Masyarakat Indonesia.*

> **Ini adalah bagian akhir laporan**
