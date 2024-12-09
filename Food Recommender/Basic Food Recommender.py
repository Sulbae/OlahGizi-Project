# %% [markdown]
# # Basic Food Recommender Model

# %% [markdown]
# ## Background

# %% [markdown]
# Indonesia memiliki cita-cita besar untuk menjadi negara maju. Oleh karena itu, peningkatan kualitas Sumber Daya Manusia (SDM) menjadi pilar penting yang harus diperbaiki, terutama dalam hal peningkatan kesehatan, pemenuhan gizi, dan pencegahan stunting (Kemenko PMK RI, 2024). Pemerintah Indonesia telah merencanakan untuk melaksanakan program makan bergizi gratis sesuai arahan Presiden RI Prabowo Subianto. Pemerintah berharap program tersebut dapat menurunkan jumlah kasus stunting bahkan mencegah kasus stunting baru di masa depan. Fokus awal dari program ini adalah anak-anak sekolah dan kelompok rentan lainnya. Anak sekolah yang dimaksud antara lain pelajar PAUD, SD, SMP, dan SMA (detik.com, 2024). 
# 
# Anak sekolah menjadi sasaran karena status gizi dan  stunting sangat mempengaruhi kecerdasan anak. Status gizi memberikan kontribusi terhadap kesulitan belajar sebesar 32,83%. Anak yang kekurangan nutrisi cenderung memiliki kelemahan pada sistem saraf hingga dapat menyebabkan kelainan motorik dan kognitif (Dewi, dkk, 2021). Sebagai calon penerus bangsa, maka pertumbuhan dan perkembangan anak sekolah perlu diperhatikan dengan baik agar menghasilkan potensi sumber daya manusia dengan kualitas maksimal. Hal ini dapat dicapai dengan salah satu cara yaitu pemenuhan kebutuhan nutrisi harian melalui program makanan bergizi gratis.
# 
# Angka Kecukupan Gizi (AKG) menurut Kementerian Kesehatan Republik Indonesia adalah kecukupan rata-rata gizi harian yang dianjurkan untuk sekelompok orang setiap harinya. Kebutuhan gizi ideal anak yang harus terpenuhi dalam sehari terbagi menjadi dua kelompok, yaitu zat gizi makro dan mikro. Zat gizi makro adalah semua jenis zat gizi yang dibutuhkan anak dalam jumlah banyak, seperti energi (kalori), protein, lemak, dan karbohidrat. Sementara zat gizi mikro adalah nutrisi yang dibutuhkan dalam jumlah sedikit, seperti vitamin dan mineral (Damar Upahita, 2021). Penentuan nilai gizi disesuaikan dengan jenis kelamin, kelompok umur, tinggi badan, berat badan, serta aktivitas fisik (Kemenkes RI, 2019).
# 
# Seluruh program makanan bergizi gratis harus melibatkan kolaborasi pemangku kepentingan terkait untuk dikonvergensikan sehingga bisa komprehensif dan terintegrasi. Salah satunya adalah penyediaan makanan yang efektif dan efisien. Salah satu perusahaan penyedia layanan catering dan bento, Olagizi ingin mengambil peran penting dalam penyediaan paket makanan bergizi bagi siswa SMP dan SMA. Olagizi ingin memberikan layanan dengan optimal. Oleh karena itu, Olagizi ingin membuat sebuah sistem yang dapat memberikan rekomendasi tentang makanan bergizi yang dipersonalisasi sesuai kebutuhan gizi dan selera para siswa. Di sisi lain, Olagizi juga ingin rekomendasi tersebut memberikan pilihan makanan yang dapat dimasak dalam waktu yang tidak terlalu lama agar makanan dapat disiapkan tepat pada waktu, khususnya makanan untuk sesi sarapan. Untuk pengembangan tahap awal, Olagizi ingin membuat model sistem rekomendasi makanan berdasarkan kemiripan bahan baku yang digunakan serta berdasarkan hasil ulasan rating makanan.

# %% [markdown]
# ## Business Understanding

# %% [markdown]
# ### Problem Statements

# %% [markdown]
# 1. Bagaimana sistem rekomendasi dapat memberikan pilihan makanan dengan bahan baku utama yang sama?
# 
# 2. Bagaimana sistem rekomendasi dapat memberikan berbagai pilihan makanan yang mungkin disukai oleh target pelanggan?

# %% [markdown]
# ### Goals

# %% [markdown]
# 1. Menghasilkan 10 rekomendasi makanan yang memiliki bahan baku utama yang sama.
# 
# 2. Menghasilkan 10 rekomendasi makanan yang mungkin disukai oleh target pelanggan.

# %% [markdown]
# ### Solution Approach

# %% [markdown]
# 1. Menerapkan pendekatan _content-based filtering_ menggunakan algoritma _cosine similarity_ untuk menghitung kemiripan bahan baku yang digunakan diurutkan berdasarkan nilai _similarity_ terbesar.
# 
# 2. Menerapkan pendekatan _collaborative filtering_ menggunakan algoritma _deep learning_ untuk menemukan pola pemberian rating oleh user.

# %% [markdown]
# ## Data Understanding

# %% [markdown]
# #### Overview

# %% [markdown]
# Dataset ini berasal dari platform Kaggle salah satu pengembang sistem rekomendasi makanan dengan nama akun "GRACE HEPHZIBAH M" yang dapat diakses pada link di bawah. Pada proyek ini, akan menggunakan 2 file dataset dalam format csv, yaitu food data dan rating data.
# 
# _Download raw dataset_:
# [Food Recommendation System](https://www.kaggle.com/code/gracehephzibahm/food-recommendation-system-easy-comprehensive/input)

# %% [markdown]
# ##### food_df

# %% [markdown]
# | No | Kolom | Tipe Data | Deskripsi |
# |----|-------|-----------|-----------|
# | 1 | Name | `object` | Nama makanan. |
# | 2 | Food_ID | `integer` | ID makanan. |
# | 3 | C_Type | `object` | Kategori makanan. |
# | 4 | Veg_Non | `object` | Keterangan apakah makanan mengandung bahan baku hewani atau tidak |
# | 5 | Describe | `object` | Keterangan bahan-bahan yang digunakan pada makanan tersebut. |

# %% [markdown]
# ##### rating_df

# %% [markdown]
# | No | Kolom | Tipe Data | Deskripsi |
# |----|-------|-----------|-----------|
# | 1 | User_ID | `integer` | ID pengguna yang memberikan ulasan. |
# | 2 | Food_ID | `integer` | ID resep yang diberi ulasan. |
# | 3 | Rating | `integer` | Penilaian yang diberikan (dalam skala 1 - 10). |

# %% [markdown]
# ### Import Libraries

# %% [markdown]
# Import seluruh library yang diperlukan:

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Flatten, Dot, Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

# %% [markdown]
# ### Load Data

# %% [markdown]
# Muat semua dataset yang akan digunakan:

# %%
# Data 1
food_data = "https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Food%20Recommender/raw-dataset/food.csv"
food_df = pd.read_csv(food_data)

# Data 2
rating_data = "https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Food%20Recommender/raw-dataset/ratings.csv"
rating_df = pd.read_csv(rating_data)

# %% [markdown]
# ### Explore Data

# %% [markdown]
# #### Explore food_df

# %%
food_df.info()

# %% [markdown]
# * Diketahui dataset memiliki 2 tipe data yaitu `int64` (untuk kolom Food_ID) dan `object` (untuk kolom lainnya) dengan total baris sebanyak 400.

# %%
# Check duplicates
print(f"Duplicated data:", food_df.duplicated().sum())

# recheck missing value
print(f"Missing value:", food_df.isna().sum().sum())

# %% [markdown]
# * Tidak terdapat duplikasi data maupun _missing value_.

# %% [markdown]
# Menampilkan 5 data teratas:

# %%
food_df.head()

# %% [markdown]
# Visualisasi data *C_Type*:

# %%
category_counts = food_df['C_Type'].value_counts().sort_values(ascending=True)

# Visualisasi dengan bar chart horizontal
plt.figure(figsize=(8, 6))
bars = plt.barh(category_counts.index, category_counts.values, color=sns.color_palette("pastel"))

for bar in bars:
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width()}', va='center', ha='left', fontsize=12)

plt.title('Distribusi Kategori C_Type', fontsize=16)
plt.xlabel('Frekuensi', fontsize=14)
plt.ylabel('Kategori', fontsize=14)
plt.show()

# %% [markdown]
# * Makanan pada dataset didominasi oleh makanan dengan kateogori makanan India, Healthy Food, dan Dessert.

# %%
Veg_Non_counts = food_df['Veg_Non'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(Veg_Non_counts, labels=Veg_Non_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Proporsi Kategori Veg_Non', fontsize=16)
plt.show()

# %% [markdown]
# * Enam puluh persen makanan pada dataset merupakan makanan yang tidak mengandung bahan baku hewani, sementara sisanya menggunakan bahan baku hewani.

# %% [markdown]
# #### Explore rating_df

# %%
rating_df.info()

# %% [markdown]
# * Diketahui seluruh data pada dataset memiliki tipe data `float`.

# %% [markdown]
# Menampilkan 5 data teratas:

# %%
rating_df.head()

# %% [markdown]
# Cek duplikasi dan missing value data:

# %%
# Check duplicates
print(f"Duplicated data:", rating_df.duplicated().sum())

# Check missing value
print(f"Missing value:", rating_df.isna().sum().sum())

# %% [markdown]
# * Terdapat __3__ __missing value__.

# %% [markdown]
# Menampilkan statistik deskriptif:

# %%
rating_df.describe().T

# %% [markdown]
# * Diketahui rata-rata rating yang diberikan user adalah 5,4 dan diketahui nilai median sebesar 5. Sementara rentang nilai rating berkisar dari 1 hingga 10. Artinya distribusi data cukup seimbang.

# %% [markdown]
# Cek distribusi dan outlier data _rating_ menggunakan histogram dan boxplot:

# %%
for col in rating_df.select_dtypes(include="number").columns:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    fig.suptitle(f"Distribution of {col}")

    # Histogram
    sns.histplot(rating_df[col], kde=True, ax=axes[0])
    axes[0].set_ylabel("Frequency")
    # Box PLot
    sns.boxplot(x=rating_df[col], ax=axes[1])

    plt.tight_layout()
    plt.show()

# %% [markdown]
# * Diketahui tidak terdapat outlier pada data _rating_.

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# ### Data Cleaning

# %% [markdown]
# #### Remove Missing Value

# %% [markdown]
# Menghapus data yang mengandung *missing_value*:

# %%
rating_df = rating_df.dropna()

# %% [markdown]
# Recheck:

# %%
# recheck missing value
print(f"Missing value:", rating_df.isna().sum().sum())

# %% [markdown]
# Menghapus baris data yang memiliki missing value pada  `food_df`:

# %%
food_df = food_df.dropna()

# %% [markdown]
# Recheck:

# %%
# Recheck missing value
print(f"Missing value:", food_df.isna().sum().sum())

# %%
food_df.info()

# %% [markdown]
# ### Content-Based Filtering

# %% [markdown]
# Membuat variable baru untuk memudahkan dalam proses selanjutnya serta memudahkan pemeliharaan data:

# %%
content_based_df = food_df
content_based_df

# %% [markdown]
# Check data:

# %%
content_based_df.info()

# %% [markdown]
# #### Feature Extraction

# %%
content_based_df['Describe']

# %% [markdown]
# Mengubah data teks menjadi vektor numerik:

# %%
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,1), token_pattern=r'\b[a-zA-Z]+\b')

tfidf_vectorizer.fit(content_based_df['Describe'])

teks = tfidf_vectorizer.get_feature_names_out()

# %%
teks

# %%
tfidf_matrix = tfidf_vectorizer.fit_transform(content_based_df['Describe'])

tfidf_matrix.shape

# %%
tfidf_matrix.todense()

# %%
pd.DataFrame(tfidf_matrix.todense(),
             columns=teks,
             index=content_based_df.Describe
             ).sample(20, axis=1).sample(5, axis=0)

# %% [markdown]
# ### Collaborative Filtering

# %% [markdown]
# Membuat variable baru untuk memudahkan dalam proses selanjutnya serta memudahkan pemeliharaan data:

# %%
collaborative_df = rating_df

collaborative_df.head()

# %% [markdown]
# Check data:

# %%
collaborative_df.info()

# %% [markdown]
# #### Features Encoding

# %% [markdown]
# Melakukan encoding terhadap data Food_ID:

# %%
# Mengubah recipe_id menjadi list tanpa nilai yang sama
food_ids = collaborative_df["Food_ID"].unique().tolist()
print("list Food_ID: ", food_ids)

# Melakukan encoding terhadap recipe_id
food_to_food_encoded = {x: i for i, x in enumerate(food_ids)}
print("encoded Food_ID: ", food_to_food_encoded)

# Melakukan proses encoding angka ke recipe_id
food_encoded_to_food = {i: x for i, x in enumerate(food_ids)}
print("encoded angka ke Food_ID: ", food_encoded_to_food)

# %% [markdown]
# Melakukan encoding terhadap data User_ID:

# %%
# Mengubah user_id menjadi list tanpa nilai yang sama
user_ids = collaborative_df["User_ID"].unique().tolist()
print("list User_ID: ", user_ids)

# Melakukan encoding terhadap user_id
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print("encoded User_ID: ", user_to_user_encoded)

# Melakukan proses encoding angka ke user_id
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print("encoded angka ke User_ID: ", user_encoded_to_user)

# %% [markdown]
# Memetakan recipe_id dan user_id yang telah diproses ke dataframe yang berkaitan:

# %%
# Mapping recipe_id ke dataframe recipe
collaborative_df['food'] = collaborative_df['Food_ID'].map(food_to_food_encoded)

# Mapping user_id ke dataframe user
collaborative_df['user'] = collaborative_df['User_ID'].map(user_to_user_encoded)

# %% [markdown]
# Check data:

# %%
collaborative_df.info()

# %% [markdown]
# Menampilkan 5 data teratas:

# %%
collaborative_df.head()

# %% [markdown]
# #### Data Normalization

# %% [markdown]
# Normalisasi data target untuk pelatihan:

# %%
rating_scaler = MinMaxScaler()

collaborative_df['rating'] = rating_scaler.fit_transform(collaborative_df['Rating'].values.reshape(-1, 1)).round(2)

# %% [markdown]
# Check data:

# %%
collaborative_df['rating']

# %% [markdown]
# #### Data Split

# %% [markdown]
# Data diacak terlebih dahulu agar distribusinya random:

# %%
collaborative_df = collaborative_df.sample(frac=1, random_state=42)

# %% [markdown]
# Menentukan fitur dan target:

# %%
x = collaborative_df[['user', 'food']].values
y = collaborative_df['rating'].values

# %% [markdown]
# Pembagian data ke dalam training set dan validation set:

# %%
train_indices = int(0.8 * collaborative_df.shape[0])

x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print(x, y)

# %% [markdown]
# ## Modeling

# %% [markdown]
# ### Cosine Similarity

# %% [markdown]
# Menghitung similarity:

# %%
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# %% [markdown]
# Membuat dataframe hasil perhitungan similarity:

# %%
cos_sim_df = pd.DataFrame(cos_sim, 
                          index=content_based_df['Name'],
                          columns=content_based_df['Name'])

print("Shape:", cos_sim_df.shape)

cos_sim_df.sample(10, axis=1).sample(10, axis=0)

# %% [markdown]
# ### Neural Collaborative Filtering

# %% [markdown]
# Menghitung jumlah nilai unik masing-masing fitur:

# %%
n_users = collaborative_df['user'].nunique()
print(n_users)

n_food = collaborative_df['food'].nunique()
print(n_food)

# %% [markdown]
# Membuat model rekomendasi dengan embedding layer:

# %%
class NFCRecommender(tf.keras.Model):
    
     # Insialisasi fungsi
    def __init__(self, n_users, n_food, embedding_size, dense_units, **kwargs):
        super(NFCRecommender, self).__init__(**kwargs)
        self.n_users = n_users
        self.n_food = n_food
        self.embedding_size = embedding_size
        self.dense_units = dense_units
        self.dropout = layers.Dropout(0.5)

        # Membentuk layer embedding user
        self.users_embedding = layers.Embedding(
            n_users,
            embedding_size,
            embeddings_initializer = 'he_normal',
            embeddings_regularizer = keras.regularizers.l2(1e-4)
        )
        self.users_bias = layers.Embedding(n_users, 1)

        # Membentuk layer embedding recipe
        self.food_embedding = layers.Embedding(
            n_food,
            embedding_size,
            embeddings_initializer = 'he_normal',
            embeddings_regularizer = keras.regularizers.l2(1e-4)
        )
        self.food_bias = layers.Embedding(n_food, 1)

        # Dense layers
        self.dense1 = layers.Dense(
            units=dense_units,
            activation='relu',  # Aktivasi ReLU untuk representasi non-linear
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(1e-4)
        )
        self.dense2 = layers.Dense(
            units=1, 
            activation='sigmoid',  # Output berupa probabilitas
            kernel_regularizer=keras.regularizers.l2(1e-4)
        )

    def call(self, inputs):
        users_vector = self.users_embedding(inputs[:,0]) # memanggil layer embedding 1
        users_bias = self.users_bias(inputs[:, 0]) # memanggil layer embedding 2
        food_vector = self.food_embedding(inputs[:, 1]) # memanggil layer embedding 3
        food_bias = self.food_bias(inputs[:, 1]) # memanggil layer embedding 4

        dot_users_food = tf.reduce_sum(users_vector * food_vector, axis=1, keepdims=True)

        x = dot_users_food + users_bias + food_bias

        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)

        # Menggunakan fungsi aktivasi sigmoid
        return tf.nn.sigmoid(x)

# %% [markdown]
# Compile model:

# %%
model = NFCRecommender(n_users, n_food, embedding_size=20, dense_units=32)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(),
    metrics = [tf.keras.metrics.RootMeanSquaredError()]
)

# %%
history = model.fit(
    x = x_train,
    y = y_train,
    validation_data=(x_val, y_val),
    batch_size=128,
    epochs=20
)

# %% [markdown]
# Visualisasi hasil pelatihan model:

# %%
train_result, ax = plt.subplots(figsize=(10, 6))

ax.plot(history.history["root_mean_squared_error"], label="Training RMSE", color='blue', linewidth=2)
ax.plot(history.history["val_root_mean_squared_error"], label="Validation RMSE", color='orange', linewidth=2)

ax.set_title("Evaluasi Model: RMSE per Epoch", fontsize=14)
ax.set_ylabel("Root Mean Squared Error (RMSE)", fontsize=12)
ax.set_xlabel("Epoch", fontsize=12)

ax.legend(loc="upper right", fontsize=12)

train_result.tight_layout()

train_result.show()

# %% [markdown]
# ## Get Recommendations

# %% [markdown]
# ### Content Based Filtering

# %% [markdown]
# Membuat fungsi `rekomendasi_makanan()`:

# %%
def rekomendasi_makanan(nama_makanan, similarity_data=cos_sim_df, items=content_based_df, k = 10):
    """
    Rekomendasi Resto berdasarkan kemiripan dataframe
 
    Parameter:
    ---
    nama_resto : tipe data string (str)
                Nama Restoran (index kemiripan dataframe)
    similarity_data : tipe data pd.DataFrame (object)
                      Kesamaan dataframe, simetrik, dengan resto sebagai 
                      indeks dan kolom
    items : tipe data pd.DataFrame (object)
            Mengandung kedua nama dan fitur lainnya yang digunakan untuk mendefinisikan kemiripan
    k : tipe data integer (int)
        Banyaknya jumlah rekomendasi yang diberikan
    ---
 
 
    Pada index ini, kita mengambil k dengan nilai similarity terbesar 
    pada index matrix yang diberikan (i).
    """
    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan
    # Dataframe diubah menjadi numpy Range(start, stop, step)
    index = similarity_data.loc[:, nama_makanan].to_numpy().argpartition(
        range(-1, -k, -1))

    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1 : -(k + 2) : -1]]

    # Drop nama makanan target
    closest = closest.drop(nama_makanan, errors = "ignore")

    # Gabungkan dengan DataFrame asli untuk mengambil detail makanan
    recommendations = pd.DataFrame(closest, columns=["Name"]).merge(items, on="Name")

    return recommendations.head(k)

# %% [markdown]
# Menampilkan makanan yang pernah dipilih

# %%
# Menampilkan makanan yang pernah dipilih
makanan_pertama_df = content_based_df[content_based_df['Name'] == "chicken minced salad"]
makanan_pertama_df

# %% [markdown]
# Membuat dataframe hasil rekomendasi makanan:

# %%
# Menampilkan rekomendasi makanan berdasarkan makanan yang pernah dipilih
rekomendasi_makanan_df = pd.DataFrame(rekomendasi_makanan("chicken minced salad"))

# %% [markdown]
# Menampilkan rekomendasi makanan berdasarkan makanan yang pernah dipilih menggunakan fungsi `rekomendasi_makanan()`:

# %%
rekomendasi_makanan_df

# %% [markdown]
# ### Collaborative Filtering

# %% [markdown]
# Membuat filter untuk sampel user dan makanan:

# %%
df = collaborative_df
 
# Mengambil sample user
user_id = df.User_ID.sample(1).iloc[0]
selected_food_by_user = df[df.User_ID == user_id]

food_not_selected = food_df[~food_df['Food_ID'].isin(selected_food_by_user.Food_ID.values)]['Food_ID'] 
food_not_selected = list(
    set(food_not_selected)
    .intersection(set(food_to_food_encoded.keys()))
)
 
food_not_selected = [[food_to_food_encoded.get(x)] for x in food_not_selected]
user_encoder = user_to_user_encoded.get(user_id)
user_resto_array = np.hstack(
    ([[user_encoder]] * len(food_not_selected), food_not_selected)
)

# %% [markdown]
# Menggunakan model untuk memberikan rekomendasi makanan:

# %%
ratings = model.predict(user_resto_array).flatten()
 
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_food_ids = list(set([
    food_encoded_to_food.get(food_not_selected[x][0])
    for x in top_ratings_indices
]))
 
print('Showing recommendations for users: {}'.format(user_id))
print('===' * 9)
print('Food with high ratings from user')
print('----' * 8)

top_food_user = (
    selected_food_by_user.sort_values(
        by = 'rating',
        ascending=False
    )
    .head(2)
    .Food_ID.values
)
 
food_df_rows = food_df[food_df['Food_ID'].isin(top_food_user)]
for row in food_df_rows.itertuples():
    print(row.Name)
 
print('----' * 8)
print('Top 10 food recommendation')
print('----' * 8)
 
recommended_food = food_df[food_df['Food_ID'].isin(recommended_food_ids)].head(10)
for row in recommended_food.itertuples():
    print('Food ID:', row.Food_ID, '-' , 'Food Name:', row.Name)
    print('----')

# %% [markdown]
# ## Evaluation

# %% [markdown]
# ### Content Based Filtering

# %% [markdown]
# Tampilkan data makanan pertama:

# %%
makanan_pertama_df

# %% [markdown]
# Tampilkan data hasil rekomendasi:

# %%
rekomendasi_makanan_df

# %% [markdown]
# Gabungkan data makanan_pertama_df dan rekomendasi_makanan_df, khususnya untuk kolom _name_ dan _ingredients_:

# %%
perbandingan_df = pd.concat([
    makanan_pertama_df,
    rekomendasi_makanan_df
], ignore_index=True)

# %%
perbandingan_df

# %% [markdown]
# Menghitung nilai relevansi dengan cara membandingkan kemiripan bahan baku pada makanan yang direkomendasikan model:

# %%
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))

# %%
vectorizer.fit(perbandingan_df['Veg_Non'])

bahan = vectorizer.get_feature_names_out()

# %% [markdown]
# Check fitur:

# %%
bahan

# %% [markdown]
# Menghitung skor relevansi:

# %%
tfidf_bahan = vectorizer.fit_transform(perbandingan_df['Veg_Non'])
tfidf_bahan.todense()

eval_cos_sim = cosine_similarity(tfidf_bahan[0:1], tfidf_bahan)

perbandingan_df['Skor Relevansi'] = eval_cos_sim.flatten()

# %% [markdown]
# Check matrix similarity:

# %%
bahan_matrix = pd.DataFrame(tfidf_bahan.todense(),
             columns=bahan,
             index=perbandingan_df.Veg_Non
)

n_cols = min(3, bahan_matrix.shape[1])  # Sampel kolom
n_rows = min(3, bahan_matrix.shape[0])  # Sampel baris

bahan_matrix.sample(n_cols, axis=1).sample(n_rows, axis=0)

# %% [markdown]
# Menampilkan hasil perhitungan skor relevansi:

# %%
perbandingan_df

# %% [markdown]
# Menghitung rata-rata skor relevansi:

# %%
print("Mean Cosine Similarity:", perbandingan_df['Skor Relevansi'].mean().round(3))

# %% [markdown]
# ### Collaborative Filtering

# %% [markdown]
# Menampilkan visualisasi history pelatihan model:

# %%
train_result

# %% [markdown]
# Catatan:
# * Grafik _history_ pelatihan model menunjukkan nilai RMSE pada data training terus menurun secara konsisten, artinya bahwa model semakin baik dalam mempelajari pola pada data training.
# * Nilai RMSE pada data validasi juga menurun, meski lebih lambat dibandingkan dengan data training. Ini menunjukkan bahwa model cukup baik dalam melakukan generalisasi pada data yang tidak terlihat sebelumnya.
# * Terlihat tidak ada kenaikan signifikan pada nilai RMSE validasi yang menandakan bahwa overfitting belum terjadi.

# %% [markdown]
# ## Conclusion

# %% [markdown]
# 1. Sistem rekomendasi dapat memberikan pilihan makanan yang memiliki kemiripan dari segi bahan baku yang digunakan dengan menggunakan model rekomendasi yang dikembangkan melalui pendekatan _content-based filtering_. Model dapat memberikan rekomendasi berbagai pilihan makanan berdasarkan makanan yang pernah dipilih oleh pelanggan. Model memanfaatkan dataset yang berisi data nama dan daftar bahan baku makanan yang tersedia. Model akan menghitung _cosine similarity_ antara bahan baku dari nama makanan yang dipilih, kemudian menghasilkan rekomendasi nama makanan yang memiliki kemiripan bahan baku dengan mengurutkannya berdasarkan nilai _similarity_ terbesar. 
# 
# 2. Sistem rekomendasi dapat memberikan pilihan makanan yang mungkin disukai oleh pelanggan dengan menggunakan model rekomendasi yang dikembangkan melalui pendekatan _collaborative filtering_. Model dapat memberikan rekomendasi berbagai pilihan makanan yang belum pernah dicoba dan mungkin akan disukai berdasarkan makanan yang sebelumnya pernah disukai oleh pelanggan. Model dikembangkan dengan memanfaatkan dataset interaksi pelanggan terhadap makanan melalui ulasan _rating_ kepuasan pelanggan. Model dilatih dengan algoritma embedding yang dikombinasikan dengan neural network sehingga dapat memprediksi makanan yang mungkin akan disukai oleh pelanggan.

# %% [markdown]
# ## References

# %% [markdown]
# 1. Damar Upahita. 2021. *Panduan Mencukupi Kebutuhan Gizi Harian Untuk Anak Usia Sekolah (6 - 9 Tahun).* https://hellosehat.com/parenting/anak-6-sampai-9-tahun/gizi-anak/kebutuhan-asupan-gizi-anak/?amp=1. 
# 
# 2. Dewi, dkk. 2021. *Pentingnya Pemenuhan Gizi Terhadap Kecerdasan Anak*. SENAPADMA:Seminar Nasional Pendidikan Dasar dan Menengah, Vol.1, pp. 16-21. Sukabumi: Universitas Nusa Putra.
# 
# 1. KA, Mutirasari. 2024. *Program Makan Bergizi Gratis: Jadwal Berlaku, Sasaran hingga Aturan Pembagian.* Diakses pada 6 Desember 2024, dari https://news.detik.com/berita/d-7617806/program-makan-bergizi-gratis-jadwal-berlaku-sasaran-hingga-aturan-pembagian.
# 
# 2. Kementerian Koordinator Bidang Pembangunan Manusia dan Kebudayaan Republik Indonesia. 2024. *Program Makan Bergizi Gratis untuk Tingkatkan Kualitas SDM Indonesia.* Diakses pada 6 Desember 2024, dari https://www.kemenkopmk.go.id/program-makan-bergizi-gratis-untuk-tingkatkan-kualitas-sdm-indonesia.
# 
# 3. Kementerian Kesehatan Republik Indonesia. 2019. *Peraturan Menteri Kesehatan Republik Indonesia Nomor 28 Tahun 2019 Tentang Angka Kecukupan Gizi Yang Dianjurkan Untuk Masyarakat Indonesia.*

# %% [markdown]
# 


