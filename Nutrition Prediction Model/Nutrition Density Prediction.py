# %% [markdown]
# # Basic Model for Predicting Nutrient Density of Food Consumption

# %% [markdown]
# ## Background

# %% [markdown]
# Adequate nutritional status is crucial for human growth and survival. Nutritional status can be assessed by evaluating individual-specific nutritional requirements and intake. An imbalance between nutritional needs and intake may lead to either deficiency or excess, both of which negatively impact health. This condition is commonly referred to as malnutrition (Bouma, 2017; Rinninella et al., 2017). According to the World Health Organization (WHO), malnutrition can occur due to an imbalance in nutrient intake, which may affect health status, disrupt food digestion, or impair nutrient absorption (Khan et al., 2022). Malnutrition is not limited to undernutrition; it also encompasses a broader scope including macronutrient and micronutrient imbalances, obesity, cachexia, sarcopenia, and malnourishment (Cederholm et al., 2019).
# 
# In Indonesia, malnutrition—including both undernutrition and obesity—remains a serious issue. The 2018 Global Nutrition Report revealed that one in five child deaths globally is associated with poor dietary intake. Based on data from Indonesia's 2018 Basic Health Research, malnutrition cases include 30.8% of stunting, 3.5% of severe malnutrition, and other related conditions such as obesity (Zianka et al., 2024). Nutritional issues in children, if left unaddressed, may continue into adolescence and adulthood (Simbolon, 2013). Such conditions have far-reaching negative implications, as children with inadequate nutritional intake may experience delayed brain development and lower intelligence. Ultimately, this may hinder national economic growth and even increase poverty. The potential economic losses due to malnutrition are significant and may create a financial burden for the healthcare system in the future (Kemenkes RI, 2018).
# 
# On the other hand, not only children but over a quarter of the adult population in Indonesia are overweight. The high prevalence of obesity among adults indicates poor dietary patterns. Consuming high-calorie, high-fat, and high-sugar foods without sufficient physical activity is a major factor contributing to obesity (Yamantri et al., 2024).
# 
# One approach to addressing malnutrition is by providing widespread education on the importance of balanced nutrition, particularly for children. This can be achieved quickly and cost-effectively by leveraging digital technology for information distribution. Additionally, advancements in artificial intelligence (AI), such as intelligent agent assistants, can help users access information quickly and accurately. For this reason, the authors have developed a machine learning model to predict the nutrition density of commonly consumed foods and beverages. Nutrition density is a metric that reflects the amount of nutrients provided per unit of energy or calories. This concept is designed to evaluate the nutritional quality of a food item based on its nutrient content relative to its caloric value, thereby helping individuals choose nutrient-dense foods without excess energy or unnecessary calories (EUFIC, 2021). The model aims to be applied to a nutritional balance information service that enables users to make informed food choices suited to their individual needs and conditions.
# 
# The machine learning model will be developed using Google’s TensorFlow framework, chosen for its flexibility and scalability, allowing the authors to build machine learning applications from small to large scales. The model will employ a Neural Network algorithm, selected for its capability to handle complex data and hidden layers that facilitate automatic feature extraction, thus recognizing correlations among nutritional attributes without requiring extensive manual data processing (RevoU, 2024; Cakrawala, 2024). This enables the model to deliver accurate predictions of nutrition density based on available nutritional data for various food items.
# 
# _Keywords: Malnutrition, Nutrition Density, Neural Network, TensorFlow_

# %% [markdown]
# ## Business Understanding

# %% [markdown]
# ### Problem Statements

# %% [markdown]
# 1. How would the model work in predicting nutrition density?
# 
# 2. How to carry out good data processing that fits the model architecture?
# 
# 3. What factors need to be considered to develop the best model?

# %% [markdown]
# ### Goals

# %% [markdown]
# 1. Understanding the way neural network model works in predicting nutrition density.
# 
# 2. Processing raw data into clean data that is ready to be used to train neural network models.
# 
# 3. Designing the best model with the smallest prediction error.

# %% [markdown]
# ### Solution Statements

# %% [markdown]
# 1. Determine the dataset to be used and the expected output so we can find out the right type of prediction architecture.
# 
# 2. Carrying out an iterative process that includes Exploratory Data Analysis to understand data characteristics and data transformation to adjust the data format to the neural network model architecture.
# 
# 3. Carrying out feature engineering stages to select features that most influence nutrition density values and hyperparameter tuning (For example, training 2 models with different architecture configurations) to optimize model performance.

# %% [markdown]
# ## Data Understanding

# %% [markdown]
# ### Dataset

# %% [markdown]
# Resouce:
# [Food Nutrition Dataset](https://www.kaggle.com/datasets/utsavdey1410/food-nutrition-dataset/data)

# %% [markdown]
# #### Overview

# %% [markdown]
# The Comprehensive Nutritional Food Database provides detailed nutritional information for a wide range of food items commonly consumed around the world. This dataset aims to support dietary planning, nutritional analysis, and educational purposes by providing extensive data on the macro and micronutrient content of foods.

# %% [markdown]
# #### Column Description

# %% [markdown]
# No | Column | Description
# ---|--------|------------
# 1 | Food | The name or type of the food item
# 2 | Caloric Value | Total energy provided by the food, typically measured in kilocalories(kcal) per 100 grams.
# 3 | Fat (in g) | Total amount of fats in grams per 100 grams, including the breakdowns that follow
# 4 | Saturated Fats (in g) | Amount of saturated fats (fats that typically raise the level of cholesterol in the blood) in grams per 100 grams.
# 5 | Monounsaturated Fats (in g) | Amount of monounsaturated fats (considered heart-healthy fats) in grams per 100 grams.
# 6 | Polyunsaturated Fats (in g) | Amount of polyunsaturated fats (include essential fats your body needs but can't produce itself) in grams per 100 grams.
# 7 | Carbohydrates (in g) | Total carbohydrates in grams per 100 grams, including sugars.
# 8 | Sugars (in g) | Total sugars in grams per 100 grams, a subset of carbohydrates.
# 9 | Protein (in g) | Total proteins in grams per 100 grams, essential for body repair and growth.
# 10 | Dietary Fiber (in g) | Fiber content in grams per 100 grams, important for digestive health.
# 11 | Cholesterol (in mg) | Cholesterol content in milligrams per 100 grams, pertinent for cardiovascular health.
# 12 | Sodium (in mg) | Sodium content in milligrams per 100 grams, crucial for fluid balance and nerve function.
# 13 | Water (in g) | Water content in grams per 100 grams, which affects the food's energy density.
# 14 | Vitamin A (in mg) | Amount of Vitamin A in micrograms per 100 grams, impoertant for vision and immune functioning.
# 15 | Vitamin B1 (Thiamine)(in mg) | Essential for glucose metabolism.
# 16 | Vitamin B11 (Folic Acid)(in mg) | Crucial for cell function and tissue growth, particularly important in pregnancy.
# 17 | Vitamin B12(in mg) | Important for brain function and blood formation.
# 18 | Vitamin B2 (Riboflavin)(in mg) | Necessary for energy production, cell function, and fat metabolism.
# 19 | Vitamin B3 (Niacin)(in mg) | Support digestive system, skin, and nerves health.
# 20 | Vitamin B5 (Pantothenic Acid)(in mg) | Necessary for making blood cells and helps convert food into energy.
# 21 | Viatmin B6 (in mg): Important for normal brain development and keeping the nervous and immune system healthy.
# 22 | Vitamin C (in mg) | Important for the repair of all body tissues.
# 23 | Vitamin D (in mg) | Crucial for absorption of calcium, promoting bone growth and health.
# 24 | Vitamin E (in mg) | Acts as an antioxidant, helping to protect cells from the damage caused by free radicals.
# 25 | Vitamin K (in mg) | Necessary for blood clotting and bone health.
# 26 | Calcium (in mg) | Vital for building and maintaining strong bones and teeth.
# 27 | Copper (in mg) | Helps with the formation of collagen, increases the absorption of iron and plays a role in energy production.
# 28 | Iron (in mg) | Essential for the creation of red blood cells.
# 29 | Magnesium (in mg) | Important for many processes in the body including regulation of muscle and nerve function, blood sugar levels, and blood pressure and making protein, bone, and DNA.
# 30 | Manganese (in mg) | Involved in the formation of bones, blood clotting factors, and enzymes that play a role in fat and carbohydrate metabolism, calcium absorption, and blood sugar regulation.
# 31 | Phosporus (in mg) | Helps with the formation of bones and teeth and is necessary for the body to make protein for the growth, maintenance, and repair of cells and tissues.
# 32 | Potassium (in mg) | Helps regulate fluid balance, muscle contractions, and nerve signals.
# 33 | Selenium (in mg) | Important for reproduction, thyroid gland function, DNA production, and protecting the body from damage caused by free radicals and from infection.
# 34 | Zinc (in mg) | Necessary for the immune system to properly function and plays a role in cell division, cell growth, wound healing, and the breakdown of carbohydrates.
# 35 | Nutrition Density | A metric indicating the nutrient richness of the food per calorie.

# %% [markdown]
# ### Explore

# %% [markdown]
# #### Import Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import mse, huber
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %% [markdown]
# #### Load Data

# %%
# Data 1
data_1_url = "https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/dataset/FOOD-DATA-GROUP1.csv"
data_1_df = pd.read_csv(data_1_url)

# Data 2
data_2_url = "https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/dataset/FOOD-DATA-GROUP2.csv"
data_2_df = pd.read_csv(data_2_url)

# Data 3
data_3_url = "https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/dataset/FOOD-DATA-GROUP3.csv"
data_3_df = pd.read_csv(data_3_url)

# Data 4
data_4_url = "https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/dataset/FOOD-DATA-GROUP4.csv"
data_4_df = pd.read_csv(data_4_url)

# Data 5
data_5_url = "https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/dataset/FOOD-DATA-GROUP5.csv"
data_5_df = pd.read_csv(data_5_url)

# %% [markdown]
# #### Check Data

# %%
all_data = [data_1_df, data_2_df, data_3_df, data_4_df, data_5_df]

for data in all_data:
    print(data.shape)
    print(data.columns.tolist())

# %% [markdown]
# _Findings:_
# * All data have 37 typical columns
# * Data Shape:
#     - Data 1 has 551 rows, 
#     - Data 2 has 319 rows, 
#     - Data 3 has 571 rows, 
#     - Data 4 has 232 rows, 
#     - Data 5 has 722 rows.
# * There are two unknown columns that look useless, namely columns Unnamed: 0.1 and Unnamed: 0.

# %%
# Drop the unkonwn columns
columns_to_drop = ["Unnamed: 0.1", "Unnamed: 0"]

for data in all_data:
    data.drop(columns=columns_to_drop, inplace=True)
    print(data.shape)
    print(data.columns.tolist())

# %% [markdown]
# ##### Data 1

# %%
data_1_df.info()

# %%
# Check duplicates
print(f"Duplicated data:", data_1_df.duplicated().sum())

# recheck missing value
print(f"Missing value:", data_1_df.isna().sum().sum())

# %%
data_1_df.head()

# %%
data_1_df.describe(include="all").T

# %% [markdown]
# ##### Data 2

# %%
data_2_df.info()

# %%
# Check duplicates
print(f"Duplicated data:", data_2_df.duplicated().sum())

# recheck missing value
print(f"Missing value:", data_2_df.isna().sum().sum())

# %%
data_2_df.head()

# %%
data_2_df.describe(include="all").T

# %% [markdown]
# ##### Data 3

# %%
data_3_df.info()

# %%
# Check duplicates
print(f"Duplicated data:", data_3_df.duplicated().sum())

# recheck missing value
print(f"Missing value:", data_3_df.isna().sum().sum())

# %%
data_3_df.head()

# %%
data_3_df.describe(include="all").T

# %% [markdown]
# ##### Data 4

# %%
data_4_df.info()

# %%
# Check duplicates
print(f"Duplicated data:", data_1_df.duplicated().sum())

# recheck missing value
print(f"Missing value:", data_1_df.isna().sum().sum())

# %%
data_4_df.head()

# %%
data_4_df.describe(include="all").T

# %% [markdown]
# ##### Data 5

# %%
data_5_df.info()

# %%
# Check duplicates
print(f"Duplicated data:", data_1_df.duplicated().sum())

# recheck missing value
print(f"Missing value:", data_1_df.isna().sum().sum())

# %%
data_5_df.head()

# %%
data_5_df.describe(include="all").T

# %% [markdown]
# #### Merge Data

# %%
# Merge all data into one
merged_data = pd.concat([data_1_df, data_2_df, data_3_df, data_4_df, data_5_df], axis=0, ignore_index=True)

# %%
merged_data.info()

# %%
# Ensure there is no duplicated data
print(f"Duplicated data:", merged_data.duplicated().sum())

# %%
merged_data.describe(include="all").T

# %% [markdown]
# #### Convert Data

# %%
merged_data.head()

# %% [markdown]
# _Note:_
# * Caloric Value (in kcal)
# * Fat, Saturated Fats, Monounsaturated Fats, Polyunsaturated Fats, Carbohydrates, Sugars, Protein, Dietary Fiber, and Water (in g)
# * __all of these should be in mg__

# %%
# kcal to mg
merged_data["Caloric Value"] = merged_data["Caloric Value"] * 129.6 

# %%
# grams to milligrams
column_in_grams = ["Fat", 
                   "Saturated Fats",
                   "Monounsaturated Fats",
                   "Polyunsaturated Fats",
                   "Carbohydrates",
                   "Sugars",
                   "Protein",
                   "Dietary Fiber",
                   "Water"]

merged_data[column_in_grams] = merged_data[column_in_grams] * 1000

# %%
merged_data.head()

# %%
merged_data.describe().T

# %% [markdown]
# #### Univariate Analysis

# %%
# Check the data distribution of numerical columns
for col in merged_data.select_dtypes(include="number").columns:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    fig.suptitle(f"Distribution of {col}")

    # Histogram
    sns.histplot(merged_data[col], kde=True, ax=axes[0])
    axes[0].set_ylabel("Frequency")
    # Box PLot
    sns.boxplot(x=merged_data[col], ax=axes[1])

    plt.tight_layout()
    plt.show()

# %% [markdown]
# #### Multivariate Analysis

# %%
# Check data correlation
plt.figure(figsize=(30, 30))
sns.heatmap(merged_data.select_dtypes(include="number").corr(),
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1)
plt.title("Correlation Matrix of Nutrient")
plt.show()

# %% [markdown]
# _Fingdings:_
# * According to heatmap above, there are some nutrients that have small contribution to the nutrition density such as
#     - Vitamin A
#     - Vitamin B11
#     - Vitamin B12
#     - Vitamin D
#     - Vitamin K
#     - Copper
#     - Manganese
#     - Selenium

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# ### Feature Enginering

# %%
columns_to_eliminate = ["food", "Vitamin A", "Vitamin B11", "Vitamin B12", "Vitamin D", "Vitamin K", "Copper", "Manganese", "Selenium"]

selected_features = merged_data.drop(columns=columns_to_eliminate, axis=1)
selected_features.info()

# %% [markdown]
# ### Data Normalization

# %% [markdown]
# #### Log Transform

# %%
# Log transform data
columns_to_log = selected_features.columns

selected_features[columns_to_log] = selected_features[columns_to_log].apply(lambda x: np.log(x + 1))

# %%
selected_features.head()

# %%
selected_features.describe().T

# %%
# Check the data distribution of Caloric Value after log transform as a sample
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
fig.suptitle(f"Distribution of Caloric Value")

# Histogram
sns.histplot(selected_features["Caloric Value"], kde=True, ax=axes[0])
axes[0].set_ylabel("Frequency")
# Box PLot
sns.boxplot(x=selected_features["Caloric Value"], ax=axes[1])

plt.tight_layout()
plt.show()

# %%
# Check the data distribution of Nutrition Density after log transform as a sample
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
fig.suptitle(f"Distribution of Nutrition Density")

# Histogram
sns.histplot(selected_features["Nutrition Density"], kde=True, ax=axes[0])
axes[0].set_ylabel("Frequency")
# Box PLot
sns.boxplot(x=selected_features["Nutrition Density"], ax=axes[1])

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Normalization

# %%
normalizer = RobustScaler()

data_normalized = normalizer.fit_transform(selected_features)

data_normalized = pd.DataFrame(data_normalized)

data_normalized.head()

# %%
data_normalized.describe().T

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
fig.suptitle(f"Distribution of Calric Value")

# Histogram
sns.histplot(data_normalized.iloc[0], kde=True, ax=axes[0])
axes[0].set_ylabel("Frequency")
# Box PLot
sns.boxplot(x=data_normalized.iloc[0], ax=axes[1])

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
fig.suptitle(f"Distribution of Nutrition Density")

# Histogram
sns.histplot(data_normalized.iloc[25], kde=True, ax=axes[0])
axes[0].set_ylabel("Frequency")
# Box PLot
sns.boxplot(x=data_normalized.iloc[25], ax=axes[1])

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Data

# %%
# Feature
X = data_normalized.iloc[:, :-1]

# Target
y = data_normalized.iloc[:, -1]

# %% [markdown]
# ### Split Data

# %%
TEST_SIZE_1 = 0.1
TEST_SIZE_2 = 0.2

# %%
## Split Function
def split_data(X, y, TEST_SIZE):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y,
                                                        test_size=TEST_SIZE,
                                                        random_state=123)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                    test_size=0.5,
                                                    random_state=123)
    return X_train, X_val, X_test, y_train, y_val, y_test

# %%
# Split data into data_train, data_val, and data_test
## TEST_SIZE_1
X_train_1, X_val_1, X_test_1, y_train_1, y_val_1, y_test_1 = split_data(X, y, TEST_SIZE_1)

## TEST_SIZE_2
X_train_2, X_val_2, X_test_2, y_train_2, y_val_2, y_test_2 = split_data(X, y, TEST_SIZE_2)

# %% [markdown]
# ## Modeling

# %%
num_of_features = 25

# %% [markdown]
# ### Scheme 1

# %% [markdown]
# #### Model Design

# %%
# Model Parameter Config 1
OPTIMIZER_1 = RMSprop()
LOSS_1 = "mse"
METRIC_1 = "mae"
OUTPUT_ACTIVATION_1 = "linear"
DROPOUT_1 = 0.3

# %%
input_shape = num_of_features

model_1 = tf.keras.models.Sequential([
    Input(shape=(input_shape,)),
    Dense(units= 64, activation="relu"),
    BatchNormalization(),
    Dropout(DROPOUT_1),
    Dense(units= 32, activation="relu"),
    Dense(units=16, activation="relu"),
    Dense(units=1, activation=OUTPUT_ACTIVATION_1)
])

model_1.compile(
    optimizer=OPTIMIZER_1,
    loss=LOSS_1,
    metrics=[METRIC_1]
)

model_1.summary()

# %% [markdown]
# #### Training

# %%
# Training Parameters Config 1
EPOCHS = 100
BATCH = 128

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_mae',
    patience=20,
    restore_best_weights=True
)

# %%
# Train Model
history_1 = model_1.fit(
    X_train_1,
    y_train_1,
    validation_data=(X_val_1, y_val_1),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=[early_stopping]
)

# %% [markdown]
# ### Scheme 2

# %% [markdown]
# #### Model Design

# %%
# Model Parameter Config 2
OPTIMIZER_2 = Adam()
LOSS_2 = "huber"
METRIC_2 = ["mae", "mse"]
OUTPUT_ACTIVATION_2 = "linear"
DROPOUT_2 = 0.3

# %%
input_shape = num_of_features

model_2 = tf.keras.models.Sequential([
    Input(shape=(input_shape,)),
    Dense(units= 64, activation="relu"),
    BatchNormalization(),
    Dropout(DROPOUT_2),
    Dense(units= 32, activation="relu"),
    Dense(units=1, activation=OUTPUT_ACTIVATION_2)
])

model_2.compile(
    optimizer=OPTIMIZER_2,
    loss=LOSS_2,
    metrics=[METRIC_2]
)

model_2.summary()

# %% [markdown]
# #### Training

# %%
# Training Parameters Config 2
EPOCHS_2 = 300
BATCH_2 = 128

# Callbacks
early_stopping_2 = EarlyStopping(
    monitor='val_mae',
    patience=20,
    restore_best_weights=True
)

# %%
# Train Model
history_2 = model_2.fit(
    X_train_2,
    y_train_2,
    validation_data=(X_val_2, y_val_2),
    epochs=EPOCHS_2,
    batch_size=BATCH_2,
    callbacks=[early_stopping_2]
)

# %% [markdown]
# ## Evaluation

# %% [markdown]
# ### Scheme 1

# %%
plt.figure(figsize=(12, 5))

# Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_1.history['loss'], label='Training Loss')
plt.plot(history_1.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# MAE
plt.subplot(1, 2, 2)
plt.plot(history_1.history['mae'], label='Training MAE')
plt.plot(history_1.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error (MAE)')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# %%
loss, mae = model_1.evaluate(X_val_1, y_val_1)

print(f"Loss Model 1: {loss:.4f} \nMAE Model 1: {mae:4f}")

# %%
# Predict
y_pred_1 = model_1.predict(X_test_1)

mse_1 = mean_squared_error(y_test_1, y_pred_1)
mae_1 = mean_absolute_error(y_test_1, y_pred_1)
r2_1 = r2_score(y_test_1, y_pred_1)

# Recap
evaluasi_model_1 = pd.DataFrame({
    "Model Name": ["Model 1", "Model 1", "Model 1"],
    "Model Version": ["v1.0", "v1.0", "v1.0"],
    "Metrics": ["MSE", "MAE", "R-squared"],
    "Values": [mse_1, mae_1, r2_1]
})

model_1_evaluation = evaluasi_model_1.round(4)
model_1_evaluation

# %% [markdown]
# ### Scheme 2

# %%
plt.figure(figsize=(18, 5))

# Loss

plt.subplot(1, 3, 1)
plt.plot(history_2.history['loss'], label='Training Loss')
plt.plot(history_2.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# MAE
plt.subplot(1, 3, 2)
plt.plot(history_2.history['mae'], label='Training MAE')
plt.plot(history_2.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error (MAE)')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

# MSE
plt.subplot(1, 3, 3)
plt.plot(history_2.history['mse'], label='Training MSE')
plt.plot(history_2.history['val_mse'], label='Validation MSE')
plt.title('Model Mean Squared Error (MSE)')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.tight_layout()
plt.show()

# %%
loss_2, mae_2, mse_2 = model_2.evaluate(X_val_2, y_val_2)

print(f"Loss Model 2: {loss_2:.4f} \nMAE Model 2: {mae_2:4f} \nMSE Model 2: {mse_2:4f}")

# %%
# Predict
y_pred_2 = model_2.predict(X_test_2)

mse_2 = mean_squared_error(y_test_2, y_pred_2)
mae_2 = mean_absolute_error(y_test_2, y_pred_2)
r2_2 = r2_score(y_test_2, y_pred_2)

# Recap
evaluasi_model_2 = pd.DataFrame({
    "Model Name": ["Model 2", "Model 2", "Model 2"],
    "Model Version": ["v1.0", "v1.0", "v1.0"],
    "Metrics": ["MSE", "MAE", "R-squared"],
    "Values": [mse_2, mae_2, r2_2]
})

model_2_evaluation = evaluasi_model_2.round(4)
model_2_evaluation

# %% [markdown]
# ### Model Comparison

# %%
model_comparison = pd.concat([model_1_evaluation, model_2_evaluation], ignore_index=True)

model_comparison = model_comparison.pivot_table(index='Metrics', columns='Model Name', values='Values')

model_comparison

# %%
plt.figure(figsize=(18, 8))

# Model 1
plt.subplot(1, 2, 1)

# Actual Value
plt.scatter(y_test_1, y_test_1, color='blue', alpha=0.6, label='Actual Values')

# Predicted Value
plt.scatter(y_test_1, y_pred_1, color='orange', alpha=0.6, label='Predicted Values')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Model 1: Actual vs. Predicted Values')
plt.legend()  
plt.grid()

# Model 1
plt.subplot(1, 2, 2)

# Actual Value
plt.scatter(y_test_2, y_test_2, color='blue', alpha=0.6, label='Actual Values')

# Predicted Value
plt.scatter(y_test_2, y_pred_2, color='orange', alpha=0.6, label='Predicted Values')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Model 2: Actual vs. Predicted Values')
plt.legend()  
plt.grid()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Inference

# %% [markdown]
# ### Create Dummy Data

# %%
# Create Dummy Data
random_st = 123
np.random.seed(random_st)

dummy_data = pd.DataFrame({
    col: np.random.uniform(merged_data[col].min(), merged_data[col].median(), size=100)
    for col in merged_data.select_dtypes("number").columns[:-1]
})

dummy_data.info()

# %%
dummy_data.round(2).head()

# %% [markdown]
# ### Preprocessing Dummy Data

# %%
columns_to_eliminate[1:]

# %%
dummy_data = dummy_data.drop(columns=columns_to_eliminate[1:], axis=1)

# %%
print(f"Number of columns: {dummy_data.shape[1]}")

# %% [markdown]
# ### Predict Nutrition Density

# %%
# Use Model 1 as the best trained model
nutrition_density_pred = model_1.predict(dummy_data)
dummy_data["Nutrition Density"] = nutrition_density_pred

# %%
dummy_data.head()

# %%
dummy_data.describe().T

# %%
dummy_data.to_csv('predictions_of_dummy_data.csv', index=False)

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Histogram 
sns.histplot(dummy_data["Nutrition Density"], bins=20, color='skyblue', edgecolor='black', kde=True, ax=axes[0])
axes[0].set_title('Histogram of Predicted Nutrition Density')
axes[0].set_xlabel('Predicted Nutrition Density')
axes[0].set_ylabel('Frequency')
axes[0].grid(axis='y', alpha=0.75)

# Boxplot
sns.boxplot(x=dummy_data["Nutrition Density"], ax=axes[1], color='lightgreen')
axes[1].set_title('Boxplot of Predicted Nutrition Density')
axes[1].set_xlabel('Predicted Nutrition Density')
axes[1].grid(axis='y', alpha=0.75)


plt.tight_layout()
plt.show()

# %% [markdown]
# ## Conclusion

# %% [markdown]
# 1. The model is regression model because it aims to predict a continuous outcome (Nutrition Density Values) based on various input features (nutrient values). The prediction is estimating nutrient density values, which inherently involves predicting numerical quantities.
# 
# 2. All data have skewed distribution and outliers, whereas the neural network model is more suitable for normally distributed data. Hence, the data needs to transformed to logarithmic values and then normalized. The logarithmic transformation helps stabilize variance and reduce the effect of outliers by compressing the range of the data. Data normalization helps in adjusting the features scales, which is crucial for neural network as they are sensitive to the scale of input features. It will change the data to be normally distributed or close to it. By pre-processing the data this way, we can enhance the model's ability to learn meaningful patterns, leading to better performance and generalization on unseen data.
# 
# 3. Based on model evaluation, architecture design of Model 1 is better than Model 2. This is drawn from comparing metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared values between the two models. Model 1 have shown lower error rates in predicting nutrition density. The choice of architecture such as the number of layers and nodes, loss functions, and optimization techniques used in Model 1, likely contributed to its performance. In addition, there are recommendation to develop model with superior perfomance as follows:
# 
#     * Further refinement of feature selection may enhance model performance. Identifying key features that significantly impact nutrition density could lead to improved predictive accuracy.
#     * Explore different architectures and hyperparameter settings then identify the optimal configuration for the model.
#     * Implementing regularization techniques such as dropout, L1/L2 regularization, or batch normalization can help prevent overfitting in complex model.

# %% [markdown]
# ## Development Opportunity

# %% [markdown]
# __1. Meal Reccomendation System__.
# By integrating this dataset with broader dietary data, machine learning models can recommend dietary adjustments to individuals. For instance, a recommendation system could suggest lower-calorie or lower-sugar food alternatives to users looking to reduce their calorie intake but who still want to get high nutrition. Machine learning models can integrate food consumption data into broader dietary tracking tools used in fitness and health apps, providing users with insights for their dietary meal plan.
# 
# __2. Predictive Modeling for Health Impacts__.
# With sufficient data linking meal consumption to health outcomes, predictive models could forecast health impacts based on meal consumption patterns. This could be particularly useful public health.

# %% [markdown]
# ## References

# %% [markdown]
# 1. Alfarisi, B. I., et al. (2023). *Mengungkap Kesehatan Melalui Angka: Prediksi Malnutrisi Melalui Penilaian Status Gizi dan Asupan Makronutrien.* Prosiding SNPPM-5, 299-311.
# 2. Bouma, S. (2017). *Diagnosing Pediatric Malnutrition: Paradigm Shifts of Etiology-Related Definitions and Appraisal of the Indicators.* Nutrition in Clinical Practice, 32(1), 52–67.
# 3. Cakrawala. (2024). *Apa itu Neural Network? Ini Pengertian, Konsep, dan Contohnya.* Retrieved October 31, 2024, from [https://www.cakrawala.ac.id/berita/neural-network-adalah](https://www.cakrawala.ac.id/berita/neural-network-adalah).
# 4. Cederholm, T., et al. (2019). *GLIM criteria for the diagnosis of malnutrition – A consensus report from the global clinical nutrition community.* Journal of Cachexia, Sarcopenia and Muscle, 10(1), 207–217.
# 5. European Food Information Council (EUFIC). (2021). *What is nutrient density?* Retrieved October 31, 2024, from [https://www.eufic.org/en/understanding-science/article/what-is-nutrient-density](https://www.eufic.org/en/understanding-science/article/what-is-nutrient-density).
# 6. Khan, D. S. A., et al. (2022). *Nutritional Status and Dietary Intake of School-Age Children and Early Adolescents: Systematic Review in a Developing Country and Lessons for the Global Perspective.* Frontiers in Nutrition, 8(February).
# 7. Ministry of Health of the Republic of Indonesia. (2018). Situasi Balita Pendek (Stunting) di Indonesia.
# 8. RevoU. (2024). *Apa itu Neural Network.* Retrieved October 31, 2024, from [https://revou.co/kosakata/neural-network](https://revou.co/kosakata/neural-network).
# 9. Rinninella, E., et al. (2017). *Clinical tools to assess nutritional risk and malnutrition in hospitalized children and adolescents.* European Review for Medical and Pharmacological Sciences, 21(11), 2690–2701.
# 10. Simbolon, D. (2013). *Model Prediksi Indeks Massa Tubuh Remaja Berdasarkan Riwayat Lahir dan Status Gizi Anak.* Kesmas, 8(1), 19–27.
# 11. Yamantri, A. B., & Rifa’i, A. A. (2024). *Penerapan Algoritma C4.5 Untuk Prediksi Faktor Risiko Obesitas Pada Penduduk Dewasa.* Jurnal Komputer Antartika, 2(3), 118–125.
# 12. Zianka, I. D., et al. (2024). *The Design Android Application Nutrition Calculation to Prevent Stunting with CNN Method in Jakarta.* MALCOM: Indonesian Journal of Machine Learning and Computer Science, 4, 99–107.


