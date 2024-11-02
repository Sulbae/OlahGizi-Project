# Project Report
# Basic Neural Network Model for Predicting Nutrient Density of Food Consumption
## __Background__ 
Adequate nutritional status is crucial for human growth and survival. Nutritional status can be assessed by evaluating individual-specific nutritional requirements and intake. An imbalance between nutritional needs and intake may lead to either deficiency or excess, both of which negatively impact health. This condition is commonly referred to as malnutrition (Bouma, 2017; Rinninella et al., 2017). According to the World Health Organization (WHO), malnutrition can occur due to an imbalance in nutrient intake, which may affect health status, disrupt food digestion, or impair nutrient absorption (Khan et al., 2022). Malnutrition is not limited to undernutrition; it also encompasses a broader scope including macronutrient and micronutrient imbalances, obesity, cachexia, sarcopenia, and malnourishment (Cederholm et al., 2019).

In Indonesia, malnutrition—including both undernutrition and obesity—remains a serious issue. The 2018 Global Nutrition Report revealed that one in five child deaths globally is associated with poor dietary intake. Based on data from Indonesia's 2018 Basic Health Research, malnutrition cases include 30.8% of stunting, 3.5% of severe malnutrition, and other related conditions such as obesity (Zianka et al., 2024). Nutritional issues in children, if left unaddressed, may continue into adolescence and adulthood (Simbolon, 2013). Such conditions have far-reaching negative implications, as children with inadequate nutritional intake may experience delayed brain development and lower intelligence. Ultimately, this may hinder national economic growth and even increase poverty. The potential economic losses due to malnutrition are significant and may create a financial burden for the healthcare system in the future (Kemenkes RI, 2018).

Obesity, on the other hand, affects over a quarter of the adult population in Indonesia. The high prevalence of obesity among adults indicates poor dietary patterns. Consuming high-calorie, high-fat, and high-sugar foods without sufficient physical activity is a major factor contributing to obesity (Yamantri et al., 2024).

One approach to addressing malnutrition is by providing widespread education on the importance of balanced nutrition, particularly for children. This can be achieved quickly and cost-effectively by leveraging digital technology for information distribution. Additionally, advancements in artificial intelligence (AI), such as intelligent agent assistants, can help users access information quickly and accurately. For this reason, the authors have developed a machine learning model to predict the nutrition density of commonly consumed foods and beverages. Nutrition density is a metric that reflects the amount of nutrients provided per unit of energy or calories. This concept is designed to evaluate the nutritional quality of a food item based on its nutrient content relative to its caloric value, thereby helping individuals choose nutrient-dense foods without excess energy or unnecessary calories (EUFIC, 2021). The model aims to be applied to a nutritional balance information service that enables users to make informed food choices suited to their individual needs and conditions.

The machine learning model will be developed using Google’s TensorFlow framework, chosen for its flexibility and scalability, allowing the authors to build machine learning applications from small to large scales. The model will employ a Neural Network algorithm, selected for its capability to handle complex data and hidden layers that facilitate automatic feature extraction, thus recognizing correlations among nutritional attributes without requiring extensive manual data processing (RevoU, 2024; Cakrawala, 2024). This enables the model to deliver accurate predictions of nutrition density based on available nutritional data for various food items.

_Keywords: Malnutrition, Nutrition Density, Neural Network, TensorFlow_

## __Business Understanding__
### Problem Statements
  1. How would the model work in predicting nutrient density?
  2. How to carry out good data processing that fits the model architecture?
  3. What factors need to be considered to develop the best model?
### Goals
  1. Understanding how the neural network model works in predicting nutrition density.
  2. Processing raw data into clean data that is ready to be used to train neural network models.
  3. Designing the best architecture for the model.
### Solution Statements
  1. Determine the dataset to be used and the expected output so we can find out the right type of prediction architecture.
  2. Carrying out an iterative process that includes Exploratory Data Analysis to understand data characteristics and data transformation to adjust the data format to the neural network model architecture.
  3. Carrying out feature engineering stages to select features that most influence nutrition density values and hyperparameter tuning to optimize model performance.

## __Data Understanding__
### Dataset
Resouce: [Food Nutrition Dataset](https://www.kaggle.com/datasets/utsavdey1410/food-nutrition-dataset/data)

### Overview
The Comprehensive Nutritional Food Database provides detailed nutritional information for a wide range of food items commonly consumed around the world. This dataset aims to support dietary planning, nutritional analysis, and educational purposes by providing extensive data on the macro and micronutrient content of foods.

### Column Description
No | Column | Description
---|--------|------------
1 | Food | The name or type of the food item
2 | Caloric Value | Total energy provided by the food, typically measured in kilocalories(kcal) per 100 grams.
3 | Fat (in g) | Total amount of fats in grams per 100 grams, including the breakdowns that follow
4 | Saturated Fats (in g) | Amount of saturated fats (fats that typically raise the level of cholesterol in the blood) in grams per 100 grams.
5 | Monounsaturated Fats (in g) | Amount of monounsaturated fats (considered heart-healthy fats) in grams per 100 grams.
6 | Polyunsaturated Fats (in g) | Amount of polyunsaturated fats (include essential fats your body needs but can't produce itself) in grams per 100 grams.
7 | Carbohydrates (in g) | Total carbohydrates in grams per 100 grams, including sugars.
8 | Sugars (in g) | Total sugars in grams per 100 grams, a subset of carbohydrates.
9 | Protein (in g) | Total proteins in grams per 100 grams, essential for body repair and growth.
10 | Dietary Fiber (in g) | Fiber content in grams per 100 grams, important for digestive health.
11 | Cholesterol (in mg) | Cholesterol content in milligrams per 100 grams, pertinent for cardiovascular health.
12 | Sodium (in mg) | Sodium content in milligrams per 100 grams, crucial for fluid balance and nerve function.
13 | Water (in g) | Water content in grams per 100 grams, which affects the food's energy density.
14 | Vitamin A (in mg) | Amount of Vitamin A in micrograms per 100 grams, important for vision and immune functioning.
15 | Vitamin B1 (Thiamine)(in mg) | Essential for glucose metabolism.
16 | Vitamin B11 (Folic Acid)(in mg) | Crucial for cell function and tissue growth, particularly important in pregnancy.
17 | Vitamin B12(in mg) | Important for brain function and blood formation.
18 | Vitamin B2 (Riboflavin)(in mg) | Necessary for energy production, cell function, and fat metabolism.
19 | Vitamin B3 (Niacin)(in mg) | Support digestive system, skin, and nerves health.
20 | Vitamin B5 (Pantothenic Acid)(in mg) | Necessary for making blood cells and helps convert food into energy.
21 | Viatmin B6 (in mg): Important for normal brain development and keeping the nervous and immune system healthy.
22 | Vitamin C (in mg) | Important for the repair of all body tissues.
23 | Vitamin D (in mg) | Crucial for absorption of calcium, promoting bone growth and health.
24 | Vitamin E (in mg) | Acts as an antioxidant, helping to protect cells from the damage caused by free radicals.
25 | Vitamin K (in mg) | Necessary for blood clotting and bone health.
26 | Calcium (in mg) | Vital for building and maintaining strong bones and teeth.
27 | Copper (in mg) | Helps with the formation of collagen, increases the absorption of iron and plays a role in energy production.
28 | Iron (in mg) | Essential for the creation of red blood cells.
29 | Magnesium (in mg) | Important for many processes in the body including regulation of muscle and nerve function, blood sugar levels, and blood pressure and making protein, bone, and DNA.
30 | Manganese (in mg) | Involved in the formation of bones, blood clotting factors, and enzymes that play a role in fat and carbohydrate metabolism, calcium absorption, and blood sugar regulation.
31 | Phosphorus (in mg) | Helps with the formation of bones and teeth and is necessary for the body to make protein for the growth, maintenance, and repair of cells and tissues.
32 | Potassium (in mg) | Helps regulate fluid balance, muscle contractions, and nerve signals.
33 | Selenium (in mg) | Important for reproduction, thyroid gland function, DNA production, and protecting the body from damage caused by free radicals and from infection.
34 | Zinc (in mg) | Necessary for the immune system to properly function and plays a role in cell division, cell growth, wound healing, and the breakdown of carbohydrates.
35 | Nutrition Density | A metric indicating the nutrient richness of the food per calorie.

### Explore
According to data exploration, there is some basic information such as: 
* The dataset has 2395 rows and 35 columns.
* It consists of object data (columns "food") and numerical data (int64 for column "Caloric Value" and float64 for other columns).
* There is no missing value or duplicated data.
* All values of column "food" are unique.
* We need to equate the units for some columns (Fat, Saturated Fats, Monounsaturated Fats, Polyunsaturated Fats, Carbohydrates, Sugars, Protein, Dietary Fiber, and Water) to milligrams.

#### Data Conversion
We need to convert data unit that is in grams to milligrams by using the following formula:

`x (g) = x × 1000 (mg)`

#### Data Distribution
Most data don't have a normal distribution and have some outliers. Besides that, some data also have different value ranges.
For example, this is a visualization of Caloric Value and Nutrition Density data:

<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; margin-right: 10px;">
    <img src="https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/Chart/Caloric%20Value.png">
  </div>
  <div style="flex: 1; margin-left: 10px;">
    <img src="https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/Chart/Nutrition%20Density.png">
  </div>
</div>

These outliers are possible and normal because each food has a unique value. Those are valid data points. Outliers can represent real variations in the nutritional content of foods.

#### Correlation
<div style="display: flex; justify-content: center;">
  <div style="flex: 1; margin: 10px;">
    <img src="https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/Chart/Nutrient%20Correlation.png">
  </div>
</div>

According to the heatmap above, some nutrients have little contribution to the calculation of nutrition density such as Vitamin A, Vitamin B11, Vitamin B12, Vitamin D, Vitamin K, Copper, Manganese, and Selenium.

## __Data Preparation__
### Feature Engineering
As mentioned, some nutrients almost do not correlate with nutrition density calculation. Hence we need to eliminate it to reduce the features to be trained. If a feature has a correlation value close to zero to the target, this indicates that the feature does not have a significant linear relationship with the target and is unlikely to make a significant contribution to the prediction model.

### Data Normalization
As we know almost all of the data have a skewed distribution, different value ranges, and outliers. 
#### Log Transform
First, transform the data into logarithmic value using `np.log()`. It will handle the outliers and reduce the skewness. The logarithmic transformation helps stabilize variance and reduce the effect of outliers by compressing the range of the data. For example, the Nutrient Density values ​​change to the following:
<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; margin-right: 10px;">
    <img src="https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/Chart/Log%20Transform%20Caloric%20Value.png">
  </div>
  <div style="flex: 1; margin-left: 10px;">
    <img src="https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/Chart/Log%20Transform%20Nutrition%20Density.png">
  </div>
</div>

Second, normalize the data using `RobustScaler()` to transform the distribution to be as close to normal as possible. Data normalization helps in adjusting the feature scales, which is crucial for neural networks as they are sensitive to the scale of input features. It will change the data to be normally distributed or close to it.
<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; margin-right: 10px;">
    <img src="https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/Chart/Normalized%20Caloric%20Value.png">
  </div>
  <div style="flex: 1; margin-left: 10px;">
    <img src="https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/Chart/Normalized%20Nutrition%20Density.png">
  </div>
</div>

### Data Split
There are two data-splitting schemes as follows:
1. `TEST_SIZE_1 = 0.1`
2. `TEST_SIZE_2 = 0.2`

## __Modeling__
Parameter | Model 1 | Model 2 
-------|------------|--------
Optimizer | RMSprop | Adam
Loss | MSE | Huber
Output Activation | Linear | Linear
Batch Normalization | 1 | 1
Dropout | 0.3 | 0.3
Dense Layers | 4 | 3
Epochs | 100 | 300
Batch | 128 | 128

The model structure can be seen in the following diagram:


## __Training__
The model training stage consists of the following steps:
1. __Forward Pass__
   * _Input Data_: Nutritional Data (features) will be fed into the `input layer`.
   * _Prediction_: The model calculates the `weight` and then produces Nutrition Density as a prediction by calculating the `activation function` in every neuron.
2. __Loss__
  *  _Loss Caculating_: The model will calculate the `loss function` value to assess how well the model's predictions compare to the actual values.
  *  The loss function for regression can be expressed by the following formula:

$$
\text{Loss} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

  where:
- $$\( y_i \)$$ is the actual value,
- $$\( \hat{y}_i \)$$ is the model prediction,
- $$\( n \)$$ is the number of examples.

3. __Backward Pass__
  *  _Gradient Descent_: The `gradient` will be calculated from the loss function against the model weights. This process is carried out using the chain rule to calculate how much each weight contributes to the output error.
  *  _Weights Updating_: The `weights` are refined using `optimization` algorithms, such as `RMSProp()` and `Adam()`, to minimize the loss.
  *  The weight update at each iteration can be expressed by the following formula:

$$
w \gets w - \eta \frac{\partial \text{Loss}}{\partial w}
$$

  where:
- $$\( w \)$$ is the weight,
- $$\( \eta \)$$ is the learning rate,
- $$\( \frac{\partial \text{Loss}}{\partial w} \)$$ is the gradient of the loss function to the weight.

4. __Training Evaluation__
   *  _Validation_: At each `epoch` or every few epochs, the model is evaluated on the validation set. The loss function on the validation set is calculated to see how the model generalizes beyond the training data.

## __Evaluation__
This is the result of training on Model 1:
<div style="display: flex; justify-content: center;">
  <div style="flex: 1; margin: 10px;">
    <img src="https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/Chart/evaluation%20model%201.png">
  </div>
</div>

This is the result of training on Model 2:
<div style="display: flex; justify-content: center;">
  <div style="flex: 1; margin: 10px;">
    <img src="https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/Chart/evaluation%20model%202.png">
  </div>
</div>

### Model Comparison
Metrics | Model 1 | Model 2
--------|---------|--------
MAE | 0.0505 | 0.0741
MSE | 0.0042 | 0.0114
R-squared | 0.9917 | 0.9768

Model 1 has shown lower error rates in predicting nutrition density. These are comparisons of the model prediction to the actual data.

<div style="display: flex; justify-content: center;">
  <div style="flex: 1; margin: 10px;">
    <img src="https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/Chart/Prediction%20Model.png">
  </div>
</div>

## __Inference__
Nutrition density prediction is done using dummy data containing the values ​​of various nutrients contained in food like the original data.

The predictions were saved into a `.csv` file as follows:
[Prediction_of_dummy_data](https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/predictions_of_dummy_data.csv)

<div style="display: flex; justify-content: center;">
  <div style="flex: 1; margin: 10px;">
    <img src="https://raw.githubusercontent.com/Sulbae/OlahGizi-Project/refs/heads/main/Nutrition%20Prediction%20Model/Chart/Predicted%20Nutrition%20Density.png">
  </div>
</div>

## __Conclusion__
1. The model is a regression model because it aims to predict a continuous outcome (Nutrition Density Values) based on various input features (nutrient values). The prediction is estimating nutrient density values, which inherently involves predicting numerical quantities.

2. All data have skewed distribution and outliers, whereas the neural network model is more suitable for normally distributed data. Hence, the data needs to be transformed to logarithmic values and then normalized. By pre-processing the data this way, we can enhance the model's ability to learn meaningful patterns, leading to better performance and generalization on unseen data.

3. Based on the model evaluation, the architecture design of Model 1 is better than Model 2. This is drawn from comparing metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared values between the two models. The choice of architecture such as the number of layers and nodes, loss functions, and optimization techniques used in Model 1, likely contributed to its performance. In addition, there are recommendations to develop a model with superior performance as follows:

    * Further refinement of feature selection may enhance model performance. Identifying key features that significantly impact nutrition density could lead to improved predictive accuracy.
    * Explore different architectures and hyperparameter settings then identify the optimal configuration for the model.
    * Implementing regularization techniques such as dropout, L1/L2 regularization, or batch normalization can help prevent overfitting in complex models.

## __Development Opportunity__
__1. Meal Recommendation System__.
By integrating this dataset with broader dietary data, machine learning models can recommend dietary adjustments to individuals. For instance, a recommendation system could suggest lower-calorie or lower-sugar food alternatives to users looking to reduce their calorie intake but who still want to get high nutrition. Machine learning models can integrate food consumption data into broader dietary tracking tools used in fitness and health apps, providing users with insights into their dietary meal plans.

__2. Predictive Modeling for Health Impacts__.
With sufficient data linking meal consumption to health outcomes, predictive models could forecast health impacts based on meal consumption patterns. This could be particularly useful for public health.

## __References__
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
