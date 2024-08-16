# WellnessAI-Interactive-Health-Risk-Analysis

## Contents:
- [Objective](#objective)
- [Data Overview](#data-overview)
- [Installation](#installation)
    - Cloning the Repository
    - Installing the Packages
    - Running the file
    - Running Streamlit UI

### Objective

The objective of this project is to develop and evaluate machine learning models for predicting diabetes based on a comprehensive dataset. The goal is to build accurate predictive models that can help in early diagnosis and management of diabetes by analyzing various health and demographic factors.

### Data Overview
- High Blood Pressure (high_bp): Represents the level of high blood pressure, where 0.0 indicates no high blood pressure, and 1.0 represents a very high level.

- High Cholesterol (high_chol): Represents the level of high cholesterol, where 0.0 indicates no high cholesterol, and 1.0 represents a very high level.

- Cholesterol Check (chol_check): Indicates whether the individual has undergone a cholesterol check.

- Height (height): The height of the individual in centimeters.

- Weight (weight): The weight of the individual in kilograms.

- Body Mass Index (BMI) (bmi): The Body Mass Index calculated as (weight / height^2) * 10000, estimating body fat.

- Smoking Frequency (smoker): Represents the smoking habits of the individual.

- Stroke (stroke): Indicates whether the individual has ever had a stroke.

- Heart Disease (heart_disease): Indicates whether the individual has ever had heart disease or an attack.

- Physical Activity (phys_activity): Represents the level of physical activity of the individual.

- Fruit Consumption (fruits): Frequency of fruit consumption by the individual.

- Vegetable Consumption (veggies): Frequency of vegetable consumption by the individual.

- Heavy Alcohol Consumption (hvy_alcohol): Indicates the level of heavy alcohol consumption.

- Avoidance of Doctor Due to Cost (no_doc_cost): Indicates how often the individual avoids seeing a doctor due to cost.

- General Health (gen_hlth): A self-assessment of general health, rated on a scale from 1 to 5.

- Mental Health (ment_hlth): The number of mentally unhealthy days experienced in a month.

- Physical Health (phys_hlth): The number of physically unhealthy days experienced in a month.

- Difficulty Walking (diff_walk): Indicates whether the individual has difficulty walking.

- Sex (sex): Gender of the individual.

### Installation

##### 1. Clone this repository.
```bash
git clone https://github.com/RawatRahul14/WellnessAI-Interactive-Health-Risk-Analysis.git
```

##### 2. Installing the Packages
```bash
pip install -r requirements.txt
```

##### 3. Running the File

```bash
python main.py
```

##### 4. Running the Streamlit UI

```bash
python app.py
```