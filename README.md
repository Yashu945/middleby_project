# Business Analysis Report on Supermarket Transaction Data

## Executive Summary
The transaction data analysis of the supermarket has been carried out in this report based on two years of data. The goal was to purify, transmute, and methodologically dissect the data to reach usable commercial knowledge to support decisions in marketing TCO and improve effectiveness. Various methods of data analysis, such as machine learning, were applied to explore patterns and relationships between sales, promotions, and customers’ buying habits.

---

## Introduction

### Background
Retail market competition can be explained using a rational choice model where customer behavior, inventories, and promotions are critical success factors. Using detailed transactional data from several supermarkets, this analysis evaluates strengths and weaknesses to pinpoint areas for improvement and enhance customer experience.

### Scope
The scope of this analysis included:
- Data Cleaning and Preparation
- Exploratory Data Analysis (EDA)
- Predictive Modeling
- Business Insight Generation

### Data Description
The analysis utilized four key datasets:
- **Items.csv**: Contains item details, descriptions, categories, brands, and sizes.
- **Sales.csv**: Includes the date and time records of the number of items sold and their corresponding values.
- **Promotion.csv**: Describes general promotional activities as per Vidler (2010), associated with tangible products and places.
- **Supermarkets.csv**: Stores metadata and coordinates of supermarkets.

---

## Methodology

### Data Preparation
1. **Loading and Inspection**: Used the Python toolkit `pandas` to load and inspect datasets for completeness, missing values, and data types.
2. **Cleaning**: 
   - Cleaned the promotion dataset to eliminate duplicate records and ensure data integrity.
3. **Transformation and Integration**:
   - Standardized column names for consistency across datasets.
   - Applied one-hot encoding to categorical variables for machine learning compatibility.
   - Merged datasets on relevant keys to create a unified analytical framework.

### Exploratory Data Analysis (EDA)
- **Sales Distribution Analysis**: Used histograms to understand the distribution of transaction values and identify outliers.
- **Feature Correlation**: Employed correlation matrices to reveal relationships between variables, especially focusing on how promotional activities impact sales.

### Predictive Modeling
- **Model Selection**: Selected a robust model combining the strengths of Random Forest to handle both linear and non-linear data.
- **Training and Testing**: Split the data into 80% training and 20% testing to validate model predictions.
- **Model Evaluation**: Used Mean Squared Error (MSE) to measure error and R² Score to indicate explained variance.

---

## Results

### Key Findings
- **Promotional Impact**: Identified specific promotional types with high correlations to increased sales volumes.
- **Regional Sales Trends**: Observed non-uniform sales quantities across provinces, indicating the need for region-specific strategies.

### Visualizations
- **Heatmaps**: Showed feature correlations.
- **Time-Series Plots**: Illustrated sales trends over time.
- **Graphs and Charts**: Provided clear, well-labeled visual representations of key insights.

---

## Discussion

### Challenges
- Data integration faced difficulties due to format inconsistencies and missing records.
- Feature selection and hyperparameter tuning were critical to ensure robust predictive modeling.

### Learnings
- Reaffirmed the importance of data quality for accurate analysis.
- Gained insights into the efficiency of different promotional methods for future investment decisions.

---

## Conclusion and Recommendations
1. Conduct region-specific sales promotions based on identified sales trends.
2. Incorporate more customer demographic information to optimize group identification and develop detailed marketing strategies.
3. Personalize marketing efforts by further integrating demographic data.

---

## Future Work
- Explore advanced machine learning algorithms, such as neural networks, for deeper predictive insights.
- Consider real-time data integration for dynamic pricing and inventory management.
