<h1 align="center">TASK 1</h1>

## Business Analysis Report on Supermarket Transaction Data

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

<h1 align="center">TASK 02</h1>

# Comprehensive Analysis Report

## Executive Summary
This report provides a detailed investigation and discussion of two distinct projects solved using state-of-the-art machine learning methods. These projects target real-life business issues and the improvements made in business operations. The first project applies supervised learning to predict sales outcomes and assess promotional impacts in specific supermarket deals. The second project explores the feasibility of using reinforcement learning to solve challenging maze navigation problems, which serves as a stand-in for strategic planning and improvement challenges in specific environments. These projects demonstrate how data-driven methods and rationality can significantly impact business ventures and their underlying strategies.

## Task Overview

### Supervised Learning Task:
- **Objective**: The objective is to create a prognostic model to predict sales volumes and assess the impact of promotions. The analysis involves working with big data and turning the findings into actionable recommendations for improving marketing and sales strategies.

### Maze Navigation Task:
- **Objective**: To resolve tactical thinking and pathfinding issues by designing an intelligent system that employs reinforcement learning to navigate through a maze.

## Data Cleaning & Transformation

### Supervised Learning Model:
- **Cleaning Process**: The data was pre-processed with careful handling of missing values and other inconsistencies. Missing attribute values were imputed, and nominal data was categorized to ensure data utility.
- **Transformation Process**: The data was scaled to a normal scale equivalent, and qualitative data was converted into N+1 categories to ensure compatibility with machine learning algorithms.

### Maze Navigation Model:
- **Preprocessing**: The maze data was transformed into a grid format, where `1` represented barriers and `0` represented open paths, making it understandable for the learning algorithm.

## Supervised Learning Model

- **Problem Definition**: The goal was to identify the relationship between promotional variables and sales volume responses in multiple supermarkets. Additionally, sales cycle patterns were to be identified to predict peak sales periods.

- **Model Explanation**:
  - **Chosen Model**: The `RandomForestRegressor` was selected due to its versatility with different datasets and its ability to model non-linear problems with minimal parameter adjustments.
  - **Features Used**: Promotional flags like day of the week and repeat sales from prior promotions were chosen to assess their influence on purchase decisions.
  - **Training Process**: The model was trained using cross-validation to minimize overfitting, dividing the dataset into K parts and testing on each part.
  - **Evaluation of Metrics**: The model's performance was evaluated using Mean Squared Error (MSE) and R-Squared scores, which showed improvements over basic models.
  
- **Insights**: The model revealed key insights into the effectiveness of various promotional campaigns and their impact on sales, depending on the store’s location and type.

### Business Insights:
- The model helped identify the most effective promotions and the timing for their execution, aiding in stock management, promotional strategy development, and better overall business strategy formulation.

## Maze Model Design

- **Model Explanation**: The maze navigation model used Q-learning, a reinforcement learning technique well-suited for problems with clearly defined state and action spaces.
  
- **Reinforcement Learning Approach**: 
  - **Approach Details**: Both A3C and Q-learning were utilized, with rewards defined for reaching the goal, hitting obstacles, or normal movement. The exploration vs exploitation trade-off was a critical factor.
  
- **Training Results**: The model demonstrated strong learning, solving the maze in fewer attempts over successive episodes and with fewer trials.
  
- **Performance Analysis**: Analyzing the Q-table showed how the agent improved its pathfinding strategy by avoiding obstacles and reducing the path length to the goal.

## Challenges and Lessons Learned
- Several challenges were faced, such as handling high-dimensional data in the supervised learning model and tuning hyperparameters in the reinforcement learning model. The projects led to key discoveries about data quality and the effectiveness of reinforcement learning in solving complex spatial problems.

## Conclusion
The projects demonstrated that machine learning methods are feasible and useful for solving business problems. Predictive modeling proved well-suited for business environments, and AI can be effectively applied in strategic decision-making situations.



