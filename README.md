Steel Factory ML-Driven Energy & Emissions Analysis Academic Project, 2025
This repository contains the code and report for an academic project focused on applying machine learning techniques to analyze energy usage and emissions data in a steel factory setting. The project demonstrates a complete data science pipeline—from data cleaning and exploratory analysis to predictive modeling, association rule mining, and clustering.

Overview
Data Cleaning & Preprocessing:
Removes duplicates, fixes date formats, and applies one-hot encoding to transform categorical variables for modeling.

Exploratory Data Analysis (EDA):
Uses histograms, boxplots, and scatter plots to visualize energy usage, CO₂ emissions, and power factors.

Predictive Modeling:

Naïve Bayes, Decision Tree, SVM:
These models classify energy usage levels based on engineered features.

Models are evaluated using accuracy, F1-score, and confusion matrices.

Association Rules Analysis:
Applies the Apriori algorithm to mine frequent itemsets and generate association rules, identifying key relationships between energy consumption and emissions.

Clustering Analysis:
Uses DBSCAN with PCA for dimensionality reduction to reveal natural clusters within the data.

Repository Structure
bash
Copy
.
├── README.md            # Project overview and instructions
├── Takehome_Midterm_BSAN_6070_Lestyk.py  # Main Python script containing the complete workflow
└── Report/              # Folder containing the full project report
    └── Takehome Midterm.docx
Getting Started
Prerequisites
Ensure you have the following installed:

Python 3.7 or higher

pip (Python package installer)

Dependencies
Install the required Python packages using the following command:

bash
Copy
pip install pandas matplotlib seaborn scikit-learn mlxtend
(If you are missing any dependencies, refer to the top of the Python script for additional required libraries.)

Running the Code
Clone the Repository:

bash
Copy
git clone https://github.com/yourusername/steel-factory-ml-analysis.git
cd steel-factory-ml-analysis
Execute the Python Script:

Run the main script:

bash
Copy
python Takehome_Midterm_BSAN_6070_Lestyk.py
The script will:

Clean and preprocess the data.

Generate exploratory plots.

Train and evaluate machine learning models.

Perform association rule mining.

Execute clustering analysis and visualize the clusters.

View the Report:

Check out the detailed report in the Report folder for additional insights and documentation regarding the project's methodology and results.

Project Report
The project report is contained in the file Takehome Midterm.docx within the Report folder. It outlines the following:

Business context and data source.

Detailed methodology for each machine learning technique used.

Evaluation metrics and comparison of model performances.

Business insights derived from the analysis.

Challenges and reflections on the methods applied.

