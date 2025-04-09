Steel Factory ML-Driven Energy & Emissions Analysis Academic Project, 2025:
This repository contains the code and report for an academic project focused on applying machine learning techniques to analyze energy usage and emissions data in a steel factory setting. The project demonstrates a complete data science pipeline—from data cleaning and exploratory analysis to predictive modeling, association rule mining, and clustering.

Overview:
• Data Cleaning & Preprocessing: Removes duplicates, fixes date formats, and applies one-hot encoding to transform categorical variables for modeling.
• Exploratory Data Analysis (EDA): Uses histograms, boxplots, and scatter plots to visualize energy usage, CO₂ emissions, and power factors.
• Predictive Modeling: Implements Naïve Bayes, Decision Tree, and SVM to classify energy usage levels based on engineered features. Models are evaluated using accuracy, F1-score, and confusion matrices.
• Association Rules Analysis: Applies the Apriori algorithm to mine frequent itemsets and generate association rules, identifying key relationships between energy consumption and emissions.
• Clustering Analysis: Uses DBSCAN with PCA for dimensionality reduction to reveal natural clusters within the data.

Repository Structure:
├── README.md: Project overview and instructions.
├── Takehome_Midterm_BSAN_6070_Lestyk.py: Main Python script containing the complete workflow.
└── Report/ Takehome Midterm.docx: Contains the full project report with detailed methodology and analysis.

Getting Started:
Prerequisites: Python 3.7 or higher and pip (Python package installer) are required.
Dependencies: Install the necessary packages with:
    pip install pandas matplotlib seaborn scikit-learn mlxtend
Running the Code: 
    • Clone the repository: git clone https://github.com/yourusername/steel-factory-ml-analysis.git && cd steel-factory-ml-analysis
    • Execute the script: python Takehome_Midterm_BSAN_6070_Lestyk.py
    • The script will clean and preprocess the data, generate exploratory plots, train and evaluate machine learning models, perform association rule mining, and execute clustering analysis with visualization.
    • View the report in the Report folder for more details.

