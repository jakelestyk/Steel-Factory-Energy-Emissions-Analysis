# ----------------------------
# Section 1: Data Cleaning and Preprocessing
# ----------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load the raw steel industry data
file_path = r"C:\Users\jakel\Downloads\BSAN 6070 Intro to ML\Steel_industry_data.csv"
df = pd.read_csv(file_path)

# remove duplicate rows (just to be safe)
df = df.drop_duplicates()

# convert date to datetime and drop invalid dates
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['date'])

# extract week number and day name (for time series paneling)
df['Wk'] = df['date'].dt.isocalendar().week
df['Day'] = df['date'].dt.day_name()

# create a column to indicate weekends (1 if Saturday or Sunday, else 0)
df['WeekStatus_Weekend'] = df['Day'].isin(['Saturday', 'Sunday']).astype(int)

# one-hot encode the Day and Load_Type columns to convert them into numeric format
df_encoded = pd.get_dummies(df, columns=['Day', 'Load_Type'], drop_first=False)

# drop the original date column since it's not needed for modeling
df_encoded = df_encoded.drop(columns=['date'])

# reorder columns: start with Wk, then all one-hot encoded Day columns, then the rest
day_cols = [col for col in df_encoded.columns if col.startswith('Day_')]
other_cols = [col for col in df_encoded.columns if col not in ['Wk'] + day_cols]
ordered_cols = ['Wk'] + day_cols + other_cols
df_panel = df_encoded[ordered_cols]

# save the cleaned, panel-formatted data
output_path = r"C:\Users\jakel\Downloads\BSAN 6070 Intro to ML\Panel_Data_ModelReady.csv"
df_panel.to_csv(output_path, index=False)
print(f"Panel data saved to: {output_path}")

# reload the cleaned panel data so everything uses the processed file
df_panel = pd.read_csv(output_path)
df_viz = df_panel.copy()

# ----------------------------
# Section 2: Descriptive Statistics and Visualizations
# ----------------------------
# basic descriptive stats for key features
key_features = ['Usage_kWh', 'CO2(tCO2)', 'Lagging_Current_Power_Factor']
descriptive_stats = df_panel[key_features].describe()
print("Descriptive Statistics:\n", descriptive_stats)

# plot histogram for energy usage
plt.figure(figsize=(9, 5))
sns.histplot(df_panel['Usage_kWh'], bins=30, kde=True, color='cornflowerblue', edgecolor='black')
plt.title('Histogram of Energy Usage (Usage_kWh)')
plt.xlabel('Usage_kWh')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# boxplot for CO2 emissions
plt.figure(figsize=(9, 5))
sns.boxplot(x=df_panel['CO2(tCO2)'], color='lightgreen')
plt.title('Boxplot of CO2 Emissions')
plt.xlabel('CO2(tCO2)')
plt.tight_layout()
plt.show()

# scatter plot for lagging reactive power vs energy usage
plt.figure(figsize=(9, 5))
sns.scatterplot(x='Lagging_Current_Reactive.Power_kVarh', y='Usage_kWh',
                data=df_panel, color='steelblue', alpha=0.6)
plt.title('Reactive Power vs Energy Usage')
plt.xlabel('Lagging Current Reactive Power (kVarh)')
plt.ylabel('Usage_kWh')
plt.tight_layout()
plt.show()

# ----------------------------
# Section 3: Predictive Modeling
# ----------------------------
# 3.1 Naïve Bayes Classification (used Lab 5 as a reference)
median_usage = df_panel['Usage_kWh'].median()
df_panel['Usage_Level'] = df_panel['Usage_kWh'].apply(lambda x: 1 if x > median_usage else 0)

# define features for modeling
features = ['Lagging_Current_Reactive.Power_kVarh',
            'Leading_Current_Reactive_Power_kVarh',
            'CO2(tCO2)',
            'Lagging_Current_Power_Factor',
            'Leading_Current_Power_Factor',
            'NSM',
            'WeekStatus_Weekend',
            'Load_Type_Maximum_Load',
            'Load_Type_Medium_Load']

X = df_panel[features]
y = df_panel['Usage_Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# build Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Naïve Bayes Accuracy: {accuracy:.2f}")
print("classification report:\n", classification_report(y_test, y_pred))

print("\nFeature Means (Train Set):")
feature_means = pd.DataFrame(X_train.mean(), columns=["Mean (Train Set)"])
print(feature_means)

# 3.2 Decision Tree Classification
dt_model = DecisionTreeClassifier(criterion='gini', max_depth=5,
                                   min_samples_split=10, random_state=2)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_preds)
print(f"\nDecision Tree Accuracy: {dt_accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, dt_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_preds))

plt.figure(figsize=(18, 8))
plot_tree(dt_model, feature_names=features, class_names=['Low Usage', 'High Usage'],
          filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree Structure (energy usage classification)')
plt.show()

print("\nCorrelation with Usage_kWh:")
corr_with_usage = df_panel.corr(numeric_only=True)['Usage_kWh'].sort_values(ascending=False)
print(corr_with_usage)

# baseline decision tree (max_depth=1) to check for overfitting
baseline_tree = DecisionTreeClassifier(max_depth=1, random_state=3)
baseline_tree.fit(X_train, y_train)
baseline_preds = baseline_tree.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_preds)
print(f"\nBaseline Tree Accuracy (max_depth=1): {baseline_accuracy:.2f}")

# 3.3 Support Vector Machines (SVM) Classification
from sklearn.svm import SVC

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=4)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_preds)
print(f"\nSVM Accuracy: {svm_accuracy:.2f}")
print("SVM Classification Report:\n", classification_report(y_test, svm_preds))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, svm_preds))

# 3.4 Comparative Analysis
from sklearn.metrics import f1_score

nb_f1 = f1_score(y_test, y_pred, average='macro')
dt_f1 = f1_score(y_test, dt_preds, average='macro')
svm_f1 = f1_score(y_test, svm_preds, average='macro')

comparison_data = {
    'Model': ['Naïve Bayes', 'Decision Tree', 'SVM'],
    'Accuracy': [accuracy, dt_accuracy, svm_accuracy],
    'Macro F1': [nb_f1, dt_f1, svm_f1]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n3.4 Comparative Analysis")
print(comparison_df)

# ----------------------------
# Section 4: Association Rules Analysis
# ----------------------------
# Frequent Itemset Mining & Association Rules (ChatGPT helped here)
from mlxtend.frequent_patterns import apriori, association_rules

features_to_binarize = ['Usage_kWh', 'CO2(tCO2)', 'Lagging_Current_Power_Factor']
df_assoc = df_panel.copy()
medians = {feat: df_assoc[feat].median() for feat in features_to_binarize}

def assign_label(row, feat, med):
    return f"High_{feat}" if row[feat] > med else f"Low_{feat}"

for feat in features_to_binarize:
    df_assoc[f'{feat}_Label'] = df_assoc.apply(lambda row: assign_label(row, feat, medians[feat]), axis=1)

items = []
for feat in features_to_binarize:
    items.append(f"High_{feat}")
    items.append(f"Low_{feat}")

transactions = pd.DataFrame(0, index=df_assoc.index, columns=items)
for feat in features_to_binarize:
    label_col = f'{feat}_Label'
    for item in [f"High_{feat}", f"Low_{feat}"]:
        transactions.loc[df_assoc[label_col] == item, item] = 1

frequent_itemsets = apriori(transactions, min_support=0.10, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules)

# Label rules on scatter plot (had some trouble with this part, ChatGPT helped)
rules['support'] = rules['support'].astype(float)
rules['confidence'] = rules['confidence'].astype(float)
rules['lift'] = rules['lift'].astype(float)

# identify two specific rules for clarity
rule1_idx = rules[
    (rules['antecedents'].apply(lambda x: 'High_Usage_kWh' in x)) &
    (rules['consequents'].apply(lambda x: 'High_CO2(tCO2)' in x))
].index

rule2_idx = rules[
    (rules['antecedents'].apply(lambda x: 'Low_Usage_kWh' in x)) &
    (rules['consequents'].apply(lambda x: 'Low_CO2(tCO2)' in x))
].index

plt.figure(figsize=(10, 6))
scatter = plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter).set_label('Lift')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules: Support vs. Confidence (Colored by Lift)')

for i in rule1_idx:
    x = rules.loc[i, 'support']
    y = rules.loc[i, 'confidence']
    plt.scatter(x, y, color='red', edgecolor='black', s=150, marker='D', label='Rule 1')
    plt.text(x + 0.002, y + 0.002, "Rule 1: High Usage => High CO2", fontsize=9)

for i in rule2_idx:
    x = rules.loc[i, 'support']
    y = rules.loc[i, 'confidence']
    plt.scatter(x, y, color='blue', edgecolor='black', s=150, marker='D', label='Rule 2')
    plt.text(x + 0.002, y + 0.002, "Rule 2: Low Usage => Low CO2", fontsize=9)

handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys(), loc='lower right')
plt.show()

# ----------------------------
# Section 5: Clustering Analysis with DBSCAN
# ----------------------------
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# clustering features (had some trouble here; got help from ChatGPT)
clustering_features = ['Usage_kWh', 'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'NSM']
X_cluster = df_panel[clustering_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels = dbscan.fit_predict(X_scaled)
df_panel['DBSCAN_Cluster'] = cluster_labels

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                hue=cluster_labels, palette='Set2', s=50, alpha=0.8)
plt.title("DBSCAN Clustering (PCA-Reduced)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

print("Cluster Counts:\n", df_panel['DBSCAN_Cluster'].value_counts())
