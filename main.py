# libraries
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# set options for display of dataframes
pd.set_option('display.max_columns', None)


# initialize the data-frames (child and parent)
child_df = pd.read_csv("./Data/ChildData.csv")
parent_df = pd.read_csv("./Data/ParentData.csv")

# make copies for engineering, cleaning
p_df = parent_df
c_df = child_df

# Create a new DF consisting of two cols
x = p_df.loc[:, ['auth_mth', 'deposit_cnt']]

# Convert above result into a DF and then group by dates to get sum of deposits
x = pd.DataFrame(x)
x = x.groupby(["auth_mth"]).sum()
print(x.to_dict())

# Result of above grouping is initialized as a dictionary
mth_deposits = {'2020-03-01': 9.0, '2020-04-01': 5.0, '2020-05-01': 18.0, '2020-06-01': 87.0, '2020-07-01': 761.0,
                '2020-08-01': 514.0, '2020-09-01': 760.0, '2020-10-01': 628.0, '2020-11-01': 538.0, '2020-12-01': 560.0,
                '2021-01-01': 535.0, '2021-02-01': 477.0, '2021-03-01': 910.0, '2021-04-01': 1280.0,
                '2021-05-01': 1120.0, '2021-06-01': 1060.0, '2021-07-01': 1935.0, '2021-08-01': 2054.0}

# Plot the bar graph for Number of deposits per month
plt.figure(figsize=(15,8))
plt.bar(mth_deposits.keys(), mth_deposits.values(), color=['green'])
plt.ylabel("Sum of Deposits")
plt.xlabel("Month")
plt.title("Sum of Deposits by Month")
plt.xticks(rotation=45)
plt.show()

# Create a new DF consisting of two cols
x2 = p_df.loc[:, ['auth_mth', 'cnt_user_login']]

# Convert above result into a DF and then group by dates to get sum user logins
x2 = pd.DataFrame(x2)
x2 = x2.groupby(["auth_mth"]).sum()
print(x2.to_dict())

# result of above grouping is initialized as a dictionary
mth_logins = {'2020-03-01': 25.0, '2020-04-01': 47.0, '2020-05-01': 97.0, '2020-06-01': 302.0, '2020-07-01': 3946.0,
     '2020-08-01': 3356.0, '2020-09-01': 5420.0, '2020-10-01': 2332.0, '2020-11-01': 1832.0, '2020-12-01': 1766.0,
     '2021-01-01': 1626.0, '2021-02-01': 1303.0, '2021-03-01': 2909.0, '2021-04-01': 6084.0, '2021-05-01': 4349.0,
     '2021-06-01': 3766.0, '2021-07-01': 8680.0, '2021-08-01': 12458.0}

# Bar plot for Number of logins per month
plt.figure(figsize=(15,8))
plt.bar(mth_logins.keys(), mth_logins.values(), color=['green'])
plt.ylabel("Number of logins")
plt.xlabel("Month")
plt.title("Total Logins from Parents per Month")
plt.xticks(rotation=45)
plt.show()

# Below DF is the original df unchanged, just read from csv file
print(parent_df.describe())
print(parent_df.info())

# Imputing NULL values in the integer columns as 0
values = {"trxn_amt": 0, "deposit_cnt": 0, "transfer_cnt": 0, "deposit_amt": 0, "transfer_amt": 0}
p_df = p_df.fillna(value=values)

# Only take ("trxn_amt", "deposit_cnt", "transfer_cnt", "deposit_amt", "transfer_amt") as features for K-NN
p_df = p_df.iloc[:, 8:]
print(p_df.info(), p_df.describe())

# Scatter plot of the Raw data
plt.figure(figsize=(10,8))
plt.scatter(p_df['deposit_cnt'], p_df['deposit_amt'])
plt.xlabel("Deposit Count/month")
plt.ylabel("Deposit Amount/month")
plt.title("Parents: Relation between Amount and Number of deposits")
plt.show()

print('++++++++++++++++++++++++++++++++++++++++++++++++')
# Scaling/Normalizing the data
#scaler = StandardScaler()
scaler = MinMaxScaler()
p_df_std = scaler.fit_transform(p_df)
print(p_df_std)
print('++++++++++++++++++++++++++++++++++++++++++++++++')

# PCA for dimensionality reduction
pca = PCA()
pca.fit(p_df_std)

# PCA on non-scaled data-frame
# pca2 = PCA()
# pca2.fit(p_df)

# PCA with SCALING. Below snippet plots the cumulative variance against the number of PCA components.
# Generally, you want to select the number of components that allow you to achieve at least 80% of your
# data-set's variance
print(pca.explained_variance_ratio_)
print('++++++++++++++++++++++++++++++++++++++++++++++++')
plt.figure(figsize=(10,6))
plt.plot(range(1,6), pca.explained_variance_ratio_.cumsum(), marker='x', linestyle='--')
plt.ylabel("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.title("Observed Variance by Number of Features")
plt.show()

# PCA WIHTOUT SCALING
# print(pca2.explained_variance_ratio_)
# print('++++++++++++++++++++++++++++++++++++++++++++++++')
# plt.figure(figsize=(10,6))
# plt.plot(range(1,6), pca2.explained_variance_ratio_.cumsum(), marker='x', linestyle='--')
# plt.ylabel("Cumulative Explained Variance")
# plt.xlabel("Number of Components")
# plt.title("Observed Variance by Number of Features v.2")
# plt.show()

# From Variance graph, it is observed that 2-3 is the right number of components
pca = PCA(n_components=3)
pca.fit(p_df_std)
scores_pca = pca.transform(p_df_std)
print(scores_pca)

# K-means, to figure out the best number of clusters for our dataset, this loop runs KNN on the dataset
# with the number of clusters going from 1 to 30. We use the elbow technique to determine which number is best
wcss = []
for i in range(1, 30):
    kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state= 42)
    kmeans_pca.fit(scores_pca)
    wcss.append((kmeans_pca.inertia_))

# Elbow technique
plt.figure(figsize=(10,8))
plt.plot(range(1,30), wcss, marker='x', linestyle='--')
plt.xlabel("Number of Components")
plt.ylabel("WCSS (Within Cluster Sum of Squares)")
plt.title("WCSS vs Number of Clusters")
plt.show()

# Fit data with n=4 in KNN
kmeans_pca = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans_pca.fit(scores_pca)
print("----------------------")
print(kmeans_pca.cluster_centers_)
print("----------------------")

# Create new dataframe with components and clusters added.
df_p_pca_kmeans = pd.concat([p_df.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
df_p_pca_kmeans.columns.values[-3:] = ['Component 1', 'Component 2', 'Component 3']
df_p_pca_kmeans['Segment K-Means PCA'] = kmeans_pca.labels_
df_p_pca_kmeans['Segment'] = df_p_pca_kmeans['Segment K-Means PCA'].map({0: 'first',
                                                                         1: 'second', 2: 'third', 3: 'fourth'})
# Printing out Cluster information
print("----------------------")
df_cluster_1 = df_p_pca_kmeans.loc[df_p_pca_kmeans['Segment'] == 'first']
print(df_cluster_1.describe())
print("----------------------")
df_cluster_2 = df_p_pca_kmeans.loc[df_p_pca_kmeans['Segment'] == 'second']
print(df_cluster_2.describe())
print("----------------------")
df_cluster_3 = df_p_pca_kmeans.loc[df_p_pca_kmeans['Segment'] == 'third']
print(df_cluster_3.describe())
print("----------------------")
df_cluster_4 = df_p_pca_kmeans.loc[df_p_pca_kmeans['Segment'] == 'fourth']
print(df_cluster_4.describe())
print("----------------------")

# Visualize the clusters on the first two components of PCA.
x_axis = df_p_pca_kmeans['Component 2']
y_axis = df_p_pca_kmeans['Component 1']
plt.figure(figsize=(10,8))
sns.scatterplot(x=x_axis, y=y_axis, hue=df_p_pca_kmeans['Segment'], palette=['g', 'r', 'c', 'm'])
plt.show()


'''
TODO: Child table analysis 
'''
#c_df['purchase_amt'] = c_df["purchase_amt"].apply(lambda x: x*-1)

# scaler = StandardScaler()
# scaler.fit(c_df.loc[:, ['purchase_amt', 'purchase_cnt']])
# scaler.transform(c_df.loc[:, ['purchase_amt', 'purchase_cnt']])

# print("Mean of Purchase Count: " + str(c_df['purchase_cnt'].mean()))
# print("Mean of Purchase Amount: " + str(c_df['purchase_amt'].mean()))
# plt.figure(figsize=(15,10))
# plt.scatter(c_df['purchase_cnt'], c_df['purchase_amt'], marker='x')
# plt.ylim([0,200])
# plt.xlim([0,50])
# plt.xlabel('Purchase Count')
# plt.ylabel('Purchase Amount')
# plt.show()


