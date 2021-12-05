# libraries
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)


# initialize the data-frames (child and parent)
child_df = pd.read_csv("./Data/ChildData.csv")
parent_df = pd.read_csv("./Data/ParentData.csv")

p_df = parent_df
c_df = child_df

c_df['purchase_amt'] = c_df["purchase_amt"].apply(lambda x: x*-1)

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

# print(parent_df.describe())
# print(parent_df.info())

values = {"trxn_amt": 0, "deposit_cnt": 0, "transfer_cnt": 0, "deposit_amt": 0, "transfer_amt": 0}
p_df = p_df.fillna(value=values)
p_df = p_df.iloc[:, 8:]
print(p_df.info(), p_df.describe())

plt.figure(figsize=(10,8))
plt.scatter(p_df['deposit_cnt'], p_df['deposit_amt'])
plt.xlabel("Deposit Count")
plt.ylabel("Deposit Amount")
plt.show()


print('++++++++++++++++++++++++++++++++++++++++++++++++')
scaler = StandardScaler()
#scaler = MinMaxScaler()
p_df_std = scaler.fit_transform(p_df)
print(p_df_std)
print('++++++++++++++++++++++++++++++++++++++++++++++++')

pca = PCA()
pca.fit(p_df_std)

# PCA on non-scaled data-frame
# pca2 = PCA()
# pca2.fit(p_df)

# PCA with SCALING
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

pca = PCA(n_components=3)
pca.fit(p_df_std)
scores_pca = pca.transform(p_df_std)
print(scores_pca)

# K-means
wcss = []
for i in range(1, 30):
    kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state= 42)
    kmeans_pca.fit(scores_pca)
    wcss.append((kmeans_pca.inertia_))

# Elbow technique
# plt.figure(figsize=(10,8))
# plt.plot(range(1,30), wcss, marker='x', linestyle='--')
# plt.xlabel("Number of Components")
# plt.ylabel("WCSS (Within Cluster Sum of Squares)")
# plt.title("WCSS vs Number of Clusters")
# plt.show()

kmeans_pca = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans_pca.fit(scores_pca)
print("----------------------")
print(kmeans_pca.cluster_centers_)
print("----------------------")

df_p_pca_kmeans = pd.concat([p_df.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
df_p_pca_kmeans.columns.values[-3:] = ['Component 1', 'Component 2', 'Component 3']
df_p_pca_kmeans['Segment K-Means PCA'] = kmeans_pca.labels_
df_p_pca_kmeans['Segment'] = df_p_pca_kmeans['Segment K-Means PCA'].map({0: 'first',
                                                                         1: 'second', 2: 'third', 3: 'fourth'})

print(df_p_pca_kmeans.columns)
print(df_p_pca_kmeans)


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

x_axis = df_p_pca_kmeans['Component 2']
y_axis = df_p_pca_kmeans['Component 1']
plt.figure(figsize=(10,8))
sns.scatterplot(x=x_axis, y=y_axis, hue=df_p_pca_kmeans['Segment'], palette=['g', 'r', 'c', 'm'])
plt.show()



