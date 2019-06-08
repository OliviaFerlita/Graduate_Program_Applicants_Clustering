"Graduate Program Applicants Clustering Using K-Means Clustering"
"Final Term Project for Multivariate Statisfied Analysis"
"By : Olivia Ferlita [M10702818]"

# Importing the libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Importing the dataset
data = pd.read_csv("Admission_Predict_Ver1.1.csv")
data = data.drop(['Serial No.'], axis=1)

# Making the correlation matrix
correlation = data.corr()
plt.figure(figsize=(13, 11))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="coolwarm", annot_kws={"size":16})
plt.savefig('Correlation_Matrix.png', format='png')

# Feature Scaling
scaled_data = data.copy()
sc = preprocessing.StandardScaler()
columns = data.columns[0:6]
scaled_data[columns] = sc.fit_transform(scaled_data[columns])

# Determine the optimal number of clusters using elbow method
ess = []
for k in range(1,7):
    kms = KMeans(n_clusters = k)
    kms.fit(scaled_data)
    ess.append(kms.inertia_)
    
plt.figure(figsize = (8,5)) 
plt.scatter(range(1,7),ess)
plt.plot(range(1,7), ess, linewidth=2)
plt.title('Elbow Plot', fontsize = 16)
plt.xlabel('Number of Clusters', fontsize = 14)
plt.ylabel('ESS', fontsize = 14)
plt.savefig('Elbow Plot.png', format='png')

# Cluster data with 3 clusters KMeans
kms = KMeans(n_clusters=3)
scaled_data['Cluster'] = kms.fit_predict(scaled_data.iloc[:,0:7])
ess_final = kms.inertia_

# Create PCA instance
kms_pca = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = kms_pca.fit_transform(scaled_data.iloc[:,0:7])

# Scatter plot xs vs ys
plot = sns.scatterplot(x=pca_features[:,0], y=pca_features[:,1], hue="Cluster", data=scaled_data)
plt.title('K-Means Clustering (Scatter Plot)', fontsize = 14)
plt.savefig('K-Means Clustering_scatter.png', format='png')

# Making boxplot
plot = sns.boxplot(x="Cluster", y="Chance of Admit ", data=scaled_data, palette="Set1" )
plt.title('K-Means Clustering (Box Plot)', fontsize = 14)
plt.savefig('K-Means Clustering_box.png', format='png')
#
#
#
#
#
#