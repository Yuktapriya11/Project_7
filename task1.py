import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df.iloc[:, :-1])
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['target'] = df['target']

# Visualize
plt.figure(figsize=(8, 6))
for target in [0, 1, 2]:
    plt.scatter(pca_df[pca_df.target == target]['PC1'], 
                pca_df[pca_df.target == target]['PC2'], 
                label=iris.target_names[target])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Iris Dataset')
plt.legend()
plt.grid(True)
plt.show()
