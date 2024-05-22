import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file and set the 'coin_id' column as the index.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The loaded DataFrame with 'coin_id' as the index.
    """
    return pd.read_csv(file_path, index_col='coin_id')

def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: Summary statistics of the DataFrame.
    """
    return df.describe()

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize data using StandardScaler.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The normalized DataFrame.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    return scaled_df

def fit_kmeans(df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """
    Fit KMeans clustering on the data.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    n_clusters (int): The number of clusters to form.
    
    Returns:
    pd.DataFrame: The DataFrame with cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    df['cluster'] = kmeans.fit_predict(df)
    return df

def perform_pca(df: pd.DataFrame, n_components: int) -> pd.DataFrame:
    """
    Perform PCA on the data.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    n_components (int): The number of components for PCA.
    
    Returns:
    pd.DataFrame: The DataFrame with principal components.
    """
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(df)
    pca_df = pd.DataFrame(pca_data, index=df.index)
    return pca_df

def compute_elbow_curve(df: pd.DataFrame, k_range: range) -> pd.DataFrame:
    """
    Compute the inertia for a range of k values to plot the elbow curve.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    k_range (range): The range of k values to try.
    
    Returns:
    pd.DataFrame: The DataFrame containing k values and their corresponding inertia.
    """
    inertia = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)
    
    elbow_data = {"k": list(k_range), "inertia": inertia}
    elbow_df = pd.DataFrame(elbow_data)
    
    return elbow_df

def plot_elbow_curve(elbow_df: pd.DataFrame):
    """
    Plot the elbow curve using the provided DataFrame.
    
    Parameters:
    elbow_df (pd.DataFrame): The DataFrame containing the elbow curve data.
    """
    plt.plot(elbow_df["k"], elbow_df["inertia"])
    plt.xticks(elbow_df["k"])
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Curve")
    plt.show()

def plot_pca_clusters(df: pd.DataFrame, x: str, y: str, cluster_col: str, colormap: str = "rainbow"):
    """
    Plot PCA components with cluster labels.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing PCA components and cluster labels.
    x (str): The column name for the x-axis.
    y (str): The column name for the y-axis.
    cluster_col (str): The column name for cluster labels.
    colormap (str): The colormap to use for the scatter plot.
    """
    df.plot.scatter(x=x, y=y, c=cluster_col, colormap=colormap)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'PCA of Crypto Market Data ({x} vs {y})')
    plt.show()