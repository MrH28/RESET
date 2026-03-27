import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd

# Load data using pandas for large files
df = pd.read_csv("raw_mir_data_Saturin.csv")
print(df)
print(df.shape)
print("Tipos:", df.dtypes)
print(df.head())
# Extract wavenumbers (first row, skip first column which might be sample names)
numeric_cols = pd.to_numeric(df.columns[2:], errors='coerce').notna()  # skip non-numeric, 
wavenumbers = pd.to_numeric(df.columns[2:][numeric_cols]).astype(float)
print(wavenumbers)
print(wavenumbers.values)
print(wavenumbers.shape)
print(wavenumbers.dtype)
print(wavenumbers[:10])

print("Wavenumbers extraidos:", wavenumbers)
print("Quantidade:", len(wavenumbers))
print("Range:", wavenumbers.min(), "a", wavenumbers.max())

# Extract spectra data (skip first column which might be labels)
spectra = df.iloc[:, 16:].values.astype(float)
spectra = df[(df.columns[2:][numeric_cols])].astype(float)
print(spectra)
print(spectra.shape)
print(f"shape: {spectra.shape} {spectra.shape[0]} amostras solo * {spectra.shape[1]} wavenumbers")

# Plot Espectral
# Plot teste 
spectra = df[(df.columns[2:][numeric_cols])].astype(float)

plt.figure(figsize=(12, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple']
for i in range(5):
    # use lowercase 'spectra' (defined earlier) and match df index
    plt.plot(wavenumbers[::-1], spectra.iloc[i] if hasattr(spectra, 'iloc') else spectra[i], 
             color=colors[i], linewidth=2,
             label=f"Amostra {i+1} (ID: {df.iloc[i]})")


plt.xlabel("Wavenumbers (cm$^{-1}$)", fontsize=12, fontweight='bold')
plt.ylabel("Absorbância", fontsize=12, fontweight='bold')
plt.title("Primeiros 5 Espectros MIR - Variação Composição Solo", fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.xlim(4000, 400)
plt.tight_layout()
plt.savefig('espectros_com_solo.png', dpi=300)
plt.show()


# Suavizaçao Savitzky-Golay
# Applicar filtro para suavizar ruido preservando picos 

window_length = 11  # Ímpar, ajuste baseado em ruído
polyorder = 2
spectra_smooth = savgol_filter(spectra, window_length, polyorder, axis=1)
plt.plot(wavenumbers, spectra[0].T, label='Raw')
plt.plot(wavenumbers, spectra_smooth[0].T, label='Suavizado')
plt.legend()
plt.show()


spectra_deriv = savgol_filter(spectra_smooth, window_length, polyorder, axis=1, deriv=1)
plt.plot(wavenumbers, spectra_deriv[0].T)
plt.title('1ª Derivada')
plt.show()

scaler = StandardScaler()
spectra_scaled = scaler.fit_transform(spectra_deriv)
pca = PCA(n_components=0.95)  # 95% variância
spectra_pca = pca.fit_transform(spectra_scaled)
print(f'Componentes: {pca.n_components_}, Variância: {pca.explained_variance_ratio_.sum():.2f}')
plt.scatter(spectra_pca[:,0], spectra_pca[:,1])
plt.show()
inertias = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(spectra_pca)
    inertias.append(kmeans.inertia_)
plt.plot(range(2,10), inertias)
plt.title('Elbow Method')
plt.show()

k_opt = 3  # Ajuste visualmente
kmeans = KMeans(n_clusters=k_opt, random_state=42)
clusters = kmeans.fit_predict(spectra_pca)
plt.scatter(spectra_pca[:,0], spectra_pca[:,1], c=clusters)
plt.title('Clusters K-Means')
plt.show()

# Assuma labels reais ou use clusters como proxy
X_train, X_test, y_train, y_test = train_test_split(spectra_pca, clusters, test_size=0.2)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
print(f'Acurácia: {rf.score(X_test, y_test):.2f}')



