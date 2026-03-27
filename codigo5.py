import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def resolve_csv_path(*candidates):
    for name in candidates:
        p = Path(name)
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"CSV not found. Tried: {', '.join(candidates)}")


def read_csv_robust(path):
    # NIR in this workspace is typically cp1252, so try it first.
    if "nir" in Path(path).name.lower():
        encodings = ["cp1252", "latin1", "utf-8", "utf-8-sig"]
    else:
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    separators = [",", ";", "\t"]
    last_error = None

    for enc in encodings:
        for sep in separators:
            try:
                # Fast probe before full read to skip bad combinations quickly.
                probe = pd.read_csv(path, encoding=enc, sep=sep, nrows=50, low_memory=False, dtype=str)
                if probe.shape[1] <= 1:
                    continue
                # Read as strings to avoid expensive and fragile dtype inference on large mixed columns.
                df = pd.read_csv(path, encoding=enc, sep=sep, low_memory=False, dtype=str)
                if df.shape[1] > 1:
                    print(f"Loaded {path} with encoding={enc}, sep='{sep}'")
                    return df
            except Exception as err:
                last_error = err

    raise ValueError(f"Failed to read {path}. Last error: {last_error}")


class PipelineEspectralNIR:
    def __init__(self, arquivo_csv):
        self.arquivo_csv = arquivo_csv
        self.df = read_csv_robust(arquivo_csv)
        self.wavenumbers = None
        self.spectra = None
        self.spectra_proc = None
        self.pca_result = None
        self.clusters = None
        self.sample_idx = None

    def extrair_dados(self):
        """Extract spectral columns and convert to numeric matrix."""
        spec_cols = [c for c in self.df.columns if pd.to_numeric(pd.Series([c]), errors="coerce").notna().iloc[0]]
        if not spec_cols:
            # Fallback: treat all numeric-typed columns as spectral columns.
            spec_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not spec_cols:
            raise ValueError("No spectral numeric columns were found in the NIR CSV.")

        self.wavenumbers = pd.to_numeric(spec_cols, errors="coerce")
        order = np.argsort(self.wavenumbers)
        spec_cols = [spec_cols[i] for i in order]
        self.wavenumbers = self.wavenumbers[order]

        spectra_df = self.df[spec_cols].apply(pd.to_numeric, errors="coerce")
        spectra_df = spectra_df.replace([np.inf, -np.inf], np.nan)
        spectra_df = spectra_df.interpolate(axis=1, limit_direction="both")
        spectra_df = spectra_df.fillna(spectra_df.median(numeric_only=True)).fillna(0)
        self.spectra = spectra_df.values.astype(np.float32)
        print(f"Extracted: {self.spectra.shape[0]} samples x {self.spectra.shape[1]} wavenumbers")

    def detectar_negativos(self):
        negativos = int((self.spectra < 0).sum())
        print(f"Negative values found: {negativos}")
        if negativos > 0:
            self.spectra[self.spectra < 0] = 0

    def preprocess_savitzky_golay(self, window=11, polyorder=3):
        n_features = self.spectra.shape[1]
        max_odd = n_features if n_features % 2 == 1 else n_features - 1
        min_window = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
        window = min(window, max_odd)
        if window < min_window:
            window = min_window
        if window > max_odd:
            raise ValueError(f"Not enough features for SG filter: {n_features}")

        smooth = savgol_filter(self.spectra, window_length=window, polyorder=polyorder, axis=1)
        self.spectra_proc = savgol_filter(
            smooth,
            window_length=window,
            polyorder=polyorder,
            deriv=1,
            axis=1,
        )
        print(f"Savitzky-Golay completed with window={window}, polyorder={polyorder}")

    def pca_reducao(self, n_components=40, max_samples=12000):
        matrix = self.spectra_proc
        if matrix.shape[0] > max_samples:
            rng = np.random.default_rng(42)
            self.sample_idx = np.sort(rng.choice(matrix.shape[0], size=max_samples, replace=False))
            matrix = matrix[self.sample_idx]
            print(f"Sampling {max_samples} rows for PCA/clustering (from {self.spectra_proc.shape[0]}).")
        else:
            self.sample_idx = np.arange(matrix.shape[0])

        scaler = StandardScaler()
        spectra_scaled = scaler.fit_transform(matrix)

        max_components = max(2, min(spectra_scaled.shape[0] - 1, spectra_scaled.shape[1], int(n_components)))
        pca = PCA(n_components=max_components, svd_solver="randomized", random_state=42)
        self.pca_result = pca.fit_transform(spectra_scaled)
        explained = pca.explained_variance_ratio_.sum()
        print(f"PCA: {self.pca_result.shape[1]} components, explained variance={explained:.2%}")

    def cluster_kmeans(self, n_clusters=4):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        self.clusters = kmeans.fit_predict(self.pca_result)
        print(f"K-means: {n_clusters} clusters identified")

    def gerar_figura(self, output_file="pipeline_nir.png"):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(self.wavenumbers, self.spectra[0], label="Raw")
        plt.plot(self.wavenumbers, self.spectra_proc[0], label="SG+Deriv1")
        plt.title("NIR Preprocessing")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.scatter(self.pca_result[:, 0], self.pca_result[:, 1], c=self.clusters, s=8, cmap="viridis")
        plt.title("PCA + Clusters")
        plt.xlabel("PC1")
        plt.ylabel("PC2")

        plt.subplot(1, 3, 3)
        vals, counts = np.unique(self.clusters, return_counts=True)
        plt.bar(vals.astype(str), counts)
        plt.title("Cluster Counts")
        plt.xlabel("Cluster")
        plt.ylabel("Samples")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {output_file}")


def main():
    nir_path = resolve_csv_path("raw_nir_data_saturin.csv", "raw_nir_data_Saturin.csv")
    print(f"Using NIR file: {nir_path}")

    pipeline = PipelineEspectralNIR(nir_path)
    pipeline.extrair_dados()
    pipeline.detectar_negativos()
    pipeline.preprocess_savitzky_golay(window=11, polyorder=3)
    pipeline.pca_reducao(n_components=40, max_samples=12000)
    pipeline.cluster_kmeans(n_clusters=4)
    pipeline.gerar_figura(output_file="pipeline_nir.png")

    print("Run completed successfully.")
    print(f"Final matrix shape: {pipeline.spectra.shape}")
    print(f"PCA shape: {pipeline.pca_result.shape}")


if __name__ == "__main__":
    main()
