import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def resolve_csv_path(*candidates):
    """Return the first existing CSV path from candidate names."""
    for name in candidates:
        p = Path(name)
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Nenhum arquivo encontrado entre: {', '.join(candidates)}"
    )


def read_csv_robust(path):
    """Try common encodings/separators to avoid manual CSV fixes."""
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin1']
    separators = [',', ';', '\t']
    last_error = None

    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if df.shape[1] > 1:
                    print(f"Leitura OK -> arquivo: {path}, encoding: {enc}, sep: '{sep}'")
                    return df
            except Exception as err:
                last_error = err

    raise ValueError(f"Falha ao ler {path}. Último erro: {last_error}")


def extract_non_numeric(df):
    """Extract non-numeric columns and print info."""
    non_num_cols = df.select_dtypes(exclude=[np.number]).columns
    print(f"Não-numéricos: {non_num_cols.tolist()}")
    return non_num_cols, df[non_num_cols]


def prepare_for_savgol(df):
    """Convert to float and handle NaN/Inf to avoid numerical failures."""
    out = df.astype(float).replace([np.inf, -np.inf], np.nan)
    out = out.interpolate(axis=1, limit_direction='both')
    out = out.fillna(out.median(numeric_only=True))
    out = out.fillna(0.0)
    return out


def safe_savgol(df, polyorder=2, preferred_window=11):
    """Apply Savitzky-Golay filter with automatic window adjustment."""
    n_features = df.shape[1]
    max_odd = n_features if n_features % 2 == 1 else n_features - 1
    window_length = min(preferred_window, max_odd)
    min_window = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
    if window_length < min_window:
        window_length = min_window
    if window_length > max_odd:
        raise ValueError(f"Número de features insuficiente para SG: {n_features}")

    clean = prepare_for_savgol(df)
    filt = savgol_filter(clean.values, window_length=window_length, polyorder=polyorder, axis=1)
    return pd.DataFrame(filt, index=df.index, columns=df.columns), window_length


def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 70)
    print("ANÁLISE ESPECTROSCÓPICA: NIR + MIR")
    print("=" * 70)

    # ==================== STAGE 1: LOAD DATA ====================
    print("\n[1/6] Carregando dados...")
    nir_path = resolve_csv_path('raw_nir_data_saturin.csv', 'raw_nir_data_Saturin.csv')
    mir_path = resolve_csv_path('raw_mir_data_saturin.csv', 'raw_mir_data_Saturin.csv')
    nir_df = read_csv_robust(nir_path)
    mir_df = read_csv_robust(mir_path)

    print(f"NIR: {nir_df.shape[0]} amostras × {nir_df.shape[1]} colunas")
    print(f"MIR: {mir_df.shape[0]} amostras × {mir_df.shape[1]} colunas")

    # Extract metadata and spectral data
    non_num_nir, meta_nir = extract_non_numeric(nir_df)
    non_num_mir, meta_mir = extract_non_numeric(mir_df)

    X_nir = nir_df.select_dtypes(include=[np.number])
    X_mir = mir_df.select_dtypes(include=[np.number])

    if X_nir.shape[1] == 0 or X_mir.shape[1] == 0:
        raise ValueError('Não foram encontradas colunas numéricas em NIR e/ou MIR.')

    wavenums_nir = np.linspace(4000, 10000, X_nir.shape[1])
    wavenums_mir = np.linspace(400, 4000, X_mir.shape[1])

    # ==================== STAGE 2: STATISTICS ====================
    print("\n[2/6] Calculando estatísticas...")
    means_nir = X_nir.mean(axis=1)
    cv_nir = X_nir.std(axis=1) / X_nir.mean(axis=1) * 100

    print(f"NIR - Média global: {X_nir.mean().mean():.4f}")
    print(f"NIR - CV médio: {cv_nir.mean():.4f}%")

    means_mir = X_mir.mean(axis=1)
    cv_mir = X_mir.std(axis=1) / X_mir.mean(axis=1) * 100

    # ==================== STAGE 3: VISUALIZATION ====================
    print("\n[3/6] Gerando visualizações (espectros)...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0,0].plot(wavenums_nir, X_nir.mean(axis=0), label='Média NIR', linewidth=2)
    axes[0,0].plot(wavenums_mir, X_mir.mean(axis=0), label='Média MIR', alpha=0.8, linewidth=2)
    axes[0,0].set_title('Médias Espectrais', fontsize=12, fontweight='bold')
    axes[0,0].legend()

    n_nir = min(10, len(X_nir))
    n_mir = min(10, len(X_mir))
    idx_nir = np.random.choice(X_nir.index, n_nir, replace=False)
    idx_mir = np.random.choice(X_mir.index, n_mir, replace=False)
    axes[0,1].plot(wavenums_nir, X_nir.loc[idx_nir].T, alpha=0.6)
    axes[0,1].set_title('Amostras NIR', fontsize=12, fontweight='bold')

    axes[1,0].plot(wavenums_mir, X_mir.loc[idx_mir].T, alpha=0.6)
    axes[1,0].set_title('Amostras MIR', fontsize=12, fontweight='bold')

    axes[1,1].boxplot([cv_nir, cv_mir], labels=['NIR CV', 'MIR CV'])
    axes[1,1].set_title('Coef. Variação', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('espectros_comparacao.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: espectros_comparacao.png")
    plt.close()

    # ==================== STAGE 4: FILTERING ====================
    print("\n[4/6] Aplicando filtro Savitzky-Golay...")
    X_nir_filt, win_nir = safe_savgol(X_nir, polyorder=2, preferred_window=11)
    X_mir_filt, win_mir = safe_savgol(X_mir, polyorder=2, preferred_window=11)
    print(f"SG NIR → window_length: {win_nir}")
    print(f"SG MIR → window_length: {win_mir}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(wavenums_nir, X_nir.mean(0), label='Raw', linewidth=2)
    plt.plot(wavenums_nir, X_nir_filt.mean(0), label='Filtered', linewidth=2)
    plt.legend()
    plt.title('NIR Filtro', fontweight='bold')

    plt.subplot(1,2,2)
    plt.plot(wavenums_mir, X_mir.mean(0), label='Raw', linewidth=2)
    plt.plot(wavenums_mir, X_mir_filt.mean(0), label='Filtered', linewidth=2)
    plt.legend()
    plt.title('MIR Filtro', fontweight='bold')

    plt.savefig('filtros_ruido.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: filtros_ruido.png")
    plt.close()

    # ==================== STAGE 5: PCA + CLUSTERING ====================
    print("\n[5/6] Análise de PCA e clustering...")
    n_samples = min(len(X_nir_filt), len(X_mir_filt))
    if len(X_nir_filt) != len(X_mir_filt):
        print(f"⚠ NIR ({len(X_nir_filt)}) e MIR ({len(X_mir_filt)}) tamanhos diferentes.")
        print(f"→ Usando {n_samples} amostras para combinação.")

    X_nir_aligned = X_nir_filt.iloc[:n_samples].reset_index(drop=True)
    X_mir_aligned = X_mir_filt.iloc[:n_samples].reset_index(drop=True)
    meta_nir_aligned = meta_nir.iloc[:n_samples].reset_index(drop=True)

    X_combined = np.hstack([X_nir_aligned.values, X_mir_aligned.values])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.scatter(pca_result[:,0], pca_result[:,1], alpha=0.6, s=20)
    plt.title(f'PCA (Expl Var: {pca.explained_variance_ratio_.sum():.2%})', fontweight='bold')

    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)
    plt.subplot(1,2,2)
    scatter = plt.scatter(pca_result[:,0], pca_result[:,1], c=clusters, cmap='viridis', s=20)
    plt.title('K-Means Clusters', fontweight='bold')
    plt.colorbar(scatter)
    
    plt.savefig('pca_clusters.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: pca_clusters.png")
    plt.close()

    # ==================== STAGE 6: CLASSIFICATION ====================
    print("\n[6/6] Classificação com Random Forest...")
    candidate_target_cols = ['class', 'Class', 'classe', 'Classe', 'label', 'Label', 'target', 'Target']
    target_col = next((c for c in candidate_target_cols if c in meta_nir_aligned.columns), None)

    if target_col is None:
        fallback_cols = [
            c for c in meta_nir_aligned.columns
            if meta_nir_aligned[c].nunique(dropna=True) > 1
        ]
        if not fallback_cols:
            raise KeyError(
                f"Não foi possível inferir coluna alvo. Colunas: "
                f"{meta_nir_aligned.columns.tolist()}"
            )
        target_col = fallback_cols[0]

    y = meta_nir_aligned[target_col].astype(str).fillna('NA').values
    print(f"Coluna alvo: '{target_col}' ({len(np.unique(y))} classes)")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(rf, X_scaled, y, cv=5)
    
    print(f"\n✓ Acurácia Teste: {acc:.3f}")
    print(f"✓ CV Médio: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=range(X_scaled.shape[1]))
    top_feats = importances.nlargest(20)
    plt.figure(figsize=(10, 4))
    bars = plt.bar(range(len(top_feats)), top_feats.values, color='steelblue')
    plt.title('Top 20 Features - Random Forest', fontweight='bold', fontsize=12)
    plt.xlabel('Feature Index (NIR+MIR)')
    plt.ylabel('Importância')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_importance.png")
    plt.close()

    print("\n" + "=" * 70)
    print("✅ ANÁLISE CONCLUÍDA COM SUCESSO")
    print("=" * 70)
    print(f"\nResumo de Saídas:")
    print(f"  • espectros_comparacao.png")
    print(f"  • filtros_ruido.png")
    print(f"  • pca_clusters.png")
    print(f"  • feature_importance.png")
    print()


if __name__ == "__main__":
    main()
