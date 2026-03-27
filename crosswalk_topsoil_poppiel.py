from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def robust_read_csv(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, engine="python", sep=None)


def detect_topsoil(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    horizon_cols = [
        c for c in out.columns
        if "horiz" in c.lower() or "horizon" in c.lower() or c.lower() in ["hz", "camada", "layer"]
    ]
    if not horizon_cols:
        return out

    hc = horizon_cols[0]
    hs = out[hc].astype(str).str.upper()
    mask = hs.str.contains(r"\bA\b|^A|AP|A1|A2", regex=True, na=False)
    if mask.any():
        return out[mask].copy()
    return out


def infer_cluster(df: pd.DataFrame, candidate_attrs: list[str], n_clusters: int) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    for c in ["cluster", "Cluster", "soil_cluster", "SOIL_CLUSTER", "classe", "class", "Class"]:
        if c in out.columns:
            return out, c

    usable = [c for c in candidate_attrs if c in out.columns]
    if len(usable) < 2:
        out["soil_cluster"] = 1
        return out, "soil_cluster"

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    X = out[usable].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    Z = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, n_init=30, random_state=42)
    out["soil_cluster"] = km.fit_predict(Z).astype(int) + 1
    return out, "soil_cluster"


def build_legend(summary: pd.DataFrame) -> pd.DataFrame:
    legend = summary[["cluster"]].copy()

    for base in ["Clay_gkg", "Sand_gkg", "OM_gkg", "CEC_Ph7_mmolkg", "V_pct"]:
        col = f"{base}_mean"
        if col in summary.columns:
            q1, q2 = summary[col].quantile([1 / 3, 2 / 3]).values

            def cat(v):
                if pd.isna(v):
                    return "unknown"
                if v <= q1:
                    return "low"
                if v <= q2:
                    return "medium"
                return "high"

            legend[f"{base}_level"] = summary[col].apply(cat)

    def make_label(row):
        parts = []
        for p, tag in [
            ("Clay_gkg_level", "clay"),
            ("Sand_gkg_level", "sand"),
            ("OM_gkg_level", "OM"),
            ("CEC_Ph7_mmolkg_level", "CEC"),
            ("V_pct_level", "V%"),
        ]:
            if p in row and row[p] != "unknown":
                parts.append(f"{row[p]}-{tag}")
        return ", ".join(parts) if parts else "unclassified"

    legend["soil_legend"] = legend.apply(make_label, axis=1)
    return legend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross topsoil (A horizon) attributes and soil legend by cluster with Poppiel-style results."
    )
    parser.add_argument("--root", type=str, default=".", help="Workspace root path")
    parser.add_argument("--n-clusters", type=int, default=4, help="Number of clusters when clustering is inferred")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    nir_path = root / "raw_nir_data_Saturin.csv"
    popp_path = root / "poppiel_2022_pipeline_results.csv"

    if not nir_path.exists():
        raise FileNotFoundError(f"Missing input file: {nir_path}")
    if not popp_path.exists():
        raise FileNotFoundError(f"Missing input file: {popp_path}")

    nir = robust_read_csv(nir_path)
    popp = robust_read_csv(popp_path)

    nir.columns = [str(c).strip() for c in nir.columns]
    popp.columns = [str(c).strip() for c in popp.columns]

    candidate_attrs = [
        "Clay_gkg", "Sand_gkg", "OM_gkg", "CEC_Ph7_mmolkg",
        "Ca_mmolkg", "Mg_mmolkg", "K_mmolkg", "pH_H2O", "pH_CaCl2", "BS_pct", "V_pct"
    ]

    if "V_pct" not in nir.columns and all(c in nir.columns for c in ["Ca_mmolkg", "Mg_mmolkg", "K_mmolkg", "CEC_Ph7_mmolkg"]):
        cec = pd.to_numeric(nir["CEC_Ph7_mmolkg"], errors="coerce")
        bases = (
            pd.to_numeric(nir["Ca_mmolkg"], errors="coerce")
            + pd.to_numeric(nir["Mg_mmolkg"], errors="coerce")
            + pd.to_numeric(nir["K_mmolkg"], errors="coerce")
        )
        nir["V_pct"] = (bases * 100.0) / cec.replace(0, np.nan)

    nir, cluster_col = infer_cluster(nir, candidate_attrs, n_clusters=args.n_clusters)
    top = detect_topsoil(nir)

    attrs = [c for c in candidate_attrs if c in top.columns]
    for c in attrs:
        top[c] = pd.to_numeric(top[c], errors="coerce")

    summary = top.groupby(cluster_col, dropna=False)[attrs].agg(["count", "mean", "std", "median", "min", "max"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index().rename(columns={cluster_col: "cluster"})

    legend = build_legend(summary)

    popp_best = popp.sort_values(["target", "rmse"]).groupby("target", as_index=False).first()
    keep_cols = [c for c in ["target", "split", "rmse", "r2", "n_train", "n_test"] if c in popp_best.columns]
    popp_best = popp_best[keep_cols]

    cluster_means_long = []
    for t in ["Clay_gkg", "Sand_gkg", "OM_gkg", "CEC_Ph7_mmolkg", "V_pct"]:
        if t in top.columns:
            tmp = top.groupby(cluster_col, dropna=False)[t].mean().reset_index()
            tmp.columns = ["cluster", "your_mean_value"]
            tmp["target"] = t
            cluster_means_long.append(tmp)

    if cluster_means_long:
        cluster_means_long = pd.concat(cluster_means_long, ignore_index=True)
    else:
        cluster_means_long = pd.DataFrame(columns=["cluster", "your_mean_value", "target"])

    comparison = cluster_means_long.merge(popp_best, on="target", how="left")

    out_summary = root / "topsoil_cluster_attribute_summary_crosswalk.csv"
    out_legend = root / "topsoil_cluster_soil_legend_crosswalk.csv"
    out_comparison = root / "topsoil_cluster_vs_poppiel2019_crossing.csv"

    summary.to_csv(out_summary, index=False)
    legend.to_csv(out_legend, index=False)
    comparison.to_csv(out_comparison, index=False)

    print("Done")
    print(f"Rows all NIR: {len(nir)}")
    print(f"Rows topsoil used: {len(top)}")
    print(f"Cluster column used: {cluster_col}")
    print(f"Attributes used: {attrs}")
    print(f"Saved: {out_summary.name}")
    print(f"Saved: {out_legend.name}")
    print(f"Saved: {out_comparison.name}")


if __name__ == "__main__":
    main()
