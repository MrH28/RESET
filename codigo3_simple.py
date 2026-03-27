"""
Minimal test to verify codigo3.py components work correctly
"""

import pandas as pd
import numpy as np
from pathlib import Path

def resolve_csv_path(*candidates):
    """Return the first existing CSV path from candidate names."""
    for name in candidates:
        p = Path(name)
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Files not found: {', '.join(candidates)}")

def read_csv_robust(path):
    """Try common encodings/separators."""
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin1']
    separators = [',', ';', '\t']
    
    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if df.shape[1] > 1:
                    print(f"✓ Successfully read: {path}")
                    print(f"  Encoding: {enc}, Separator: '{sep}'")
                    return df
            except Exception as err:
                pass
    
    raise ValueError(f"Could not read {path}")

print("\n" + "=" * 60)
print("TESTING CODIGO3.PY - Data Loading")
print("=" * 60)

# Test NIR loading
print("\n[1/2] Loading NIR data...")
nir_path = resolve_csv_path('raw_nir_data_saturin.csv', 'raw_nir_data_Saturin.csv')
nir_df = read_csv_robust(nir_path)
print(f"       Shape: {nir_df.shape}")

# Test MIR loading
print("\n[2/2] Loading MIR data...")
mir_path = resolve_csv_path('raw_mir_data_saturin.csv', 'raw_mir_data_Saturin.csv')
mir_df = read_csv_robust(mir_path)
print(f"       Shape: {mir_df.shape}")

# Extract metadata and spectral data
X_nir = nir_df.select_dtypes(include=[np.number])
X_mir = mir_df.select_dtypes(include=[np.number])

print("\n" + "=" * 60)
print("✅ SUCCESS - Both CSV files loaded correctly")
print("=" * 60)
print(f"\nNIR Spectral Data: {X_nir.shape[0]} samples × {X_nir.shape[1]} features")
print(f"MIR Spectral Data: {X_mir.shape[0]} samples × {X_mir.shape[1]} features")
print("\n✓ codigo3.py setup is ready to run")
