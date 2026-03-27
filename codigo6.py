import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter

# Wavenumbers típicos (invertido para MIR)
wl_visnirswir = np.linspace(350, 2500, 1000)  # nm
wl_midir = np.linspace(2500, 25000, 2000)     # nm (em nm, ou use cm-1)

# Simular reflectância: argiloso (baixa albedo, absorções OH/Fe) vs arenoso (alta)
# Argiloso: absorções ~1400nm (H2O), 2200nm (AlOH), Fe visível
reflect_argila_nir = 0.25 + 0.15 * np.sin(2*np.pi*wl_visnirswir/1000) - 0.1*np.exp(-((wl_visnirswir-1400)/200)**2) - 0.08*np.exp(-((wl_visnirswir-2200)/150)**2)
reflect_areia_nir = 0.55 + 0.1 * np.sin(2*np.pi*wl_visnirswir/2000)  # Alta reflectância quartzo

reflect_argila_mir = 0.20 * np.exp(-wl_midir/10000) - 0.15*np.exp(-((wl_midir-6000)/800)**2)  # O-H ~3000-4000nm equiv.
reflect_areia_mir = 0.45 * np.exp(-wl_midir/15000)  # Mais plana

# DataFrame para exportar (como seus CSVs)
df_nir = pd.DataFrame({
    'wl_nm': wl_visnirswir,
    'argiloso': reflect_argila_nir,
    'arenoso': reflect_areia_nir
})
df_mir = pd.DataFrame({
    'wl_nm': wl_midir,
    'argiloso': reflect_argila_mir,
    'arenoso': reflect_areia_mir
})
df_nir.to_csv('espectros_visnirswir.csv', index=False)
df_mir.to_csv('espectros_midIR.csv', index=False)

# Suavização Savitzky-Golay (como nos seus pipelines)
reflect_argila_nir_sg = savgol_filter(reflect_argila_nir, 21, 3)
reflect_areia_nir_sg = savgol_filter(reflect_areia_nir, 21, 3)

# Gráficos (Figura 3-like)
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
# vis-NIR-SWIR
axes[0].plot(wl_visnirswir, reflect_argila_nir_sg, 'r-', linewidth=2.5, label='Argiloso seco (baixa albedo, absorções OH/Fe)')
axes[0].plot(wl_visnirswir, reflect_areia_nir_sg, 'orange', linewidth=2.5, label='Arenoso seco (alta albedo, quartzo)')
axes[0].set_ylim(0, 0.7)
axes[0].set_xlabel('Comprimento de onda (nm)')
axes[0].set_ylabel('Reflectância')
axes[0].set_title('vis-NIR-SWIR: Variações intensidade e absorções')
axes[0].legend()
axes[0].grid(alpha=0.3)

# mid-IR
axes[1].plot(wl_midir, reflect_argila_mir, 'r--', linewidth=2.5, label='Argiloso (picos O-H, Si-O)')
axes[1].plot(wl_midir, reflect_areia_mir, 'orange--', linewidth=2.5, label='Arenoso (suave)')
axes[1].set_ylim(0, 0.6)
axes[1].set_xlabel('Comprimento de onda (nm)')
axes[1].set_ylabel('Reflectância')
axes[1].set_title('mid-IR: Feições vibacionais minerais')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('comportamentos_espectrais_solos.png', dpi=300, bbox_inches='tight')
plt.show()

print("Arquivos salvos: espectros_visnirswir.csv, espectros_midIR.csv")
print("Gráfico salvo: comportamentos_espectrais_solos.png")
print("Argiloso NIR média:", np.mean(reflect_argila_nir_sg))
print("Arenoso NIR média:", np.mean(reflect_areia_nir_sg))