# ANALISE DE DADOS
# Passo à passo do projeto
# Passo 1: Preparaçao e Carregamento
# Passo 2: Suavizaçao Savitzky-Golay
# Passo 3: Detecçao de Picos
# Passo 4: Analise de Resultados


# Passo 1: Preparaçao e Carregamento

# Carregar raw_mir_data_Saturin.csv usando pandas
import pandas as pd
df = pd.read_csv('raw_mir_data_Saturin.csv')
print(df)


# Exibir as primeiras linhas do DataFrame para verificas os dados
print(df.head())


# Linhas: amostra de dados, colunas: wavenumbers
# wavenumbers = df.columns[1:] # Assumindo primeira coluna  labels
print(df.columns[1:])

# Exibir as colunas do DataFrame para verificar os wavenumbers
print(df.columns)

# Separar os dados em colunas
# Corrigir os problemas
# Criar lista de nomes
lista_nomes = ["Upper_depth_cm", "Lower_depth_cm", "Sand_gkg", "Silt_gkg"]

#
# tira columas dentro da tabela para analisar

df = df.drop(["Upper_depth_cm","Lower_depth_cm", "Sand_gkg", "Silt_gkg"], axis=1)
print(df)
print(df.info())

# tira linha vazia
df = df.dropna()
print(df)

# Analise de dados
# Depois de ter tirado as columas indesejados e as linhas indesejados devo atualizar meu dataframe, mas como ? 
print(df.columns)

# EM seguido, validar os meus dados in pythons 

print("Shape:", df.shape)
print(df.head())  # Primeiras 5 linhas

# Execute meu script no terminal com por exemplo "python codigo.py"
# Validacao basicas com shema e info
# Schema : colunas esperadas
expected_cols = ['ID', 'owner', 'X', 'Y', 'country', 'Upper_depth_cm', 'lower_depth_cm', 'sand_gkg','silt_gkg','clay_gkg']
missing_cols = [col for col in expected_cols if col not in df.columns]
print("Colunas faltando:", missing_cols)
print("Tipos de dados:")
print(df.dtypes)

# Info geral 
print("Resumo:")
print(df.info())
print("Ausentes por colunas:")
print(df.isnull().sum())

# Duplicados
print("Duplicados:", df .duplicated().sum())

# Estaticas Descritivas e Outliers
# ver colunas 
print("colunas:")
print(df.columns.tolist())
print(df.columns[:20].tolist(), "...", df.columns[-5:].tolist())


# Analise Solo vars (sand_gkg etc.) para ranges reais ( ex: 0-1000 g/kg)
# Stats solo vars    (defina lista correta & ajuste nomes se necessario) 
df = pd.read_csv('raw_mir_data_Saturin.csv')
df.describe()
print(df.describe())
# Describe nas vars de solo
# Analise descritiva
# Tiras todas outras columans da tabela para analysar so os solos 

print("Estatisticas solo:")
print(df.describe())
print("Estatisticas solo:", df.describe())

# Stats extras
print("Ausentes:")
print(df.isnull().sum())

# Filtragem e Limpeza
# Filtre pais[India, Latvia,Finland], profundidade, remova ausentes/ outliers
# Filtre India, profundidade 0-100cm, sem ausentes em solo
# Romova outliers (IQR method para sand)

# Schema Final e Salvar
# Confirme schema limpo e exporto

# spectra = df.iloc[:, 1:].values

# Passo 2: Suavizaçao Savitzky-Golay
# Aplicar o filtro Savitzky-Golay para suavizar os dados

# savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

np.set_printoptions(precision=2)

wavenumbers = pd.to_numeric(wavenumber_cols).sort_values().values
spectra = df[wavenumber_cols].values.astype(np.float32)

spectra_suavizado = savgol_filter(spectra, window_length=11, polyorder=3, axis=1)

plt.figure(figsize=(12, 6))
tempo = np.arange(len(wavenumbers))  

plt.plot(wavenumbers[::-1], spectra[0], label='Espectro Original', alpha=0.7, color='blue')
plt.plot(wavenumbers[::-1], spectra_suavizado[0], label='Savitzky-Golay (11,3)', 
         color='orange', linewidth=2.5)

# Detectar picos no suavizado
picos, _ = find_peaks(spectra_suavizado[0], height=0.1)  # Ajuste height
plt.plot(wavenumbers[::-1][picos], spectra_suavizado[0][picos], 'rx', 
         markersize=8, label='Picos Detectados')

plt.title('Suavização Savitzky-Golay - Espectro MIR Solo')
plt.xlabel('Wavenumbers (cm$^{-1}$)')
plt.ylabel('Absorbância')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(4000, 400)
plt.tight_layout()
plt.show()

display 
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

np.set_printoptions (precision=2) # For compact display
x = np.array([2, 2, 5, 2, 1, 0, 1, 4, 9])

sinal_suavizado = savgol_filter(x, window_length=11, polyorder=3)
tempo = np.arange(len(x))  # Create tempo array
sinal = x  # Alias for compatibility
# Passo 3: Detecçao de Picos
# Detectar os picos no sinal suavizado
picos, _ = find_peaks(sinal_suavizado, height=0.5) # Ajuste o parametro de altura conforme necessario
# Passo 4: Analise de Resultados
# Plotar os resultados
plt.figure(figsize=(10, 6))
plt.plot(tempo, sinal, label='Sinal Original', alpha=0.5)
plt.plot(tempo, sinal_suavizado, label= 'Sinal Suavizado', color='orange')
plt.plot(tempo[picos], sinal_suavizado[picos], 'x', label='Picos Detectados', color='red')
plt.title('Analise de Dados com Savitzky-Golay e Detecçao de Picos')
plt.xlabel('Tempo')
plt.ylabel('Sinal')
plt.legend()
plt.grid()
plt.show()
display 

