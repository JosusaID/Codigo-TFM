#%% Importar librerias necesarias
import pandas as pd
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pywt
import calendar
from scipy.stats import sem, t
from matplotlib.dates import MonthLocator, DateFormatter
from statsmodels.tsa.stattools import kpss
import ta
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.trend import MACD, SMAIndicator, EMAIndicator, IchimokuIndicator, ADXIndicator, PSARIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator, StochRSIIndicator
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, MFIIndicator, VolumeWeightedAveragePrice
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from matplotlib.patches import Patch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import itertools
from statsmodels.tsa.stattools import pacf



#%%============================================================================
# Importar la base de datos numérica
#==============================================================================

# Configuración de fechas
start_date = "2007-01-01"
end_date = datetime.today().strftime('2024-12-31')

# 1. Descargar el índice S&P 500
sp500_data = yf.download("^GSPC", start=start_date, end=end_date)
sp500_data.reset_index(inplace=True)
sp500_data = sp500_data[['Date', 'Close']]
sp500_data.rename(columns={'Close': 'SP500_Close'}, inplace=True)

# 2. Descargar la lista de empresas actuales del S&P 500
sp500_tickers = pd.read_html(
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]

# Verificar columnas disponibles en el DataFrame descargado
print("Columnas disponibles:", sp500_tickers.columns)

# Ajustar columnas según disponibilidad
if 'Date first added' in sp500_tickers.columns:
    sp500_tickers = sp500_tickers[['Symbol', 'Date first added']]
    sp500_tickers['Date first added'] = pd.to_datetime(
        sp500_tickers['Date first added'], errors='coerce')
    tickers_pre_2001 = sp500_tickers[
        sp500_tickers['Date first added'] < "2001-01-01"]['Symbol'].tolist()
else:
    tickers_pre_2001 = sp500_tickers['Symbol'].tolist()

# 3. Descargar datos históricos de las empresas
all_data = {}
for ticker in tickers_pre_2001:
    print(f"Descargando datos para {ticker}...")
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data.reset_index(inplace=True)
        stock_data = stock_data[['Date', 'Close']]
        stock_data.rename(columns={'Close': ticker}, inplace=True)
        all_data[ticker] = stock_data
    except Exception as e:
        print(f"Error al descargar datos de {ticker}: {e}")

# 4. Combinar todos los datos en un único DataFrame
merged_data = sp500_data.copy()
for ticker, data in all_data.items():
    merged_data = merged_data.merge(data, on='Date', how='left')

# Exportar a Excel 
merged_data.to_csv('C:/Users/Josu/Desktop/BBDD_TFMMM.csv', index=False)

# Mostrar DataFrame final

print(merged_data)

# Para no andar descargando las bases con APIS,importamos la BBDD asi:

# Lee el archivo Excel
merged_data = pd.read_excel('C:/Users/Josu/Desktop/BBDD_TFM.xlsx', 
                            engine='openpyxl')

merged_data['Date'] = pd.to_datetime(merged_data['Date'], errors='coerce')

# Establecer la columna 'Date' como índice
merged_data.set_index('Date', inplace=True)

sp_data = pd.read_excel('C:/Users/Josu/Desktop/BBDD_TFM.xlsx',  
                        sheet_name = 'BBDD_SP500', engine='openpyxl')

sp_data['Date'] = pd.to_datetime(sp_data['Date'], errors='coerce')

# Establecer la columna 'Date' como índice
sp_data.set_index('Date', inplace=True)

df_inicial = pd.read_excel('C:/Users/Josu/Desktop/BBDD_TFM.xlsx',  
                        sheet_name = 'Empresas', engine='openpyxl')

#=============================================================================
 
# ANALISIS EXPLORATORIO

#=============================================================================

# Seleccionar la columna 'Cierre' del S&P 500 (está en la primera columna)
cierre_sp500 = sp_data['Cierre']

# Seleccionar las columnas correspondientes a las empresas (todas las columnas excepto la de 'Cierre')
empresas = merged_data

# Asignamos las columnas de 'Sector' que ya hemos añadido previamente a merged_data
sector = df_inicial.loc[0]

# Eliminamos la primera fila
sectores = sector[1:]

# Calcular las estadísticas básicas del Cierre del S&P 500
print("Estadísticas básicas del Cierre del S&P 500:")
print(cierre_sp500.describe())  # Descripción de las estadísticas básicas del Cierre del SP500
print("\n")

# Suponiendo que 'sector_dict' es el diccionario que mapea las empresas a los sectores
# Primero, asegúrate de que tienes un mapeo de sector para cada empresa
# Es probable que 'sector_dict' ya esté creado, pero lo revisamos aquí por claridad:

# Asegúrate de que las empresas y los sectores estén en el mismo orden
sector_dict = dict(zip(merged_data.columns[1:], sectores))

# Primero obtenemos las empresas (columnas del DataFrame excluyendo 'SP500_Close')
empresas_columns = merged_data.columns[1:]  # Excluye la columna 'SP500_Close'

# Mapear las empresas a sus sectores usando el diccionario sector_dict
# Esto genera un diccionario que mapea cada empresa con su sector
empresa_a_sector = {empresa: sector_dict.get(empresa, 'Desconocido') for empresa in empresas_columns}

# Ahora, asignamos la nueva columna 'Sector' al DataFrame
# Para cada fila, asignamos el sector correspondiente a cada empresa en cada columna
merged_data['Sector'] = merged_data.apply(lambda row: [empresa_a_sector[empresa] for empresa in row.index[1:]], axis=1)

# Verifica que la columna 'Sector' se haya añadido correctamente
print(merged_data[['Sector']].head())

# Seleccionar la columna 'Cierre' del S&P 500 (está en la primera columna)
cierre_sp500 = sp_data['Cierre']

# Seleccionar las columnas correspondientes a las empresas (todas las columnas excepto la de 'Cierre')
empresas = merged_data

# Asignamos las columnas de 'Sector' que ya hemos añadido previamente a merged_data
sector = df_inicial.loc[0]

# Eliminamos la primera fila
sectores = sector[1:]

# Calcular las estadísticas básicas del Cierre del S&P 500
print("Estadísticas básicas del Cierre del S&P 500:")
print(cierre_sp500.describe())  # Descripción de las estadísticas básicas del Cierre del SP500
print("\n")

# Suponiendo que 'sector_dict' es el diccionario que mapea las empresas a los sectores
# Primero, asegúrate de que tienes un mapeo de sector para cada empresa
# Es probable que 'sector_dict' ya esté creado, pero lo revisamos aquí por claridad:

# Asegúrate de que las empresas y los sectores estén en el mismo orden
# sector_dict = dict(zip(merged_data.columns[1:], sectores))

# Primero obtenemos las empresas (columnas del DataFrame excluyendo 'SP500_Close')
empresas_columns = merged_data.columns[1:]  # Excluye la columna 'SP500_Close'

# Mapear las empresas a sus sectores usando el diccionario sector_dict
# Esto genera un diccionario que mapea cada empresa con su sector
empresa_a_sector = {empresa: sector_dict.get(empresa, 'Desconocido') for empresa in empresas_columns}

# Ahora, asignamos la nueva columna 'Sector' al DataFrame
# Para cada fila, asignamos el sector correspondiente a cada empresa en cada columna
merged_data['Sector'] = merged_data.apply(lambda row: [empresa_a_sector[empresa] for empresa in row.index[1:]], axis=1)

# Verifica que la columna 'Sector' se haya añadido correctamente
print(merged_data[['Sector']].head())
# Extraer los sectores y niveles de capitalización (ya lo tienes en tu código)
empresas = df_inicial.columns[1:]  # Excluimos la columna 'Date'
sectores = df_inicial.set_index("Empresas")["Sector"]
  
niveles_cap = df_inicial.loc[1]
niveles_cap = niveles_cap[1:]  # Eliminamos la primera fila
capitalizacion = df_inicial.loc[2]
capitalizacion = capitalizacion[1:]  # Eliminamos la primera fila

import pandas as pd

# Cargar tus datos 'merged_data'
# Asegúrate de tener 'merged_data' cargado.

# Seleccionar la columna 'Cierre' del S&P 500 (está en la primera columna)
cierre_sp500 = sp_data['Cierre']

# Seleccionar las columnas correspondientes a las empresas (todas las columnas excepto la de 'Cierre')
empresas = merged_data.copy()

# Calcular las correlaciones entre cada serie temporal de las empresas y el 'Cierre' del S&P 500
correlaciones = empresas.corrwith(cierre_sp500)

# Asegúrate de que las empresas y los sectores estén en el mismo orden
sector_dict = dict(zip(merged_data.columns[1:], sectores))

# Asignamos la columna 'Sector' a las correlaciones usando el diccionario de sectores.
correlaciones_sector = correlaciones.to_frame().reset_index()
correlaciones_sector.columns = ['Empresa', 'Correlacion']

# Añadir la información de los sectores a las correlaciones
correlaciones_sector['Sector'] = correlaciones_sector['Empresa'].map(sector_dict)

# Agrupar las correlaciones por sector y calcular la media de la correlación dentro de cada grupo
correlaciones_sector_grouped = correlaciones_sector.groupby('Sector')['Correlacion'].mean()

# Mostrar los resultados ordenados
correlaciones_sector_grouped = correlaciones_sector_grouped.sort_values(ascending=False)

print("Correlaciones medias por sector con el S&P 500:")
print(correlaciones_sector_grouped)

#%% Representación gráfica del S&P 500
plt.figure(figsize=(10, 8))

# Graficar con un color azul cian (#00BCD4)
plt.plot(sp_data.index, sp_data['Cierre'], label='Precio SP500', 
         linewidth=2)

plt.title('Precio del S&P 500', fontsize=14)
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Precio SP500 (USD)', fontsize=12)

# Agregar leyenda en la parte superior izquierda
plt.legend(loc='upper left', bbox_to_anchor=(0, 1))

# Mejorar formato del eje X
plt.xticks(rotation=45)

plt.xlim(min(merged_data.index), max(merged_data.index))

# Ocultar los bordes superior y derecho del gráfico
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Mostrar el gráfico
plt.show()

#%% Representación gráfica media de las empresas del S&P 500

# Filtrar solo columnas numéricas (excluir datetime)
numeric_data = merged_data.select_dtypes(include=[np.number])

# Calcular la media diaria de los valores de cierre (filas)
daily_means = numeric_data.mean(axis=1)

# Calcular el error estándar diario
daily_sem = sem(numeric_data, axis=1, nan_policy='omit')

# Nivel de confianza
confidence = 0.95
n = numeric_data.shape[1]  # Número de columnas (empresas)
critical_value = t.ppf((1 + confidence) / 2, n - 1)

# Calcular el intervalo de confianza
interval = daily_sem * critical_value

# Graficar la media diaria con intervalo de confianza
plt.figure(figsize=(10, 8))
plt.plot(merged_data.index, daily_means, label='Media diaria de cierre', linewidth=2)
plt.fill_between(merged_data.index, daily_means - interval, daily_means + interval, alpha=0.3, label='Intervalo de confianza (95%)')
plt.title('Valores medios de las empresas del S&P 500', fontsize=14)
plt.xlim(min(merged_data.index), max(merged_data.index))
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Media de precios de cierre (USD)', fontsize=12)
plt.xticks(rotation=45)
plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()

#%% ANALISIS DE CORRELACIONES 
# Extraer los sectores y niveles de capitalización (ya lo tienes en tu código)
empresas = df_inicial.columns[1:]  # Excluimos la columna 'Date'
sectores = df_inicial.set_index("Empresas")["Sector"]
  
niveles_cap = df_inicial.loc[1]
niveles_cap = niveles_cap[1:]  # Eliminamos la primera fila
capitalizacion = df_inicial.loc[2]
capitalizacion = capitalizacion[1:]  # Eliminamos la primera fila

# Seleccionar la columna 'Cierre' del S&P 500 (está en la primera columna)
cierre_sp500 = sp_data['Cierre']

# Seleccionar las columnas correspondientes a las empresas (todas las columnas excepto la de 'Cierre')
empresas = merged_data.copy()

# Calcular las correlaciones entre cada serie temporal de las empresas y el 'Cierre' del S&P 500
correlaciones = empresas.corrwith(cierre_sp500)

# Asegúrate de que las empresas y los sectores estén en el mismo orden
sector_dict = dict(zip(merged_data.columns[1:], sectores))

# Asignamos la columna 'Sector' a las correlaciones usando el diccionario de sectores.
correlaciones_sector = correlaciones.to_frame().reset_index()
correlaciones_sector.columns = ['Empresa', 'Correlacion']

# Añadir la información de los sectores a las correlaciones
correlaciones_sector['Sector'] = correlaciones_sector['Empresa'].map(sector_dict)

# Agrupar las correlaciones por sector y calcular la media de la correlación dentro de cada grupo
correlaciones_sector_grouped = correlaciones_sector.groupby('Sector')['Correlacion'].mean()

# Mostrar los resultados ordenados
correlaciones_sector_grouped = correlaciones_sector_grouped.sort_values(ascending=False)

print("Correlaciones medias por sector con el S&P 500:")
print(correlaciones_sector_grouped)

#%% ESTADISTICAS BÁSICAS

# Seleccionar la columna 'Cierre' del S&P 500 (está en la primera columna)
cierre_sp500 = sp_data['Cierre']

# Seleccionar las columnas correspondientes a las empresas (todas las columnas excepto la de 'Cierre')
empresas = merged_data

# Asignamos las columnas de 'Sector' que ya hemos añadido previamente a merged_data
sector = df_inicial.loc[0]

# Eliminamos la primera fila
sectores = sector[1:]

# Calcular las estadísticas básicas del Cierre del S&P 500
print("Estadísticas básicas del Cierre del S&P 500:")
print(cierre_sp500.describe())  # Descripción de las estadísticas básicas del Cierre del SP500
print("\n")

# Primero obtenemos las empresas (columnas del DataFrame excluyendo 'SP500_Close')
empresas_columns = merged_data.columns[1:]  # Excluye la columna 'SP500_Close'

# Mapear las empresas a sus sectores usando el diccionario sector_dict
# Esto genera un diccionario que mapea cada empresa con su sector
empresa_a_sector = {empresa: sector_dict.get(empresa, 'Desconocido') for empresa in empresas_columns}

# Ahora, asignamos la nueva columna 'Sector' al DataFrame
# Para cada fila, asignamos el sector correspondiente a cada empresa en cada columna
merged_data['Sector'] = merged_data.apply(lambda row: [empresa_a_sector[empresa] for empresa in row.index[1:]], axis=1)

# Verifica que la columna 'Sector' se haya añadido correctamente
print(merged_data[['Sector']].head())

#%% ANALISIS DE VALORES ATIPICOS

# Valores atípicos a lo largo de los años del SP 500
sp_data['Año'] = sp_data.index.year

# Crear boxplots del cierre por año
plt.figure(figsize=(12, 6))
sp_data.boxplot(column='Cierre', by='Año', grid=False)
plt.title('Análisis de valores atípicos')
plt.suptitle('')
plt.xlabel('Año')
plt.ylabel('Precio de cierre')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()

# Valores atípicos del conjunto empresas por sector
df_inicial.boxplot(column='Capitalizacion', by='Sector', grid=False, figsize=(10,6))
plt.title('Distribución de la capitalización por sector')
plt.suptitle('')  # Elimina el subtítulo automático
plt.xlabel('Sector')
plt.ylabel('Capitalización')
plt.xticks(rotation=45)
plt.show()

#%% Comparativa de ganancias/perdidas entre años

# Crear una copia y asegurarse que el índice está en formato datetime
df = sp_data.copy()
df.index = pd.to_datetime(df.index, errors='coerce')

# Verificar que la conversión fue exitosa
print(df.index.dtype)  # Debe mostrar "datetime64[ns]"

# Extraer componentes de fecha
df['Año'] = df.index.year
df['Mes'] = df.index.month
df['Día'] = df.index.day

# ESTANDARIZACIÓN: Aplicar Z-score a cada año para hacerlos comparables
años_disponibles = sorted(df['Año'].unique())
df['Precio_Estandarizado'] = np.nan  # Inicializar columna

for año in años_disponibles:
    # Obtener datos de este año
    mask_año = df['Año'] == año
    datos_año = df.loc[mask_año, 'Cierre']
    
    # Aplicar estandarización (z-score): (x - media) / desviación estándar
    media = datos_año.mean()
    desv_std = datos_año.std()
    df.loc[mask_año, 'Precio_Estandarizado'] = (datos_año - media) / desv_std

# Crear una fecha artificial para alinear todos los años en un mismo eje X
df['Fecha_Base'] = pd.to_datetime('2000-' + df['Mes'].astype(str) + '-' + df['Día'].astype(str), errors='coerce')

# 1. GRÁFICO CON TODOS LOS AÑOS
plt.figure(figsize=(14, 7), facecolor='white')
ax = plt.subplot(111, facecolor='white')

# Eliminar las líneas de borde superior y derecho
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Generar colores usando un colormap, asignando colores distintos a cada año
cmap = plt.cm.coolwarm
colores = cmap(np.linspace(0, 1, len(años_disponibles)))

# Obtener fechas mínima y máxima para ajustar el eje X
fecha_min = df['Fecha_Base'].min()
fecha_max = df['Fecha_Base'].max()

# Trazar cada año
for i, año in enumerate(años_disponibles):
    data_año = df[df['Año'] == año]
    plt.plot(data_año['Fecha_Base'], data_año['Precio_Estandarizado'], 
             label=str(año), color=colores[i], linewidth=1.5, alpha=0.7)

# Configuración del eje X para mostrar los meses
plt.gca().xaxis.set_major_locator(MonthLocator())
plt.gca().xaxis.set_major_formatter(DateFormatter('%b'))

# Etiquetas de meses en español
meses_es = ['feb', 'mar', 'abr', 'may', 'jun', 
            'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
plt.gca().set_xticklabels(meses_es)

# Líneas verticales para separar los meses
for mes in range(2, 13):
    fecha = pd.Timestamp(f'2000-{mes:02d}-01')
    plt.axvline(fecha, color='gray', linestyle='--', alpha=0.3)

# Ajustar límites del eje X a los datos disponibles
plt.xlim(fecha_min, fecha_max)

# Configuración general del gráfico
plt.title('Estacionalidad de Índices (Todos los Años)', fontsize=16)
plt.ylabel('Valor Estandarizado (Z-score)', fontsize=12)
plt.grid(True, alpha=0.2)

# Mejorar la legibilidad de la leyenda con múltiples columnas
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
           ncol=min(8, len(años_disponibles)), fontsize=9)

plt.tight_layout()
plt.savefig('estacionalidad_todos_años_estandarizados.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. GRÁFICO CON AÑOS RELEVANTES
años_relevantes = [2008, 2009, 2020, 2022, 2023, 2024]
plt.figure(figsize=(14, 7), facecolor='white')
ax = plt.subplot(111, facecolor='white')

# Eliminar las líneas de borde superior y derecho
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Paleta de colores para años específicos
colores_específicos = {
    2008: 'darkred',    # Crisis financiera
    2009: 'darkorange',  # Continuación crisis
    2020: 'navy',       # COVID-19
    2022: 'darkgreen',  # Post-COVID/inflación
    2023: 'purple',     # Año reciente
    2024: 'royalblue'   # Año actual
}

# Trazar cada año relevante
for año in años_relevantes:
    if año in años_disponibles:  # Verificar que el año existe en el dataset
        data_año = df[df['Año'] == año]
        plt.plot(data_año['Fecha_Base'], data_año['Precio_Estandarizado'], 
                 label=str(año), color=colores_específicos.get(año, 'black'), 
                 linewidth=2.5)

# Configuración del eje X
plt.gca().xaxis.set_major_locator(MonthLocator())
plt.gca().xaxis.set_major_formatter(DateFormatter('%b'))

# Etiquetas de meses en español
plt.gca().set_xticklabels(meses_es)

# Líneas verticales para separar los meses
for mes in range(2, 13):
    fecha = pd.Timestamp(f'2000-{mes:02d}-01')
    plt.axvline(fecha, color='gray', linestyle='--', alpha=0.3)

# Ajustar límites del eje X a los datos disponibles
plt.xlim(fecha_min, fecha_max)

# Configuración general
plt.title('Estacionalidad de Índices - Años Relevantes', fontsize=16)
plt.ylabel('Valor Estandarizado (Z-score)', fontsize=12)
plt.grid(True, alpha=0.2)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('estacionalidad_años_clave_estandarizados.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Autocorrelacion S&P 500

# Asegúrate de que 'SP500_Close' sea una Serie (no un DataFrame)
sp500_series = sp_data['Cierre'].squeeze()  

# Eliminar valores NaN si los hay
sp500_series = sp500_series.dropna()

# Calcular la autocorrelación de la serie
autocorrelations_sp500 = [sp500_series.autocorr(lag=i) for i in range(1, 251)]  

# Crear el gráfico de autocorrelación
plt.figure(figsize=(12, 6))
plt.plot(range(1, 251), autocorrelations_sp500, marker='o', linestyle='-', 
         linewidth=2)

# Añadir detalles
plt.title('Análisis de autocorrelaciones del S&P 500', fontsize=14)
plt.xlabel('Retardos temporales', fontsize=12)
plt.ylabel('Autocorrelación', fontsize=12)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlim(0,251)
plt.ylim(0.94,1)
plt.grid(False)
plt.show()

#%% Autocorrelaciones parciales

# Asegúrate de que 'SP500_Close' sea una Serie (no un DataFrame)
sp500_series = sp_data['Cierre'].squeeze()  

# Eliminar valores NaN si los hay
sp500_series = sp500_series.dropna()

# Calcular la autocorrelación parcial de la serie
pacf_values = pacf(sp500_series, nlags=30)  # Aquí calculamos hasta 365 rezagos

# Crear el gráfico de autocorrelación parcial
plt.figure(figsize=(12, 6))
plt.plot(range(1, 31), pacf_values[1:], marker='o', linestyle='-', linewidth=2)  # omitimos el lag 0

# Añadir detalles
plt.title('Autocorrelación Parcial del S&P 500', fontsize=14)
plt.xlabel('Retardos temporales', fontsize=12)
plt.ylabel('Autocorrelación Parcial', fontsize=12)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlim(0, 30)
plt.ylim(-1, 1)
plt.grid(False)
plt.show()

#%% Analisis de tendencia (Coeficiente de Hurst)

import numpy as np
import matplotlib.pyplot as plt
from hurst import compute_Hc
import pandas as pd

# Supongamos que sp500_series es tu serie temporal con fechas en el índice
window_size = 126   # Tamaño de la ventana
hurst_values = []
years = []  # Lista para almacenar los años correspondientes

# Iteramos sobre la serie temporal en ventanas deslizantes
for start in range(0, len(sp_data['Cierre']) - window_size, window_size // 2):
    end = start + window_size
    window = sp_data['Cierre'].iloc[start:end]  # Seleccionar la ventana correctamente
    
    try:
        # Calcular el coeficiente de Hurst
        H, c, data_reg = compute_Hc(window)
        hurst_values.append(H)
        
        # Calcular el año en base a la posición de la ventana en la serie temporal
        year = sp_data['Cierre'].index[start + window_size // 2].year
        years.append(year)
        
    except Exception as e:
        print(f"Error en ventana {start}-{end}: {e}")
        hurst_values.append(np.nan)  
        years.append(np.nan)  

# Crear DataFrame para graficar
hurst_df = pd.DataFrame({"Hurst": hurst_values, "Year": years}).dropna()

plt.figure(figsize=(12, 6))

# Añadir un pequeño desplazamiento (por ejemplo, 0.2 o 0.3)
x_offset = 0.3
scatter = plt.scatter([x + 1 + x_offset for x in range(len(hurst_df))], 
                      hurst_df["Hurst"], 
                      c=hurst_df["Year"], 
                      cmap='viridis', 
                      label="Coef. de Hurst")

cbar = plt.colorbar(scatter)
cbar.set_label('Año')
plt.axhline(y=0.5, color='r', linestyle='-', label='Valor Hurst=0.5')
plt.title('Gráfico de dispersión del coeficiente de Hurst', fontsize=16)
plt.xlabel('Índice de ventana', fontsize=14)
plt.ylabel('Coeficiente de Hurst', fontsize=14)
plt.legend(loc='upper left')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Ajustar los límites del eje X para acomodar el desplazamiento
# plt.xlim(1, 34.5 + x_offset)  # Incluir el offset en el límite superior

plt.grid(False)
plt.show()

#%% Descomponer la serie temporal
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

# Aplicar STL con un periodo de 30 días (puedes ajustar según tu serie)
stl = STL(sp_data['Cierre'], period=252)  # Ajusta el periodo según tu serie
result = stl.fit()

# Obtener los límites del eje X
x_min = sp_data.index.min()
x_max = sp_data.index.max()

# Graficar los componentes
plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(sp_data.index, sp_data['Cierre'], label='Serie original')
plt.xlim(x_min, x_max)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(sp_data.index, result.trend, label='Tendencia', color='orange')
plt.xlim(x_min, x_max)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(sp_data.index, result.seasonal, label='Estacionalidad', color='green')
plt.xlim(x_min, x_max)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(sp_data.index, result.resid, label='Residuos', color='red')
plt.xlim(x_min, x_max)
plt.legend()

plt.tight_layout()
plt.show()

#%% Análisis de volatilidad
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

sp_data2 = sp_data.copy()

# Renombrar columnas si es necesario
sp_data2 = sp_data2.rename(columns={
    'Apertura': 'Open',
    'Maximo_dia': 'High',  
    'Minimo_dia': 'Low',   
    'Cierre': 'Close',
    'Volumen': 'Volume'
})

# Calcular medias móviles
sp_data2['SMA30'] = sp_data2['Close'].rolling(window=20).mean()
sp_data2['SMA100'] = sp_data2['Close'].rolling(window=70).mean()

# Configurar los colores de las medias móviles
mav_colors = ['blue', 'orange']

# Estilo personalizado con colores de velas y eliminación de bordes superiores y derechos
custom_style = mpf.make_mpf_style(
    base_mpl_style='fast', 
    gridcolor='none', 
    y_on_right=False,
    rc={'axes.spines.right': False, 'axes.spines.top': False},  # Elimina bordes superior y derecho
    marketcolors={
        'candle': {'up': 'green', 'down': 'red'},  # Velas verdes y rojas
        'edge': {'up': 'green', 'down': 'red'},    # Bordes de velas verdes y rojas
        'wick': {'up': 'green', 'down': 'red'},    # Mechas verdes y rojas
        'ohlc': {'up': 'green', 'down': 'red'},    # Barras OHLC si se usan
        'volume': {'up': 'green', 'down': 'red'},  # Volumen en verde y rojo
        'alpha': 1.0  # Evita error de transparencia
    }
)

# Obtener la primera y última fecha
fecha_inicial = sp_data2.index.min()
fecha_final = sp_data2.index.max()

# Gráfico de velas japonesas con ajuste del eje X
mpf.plot(sp_data2, type='candle', style=custom_style,
         mav=(30, 100), volume=False,
         title='Volatilidad del S&P 500',
         ylabel='Precio', ylabel_lower='Volumen',
         figsize=(12, 6), mavcolors=mav_colors,
         xlim=(fecha_inicial, fecha_final),  # Ajuste del eje X
         warn_too_much_data=100000)

#%% Prueba KPSS - Análisis de estacionariedad

def rolling_kpss(series, window, significance_level=0.05):
    """
    Calcula el test KPSS en ventanas deslizantes y evalúa estacionariedad dinámicamente.

    Parámetros:
    - series: Pandas Series con la serie temporal
    - window: Tamaño de la ventana deslizante
    - significance_level: Nivel de significancia (por defecto 5%)

    Retorna:
    - DataFrame con valores KPSS y si la serie es estacionaria en cada ventana
    - Lista de umbrales KPSS (uno por ventana)
    """
    kpss_values = []
    stationary_flags = []
    thresholds = []  

    for i in range(len(series) - window + 1):
        window_data = series[i : i + window]
        
        try:
            kpss_stat, _, _, critical_values = kpss(window_data, regression='c', nlags='auto')
            threshold = critical_values[f'{int(significance_level * 100)}%']  
            
            kpss_values.append(kpss_stat)
            stationary_flags.append(kpss_stat < threshold)  
            thresholds.append(threshold)  
            
        except ValueError:  
            kpss_values.append(np.nan)
            stationary_flags.append(np.nan)
            thresholds.append(np.nan)

    return pd.DataFrame({
        'Fecha': series.index[window - 1:],  
        'KPSS_Statistic': kpss_values,
        'Threshold': thresholds,
        'Es_Estacionaria': stationary_flags  
    })

def plot_kpss_results(df_kpss):
    """
    Grafica los valores KPSS en el tiempo usando un scatter plot y marca los umbrales reales.

    Parámetros:
    - df_kpss: DataFrame con los valores KPSS y umbrales dinámicos
    """
    plt.figure(figsize=(12, 6))
    
    # Scatter plot con puntos más pequeños (s=10)
    plt.scatter(df_kpss['Fecha'], df_kpss['KPSS_Statistic'], 
                c=['green' if x else 'red' for x in df_kpss['Es_Estacionaria']], 
                label='KPSS Statistic', alpha=0.7, s=10)  
    
    # Línea de umbral de estacionariedad (varía en el tiempo)
    plt.plot(df_kpss['Fecha'], df_kpss['Threshold'], color='black', linestyle='--', label='Umbral Estacionariedad')

    plt.xlabel('Fecha')
    plt.ylabel('KPSS Statistic')
    plt.title('Análisis de estacionariedad: Prueba KPSS')

    # Ajustar los ejes
    plt.xlim(df_kpss['Fecha'].min(), df_kpss['Fecha'].max())  # Ajuste del eje X
    plt.ylim(0, df_kpss['KPSS_Statistic'].max() * 1.145)  # Ajuste del eje Y con margen superior

    # Mover la leyenda arriba a la derecha
    plt.legend(loc='upper right')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.grid(False)
    plt.show()

# Calcular KPSS en ventanas de tamaño anual
df_kpss = rolling_kpss(sp_data['Cierre'], window=251)

# Graficar los resultados con scatter plot
plot_kpss_results(df_kpss)

#%% Calculo de caracteristicas

def calcular_indicadores_tecnicos(df):
    """
    Calcula indicadores técnicos completos para análisis financiero
    
    Parámetros:
    df (pandas.DataFrame): DataFrame con datos de precios OHLCV
                          Debe tener columnas: 'Apertura', 'Maximo', 'Minimo', 'Cierre', 'Volumen'
    
    Retorna:
    pandas.DataFrame: DataFrame con todos los indicadores calculados
    """
    # Copia el DataFrame para no modificar el original
    df_indicadores = df.copy()
    
    # Asegurarse de que los nombres de columnas sean los esperados por ta
    columnas_requeridas = {
        'Apertura': 'open',
        'Maximo_dia': 'high', 
        'Minimo_dia': 'low',
        'Cierre': 'close',
        'Volumen': 'volume'
    }
    
    # Crear columnas temporales con nombres estándar para la librería TA
    for nombre_original, nombre_ta in columnas_requeridas.items():
        if nombre_original in df_indicadores.columns:
            df_indicadores[nombre_ta] = df_indicadores[nombre_original]
    
    # Cálculos básicos
    df_indicadores['Cambio_Absoluto'] = df_indicadores['close'].diff()
    df_indicadores['Cambio_Porcentual'] = df_indicadores['close'].pct_change() * 100
    df_indicadores['Retorno_Log'] = np.log(df_indicadores['close'] / df_indicadores['close'].shift(1))
    
    # Medias Móviles Simples (SMA)
    for periodo in [20, 50, 200]:
        df_indicadores[f'SMA_{periodo}'] = SMAIndicator(close=df_indicadores['close'], window=periodo).sma_indicator()
    
    # Medias Móviles Exponenciales (EMA)
    for periodo in [20, 50]:
        df_indicadores[f'EMA_{periodo}'] = EMAIndicator(close=df_indicadores['close'], window=periodo).ema_indicator()
    
    # Índice de Fuerza Relativa (RSI)
    for periodo in [14, 30]:
        df_indicadores[f'RSI_{periodo}'] = RSIIndicator(close=df_indicadores['close'], window=periodo).rsi()
    
    # MACD (Moving Average Convergence Divergence)
    macd = MACD(close=df_indicadores['close'], window_slow=26, window_fast=12, window_sign=9)
    df_indicadores['MACD_Linea'] = macd.macd()
    df_indicadores['MACD_Senal'] = macd.macd_signal()
    df_indicadores['MACD_Histograma'] = macd.macd_diff()
    
    # Bandas de Bollinger
    bollinger = BollingerBands(close=df_indicadores['close'], window=20, window_dev=2)
    df_indicadores['Bollinger_Superior'] = bollinger.bollinger_hband()
    df_indicadores['Bollinger_Media'] = bollinger.bollinger_mavg()
    df_indicadores['Bollinger_Inferior'] = bollinger.bollinger_lband()
    df_indicadores['Bollinger_%B'] = bollinger.bollinger_pband()
    df_indicadores['Bollinger_Ancho'] = bollinger.bollinger_wband()
    
    # Oscilador Estocástico
    # Estocástico rápido
    stoch_rapido = StochasticOscillator(high=df_indicadores['high'], low=df_indicadores['low'], 
                                         close=df_indicadores['close'], window=14, smooth_window=3)
    df_indicadores['Estocastico_Rapido_%K'] = stoch_rapido.stoch()
    df_indicadores['Estocastico_Rapido_%D'] = stoch_rapido.stoch_signal()
    
    # Estocástico lento (basado en el %D del estocástico rápido)
    df_indicadores['Estocastico_Lento_%K'] = df_indicadores['Estocastico_Rapido_%D']
    df_indicadores['Estocastico_Lento_%D'] = df_indicadores['Estocastico_Lento_%K'].rolling(window=3).mean()
    
    # Momentum
    for periodo in [5, 10, 20]:
        df_indicadores[f'Momentum_{periodo}'] = df_indicadores['close'].diff(periodo)
        df_indicadores[f'Momentum_Pct_{periodo}'] = (df_indicadores['close'] / df_indicadores['close'].shift(periodo) - 1) * 100
    
    # Índice de Canal de Mercaderías (CCI)
    for periodo in [14, 30]:
        df_indicadores[f'CCI_{periodo}'] = ta.trend.CCIIndicator(
            high=df_indicadores['high'], 
            low=df_indicadores['low'], 
            close=df_indicadores['close'], 
            window=periodo
        ).cci()
    
    # Tasa de Cambio (ROC)
    for periodo in [5, 10, 20]:
        df_indicadores[f'ROC_{periodo}'] = ROCIndicator(close=df_indicadores['close'], window=periodo).roc()
    
    # Williams %R
    df_indicadores['Williams_%R'] = WilliamsRIndicator(
        high=df_indicadores['high'], 
        low=df_indicadores['low'], 
        close=df_indicadores['close'], 
        lbp=14
    ).williams_r()
    
    # Desviación Estándar
    for periodo in [10, 20, 30]:
        df_indicadores[f'Desviacion_Estandar_{periodo}'] = df_indicadores['close'].rolling(window=periodo).std()
    
    # ATR (Average True Range)
    df_indicadores['ATR_14'] = AverageTrueRange(
        high=df_indicadores['high'], 
        low=df_indicadores['low'], 
        close=df_indicadores['close'], 
        window=14
    ).average_true_range()
    
    # ADX (Average Directional Index)
    adx = ADXIndicator(
        high=df_indicadores['high'], 
        low=df_indicadores['low'], 
        close=df_indicadores['close'], 
        window=14
    )
    df_indicadores['ADX'] = adx.adx()
    df_indicadores['+DI'] = adx.adx_pos()
    df_indicadores['-DI'] = adx.adx_neg()
    
    # On-Balance Volume (OBV)
    df_indicadores['OBV'] = OnBalanceVolumeIndicator(
        close=df_indicadores['close'], 
        volume=df_indicadores['volume']
    ).on_balance_volume()
    
    # Ichimoku Cloud
    ichimoku = IchimokuIndicator(
        high=df_indicadores['high'], 
        low=df_indicadores['low'],
        window1=9, 
        window2=26, 
        window3=52
    )
    df_indicadores['Ichimoku_Tenkan_sen'] = ichimoku.ichimoku_conversion_line()
    df_indicadores['Ichimoku_Kijun_sen'] = ichimoku.ichimoku_base_line()
    df_indicadores['Ichimoku_Senkou_span_A'] = ichimoku.ichimoku_a()
    df_indicadores['Ichimoku_Senkou_span_B'] = ichimoku.ichimoku_b()
    # Note: Chikou Span está desplazado
    df_indicadores['Ichimoku_Chikou_span'] = df_indicadores['close'].shift(-26)
    
    # VWAP (Volume Weighted Average Price) - Intradiario
    if 'volume' in df_indicadores.columns:
        vwap = VolumeWeightedAveragePrice(
            high=df_indicadores['high'],
            low=df_indicadores['low'],
            close=df_indicadores['close'],
            volume=df_indicadores['volume'],
            window=14
        )
        df_indicadores['VWAP'] = vwap.volume_weighted_average_price()
    
    # Retornos logarítmicos
    df_indicadores['Retorno_Log_Diario'] = df_indicadores['Retorno_Log']
    # Retorno semanal (5 días)
    df_indicadores['Retorno_Log_Semanal'] = np.log(df_indicadores['close'] / df_indicadores['close'].shift(5))
    # Retorno mensual (21 días aprox)
    df_indicadores['Retorno_Log_Mensual'] = np.log(df_indicadores['close'] / df_indicadores['close'].shift(21))
    
    # Retornos acumulados
    for periodo in [5, 10, 20]:
        df_indicadores[f'Retorno_Acumulado_{periodo}'] = (
            (1 + df_indicadores['Retorno_Log']).rolling(window=periodo).apply(np.prod, raw=True) - 1
        ) * 100
    
    # Volatilidad histórica (anualizada)
    for periodo in [21, 63, 252]:  # Aprox. 1 mes, 3 meses, 1 año
        df_indicadores[f'Volatilidad_{periodo}'] = df_indicadores['Retorno_Log'].rolling(
            window=periodo
        ).std() * np.sqrt(252)  # Anualizado
    
    # Ratio de Sharpe (simple)
    risk_free_rate = 0.02 / 252  # Tasa libre de riesgo diaria (2% anual)
    for periodo in [21, 63, 252]:  # Aprox. 1 mes, 3 meses, 1 año
        returns_mean = df_indicadores['Retorno_Log'].rolling(window=periodo).mean()
        returns_std = df_indicadores['Retorno_Log'].rolling(window=periodo).std()
        df_indicadores[f'Sharpe_Ratio_{periodo}'] = (returns_mean - risk_free_rate) / returns_std * np.sqrt(252)
    
    # Cruce de Medias Móviles
    df_indicadores['SMA50_Cruza_SMA200'] = np.where(
        (df_indicadores['SMA_50'] > df_indicadores['SMA_200']) & 
        (df_indicadores['SMA_50'].shift(1) <= df_indicadores['SMA_200'].shift(1)),
        1, 0
    )
    df_indicadores['SMA200_Cruza_SMA50'] = np.where(
        (df_indicadores['SMA_50'] < df_indicadores['SMA_200']) & 
        (df_indicadores['SMA_50'].shift(1) >= df_indicadores['SMA_200'].shift(1)),
        1, 0
    )
    
    # Distancia porcentual desde máximos/mínimos
    # Máximo y mínimo de 52 semanas (aproximadamente 252 días de trading)
    df_indicadores['Max_52_Semanas'] = df_indicadores['close'].rolling(window=252).max()
    df_indicadores['Min_52_Semanas'] = df_indicadores['close'].rolling(window=252).min()
    df_indicadores['Distancia_%_Max_52'] = (df_indicadores['close'] / df_indicadores['Max_52_Semanas'] - 1) * 100
    df_indicadores['Distancia_%_Min_52'] = (df_indicadores['close'] / df_indicadores['Min_52_Semanas'] - 1) * 100
    
    # Indicador Parabólico SAR
    psar = PSARIndicator(
        high=df_indicadores['high'],
        low=df_indicadores['low'],
        close=df_indicadores['close'],
        step=0.02,
        max_step=0.2
    )
    df_indicadores['PSAR'] = psar.psar()
    df_indicadores['PSAR_Tendencia'] = np.where(df_indicadores['PSAR'] < df_indicadores['close'], 1, -1)
    
    # Índice de Flujo de Dinero (MFI)
    if 'volume' in df_indicadores.columns:
        df_indicadores['MFI'] = MFIIndicator(
            high=df_indicadores['high'],
            low=df_indicadores['low'],
            close=df_indicadores['close'],
            volume=df_indicadores['volume'],
            window=14
        ).money_flow_index()
    
    # Características de temporalidad (asumiendo que el índice es de tipo datetime)
    if isinstance(df_indicadores.index, pd.DatetimeIndex):
        df_indicadores['Dia_Semana'] = df_indicadores.index.dayofweek
        df_indicadores['Mes'] = df_indicadores.index.month
        df_indicadores['Trimestre'] = df_indicadores.index.quarter
        
        # Variables dummy para día de la semana
        for i in range(5):  # 0-4 para días laborables
            df_indicadores[f'DiaSemana_{i}'] = np.where(df_indicadores['Dia_Semana'] == i, 1, 0)
        
        # Variables dummy para mes
        for i in range(1, 13):
            df_indicadores[f'Mes_{i}'] = np.where(df_indicadores['Mes'] == i, 1, 0)
    
    # Canales de Donchian
    for periodo in [20, 50]:
        df_indicadores[f'Donchian_Alto_{periodo}'] = df_indicadores['high'].rolling(window=periodo).max()
        df_indicadores[f'Donchian_Bajo_{periodo}'] = df_indicadores['low'].rolling(window=periodo).min()
        df_indicadores[f'Donchian_Medio_{periodo}'] = (
            df_indicadores[f'Donchian_Alto_{periodo}'] + df_indicadores[f'Donchian_Bajo_{periodo}']
        ) / 2
    
    # Filtro de Kalman (simplificado como una media móvil ponderada)
    # Para un verdadero filtro de Kalman se necesitaría implementar el algoritmo completo
    df_indicadores['Kalman_Simple'] = df_indicadores['close'].ewm(alpha=0.1).mean()
    
    # Eliminar columnas temporales usadas para cálculos
    for nombre_ta in columnas_requeridas.values():
        if nombre_ta in df_indicadores.columns and nombre_ta not in df.columns:
            df_indicadores.drop(columns=[nombre_ta], inplace=True)
    
    return df_indicadores

# Ejemplo de uso:
# Asumiendo que tienes un DataFrame llamado 'data' con datos OHLCV
resultados = calcular_indicadores_tecnicos(sp_data)
resultados = resultados.dropna()

# Eliminamos una variable adicional
resultados = resultados[1:]

resultados = resultados.drop(columns=['Maximo_dia', 'Minimo_dia', 'Apertura'])

# Realizamos una descripción de las características
res_resultados = resultados.describe().T

# Nos quedamos con la columna de la desviación para ver el aporte de cada una
std_resultados = res_resultados['std']

# Seleccionamos solo las variables que tengan un std >= 2
list_carac = [var for var in std_resultados.index if std_resultados[var] >= 2]

# Filtrar las columnas de res_resultados que están en list_carac
resultados_filtrado = resultados[list_carac]

#%% Selección de características mediante mRMR
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from scipy.stats import pearsonr

def mRMR_corrected(X, y, K):
    """
    Implementación corregida del algoritmo mRMR (minimum Redundancy Maximum Relevance)
    
    Parameters:
    X: DataFrame con las características
    y: Series con la variable objetivo
    K: Número de características a seleccionar
    
    Returns:
    selected_features: Lista de características seleccionadas
    mrmr_scores: Scores mRMR para cada iteración
    relevance_scores: Scores de relevancia (correlación con y)
    redundancy_scores: Scores de redundancia promedio
    """
    
    # Calcular relevancia: correlación absoluta con la variable objetivo
    relevance = {}
    for col in X.columns:
        corr_coef, _ = pearsonr(X[col], y)
        relevance[col] = abs(corr_coef)
    
    # Inicializar
    selected_features = []
    remaining_features = list(X.columns)
    mrmr_scores = []
    relevance_scores = []
    redundancy_scores = []
    
    # Matriz de correlaciones entre características
    feature_corr = X.corr().abs()
    
    for i in range(K):
        if i == 0:
            # Primera característica: la de mayor relevancia
            best_feature = max(remaining_features, key=lambda x: relevance[x])
            redundancy = 0
            mrmr_score = relevance[best_feature]
        else:
            # Para las siguientes características: maximizar mRMR = Relevancia - Redundancia
            best_score = -np.inf
            best_feature = None
            
            for feature in remaining_features:
                # Relevancia: correlación con y
                rel_score = relevance[feature]
                
                # Redundancia: correlación promedio con características ya seleccionadas
                redundancy_vals = [feature_corr.loc[feature, selected_feat] 
                                 for selected_feat in selected_features]
                red_score = np.mean(redundancy_vals)
                
                # Score mRMR
                mrmr_score = rel_score - red_score
                
                if mrmr_score > best_score:
                    best_score = mrmr_score
                    best_feature = feature
                    redundancy = red_score
            
            mrmr_score = best_score
        
        # Agregar la mejor característica
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        
        # Guardar scores para análisis
        mrmr_scores.append(mrmr_score)
        relevance_scores.append(relevance[best_feature])
        redundancy_scores.append(redundancy)
        
        print(f"Iteración {i+1}: {best_feature}")
        print(f"  - Relevancia: {relevance[best_feature]:.4f}")
        print(f"  - Redundancia: {redundancy:.4f}")
        print(f"  - Score mRMR: {mrmr_score:.4f}")
        print()
    
    return selected_features, mrmr_scores, relevance_scores, redundancy_scores

# Alternativa usando mutual information (más robusta)
from sklearn.feature_selection import mutual_info_regression

def mRMR_mutual_info(X, y, K):
    """
    Implementación de mRMR usando mutual information
    Más robusta para relaciones no lineales
    """
    
    # Calcular mutual information con la variable objetivo
    mi_scores = mutual_info_regression(X, y, random_state=42)
    relevance = dict(zip(X.columns, mi_scores))
    
    selected_features = []
    remaining_features = list(X.columns)
    mrmr_scores = []
    
    for i in range(K):
        if i == 0:
            # Primera característica: mayor mutual information
            best_feature = max(remaining_features, key=lambda x: relevance[x])
            mrmr_score = relevance[best_feature]
        else:
            best_score = -np.inf
            best_feature = None
            
            for feature in remaining_features:
                # Relevancia
                rel_score = relevance[feature]
                
                # Redundancia: mutual information promedio con características seleccionadas
                redundancy_vals = []
                for selected_feat in selected_features:
                    mi_red = mutual_info_regression(
                        X[[feature]], X[selected_feat], random_state=42
                    )[0]
                    redundancy_vals.append(mi_red)
                
                red_score = np.mean(redundancy_vals)
                mrmr_score = rel_score - red_score
                
                if mrmr_score > best_score:
                    best_score = mrmr_score
                    best_feature = feature
            
            mrmr_score = best_score
        
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        mrmr_scores.append(mrmr_score)
        
        print(f"Iteración {i+1}: {best_feature} (mRMR: {mrmr_score:.4f})")
    
    return selected_features, mrmr_scores

# CÓDIGO PRINCIPAL CORREGIDO
print("=== SELECCIÓN DE CARACTERÍSTICAS CON mRMR CORREGIDO ===\n")

# 1️⃣ Preparar los datos
X = resultados_filtrado.drop(columns=["Cierre"])
y = resultados_filtrado["Cierre"]

# 2️⃣ Estandarizar solo las columnas numéricas
numeric_columns = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# 3️⃣ Aplicar mRMR corregido
K = 10
print("🔍 Aplicando mRMR con correlación de Pearson:")
print("-" * 50)
selected_features_corr, mrmr_scores_corr, relevance_scores, redundancy_scores = mRMR_corrected(X_scaled, y, K)

print("\n🔍 Aplicando mRMR con Mutual Information:")
print("-" * 50)
selected_features_mi, mrmr_scores_mi = mRMR_mutual_info(X_scaled, y, K)

# 4️⃣ Crear tabla de resultados
results_df = pd.DataFrame({
    'Rank': range(1, K+1),
    'Feature_Pearson': selected_features_corr,
    'mRMR_Score_Pearson': mrmr_scores_corr,
    'Relevance': relevance_scores,
    'Redundancy': redundancy_scores,
    'Feature_MI': selected_features_mi,
    'mRMR_Score_MI': mrmr_scores_mi
})

print("\n🚀 RESULTADOS FINALES:")
print("=" * 80)
print(results_df.round(4))

# 5️⃣ Comparar con tu método original (solo para referencia)
print("\n📊 COMPARACIÓN CON MÉTODO ORIGINAL:")
print("-" * 50)
F_stats = pd.Series(f_regression(X_scaled, y)[0], index=X_scaled.columns)
print("Top 10 características por F-statistic:")
top_f_stats = F_stats.nlargest(K)
for i, (feature, f_stat) in enumerate(top_f_stats.items(), 1):
    print(f"{i:2d}. {feature}: {f_stat:.2f}")

# 6️⃣ Crear dataset final con las mejores características
print(f"\n✅ Creando dataset con las {K} mejores características (método Pearson)...")
df_selected = resultados_filtrado[selected_features_mi + ["Cierre"]].copy()
print(f"Dimensiones del dataset final: {df_selected.shape}")
print(f"Características seleccionadas: {selected_features_corr}")

#%% GRID SEARCH MODELOS LSTM - GRU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# --- CLASES Y FUNCIONES ---

class FinancialTimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, prediction_horizon=1):
        self.features = data[:, :-1]
        self.targets = data[:, -1]
        self.seq_len = sequence_length
        self.prediction_horizon = prediction_horizon

    def __len__(self):
        return int(len(self.features) - self.seq_len - self.prediction_horizon)

    def __getitem__(self, idx):
        X = self.features[idx:idx+self.seq_len]
        y = self.targets[idx+self.seq_len+self.prediction_horizon-1]
        return torch.FloatTensor(X), torch.FloatTensor([y])

class FinancialLSTM(nn.Module):
    def __init__(self, input_features, hidden_size, num_layers, dropout, 
                 bidirectional=False, lstm_type='LSTM'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm_type = lstm_type
        
        # Seleccionar tipo de RNN
        if lstm_type == 'LSTM':
            self.rnn = nn.LSTM(input_features, hidden_size, num_layers, 
                              dropout=dropout, bidirectional=bidirectional, 
                              batch_first=True)
        elif lstm_type == 'GRU':
            self.rnn = nn.GRU(input_features, hidden_size, num_layers, 
                             dropout=dropout, bidirectional=bidirectional, 
                             batch_first=True)
        else:
            self.rnn = nn.RNN(input_features, hidden_size, num_layers, 
                             dropout=dropout, bidirectional=bidirectional, 
                             batch_first=True)
        
        # Dropout adicional
        self.dropout = nn.Dropout(dropout)
        
        # Capa de salida (considerar bidirecionalidad)
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, fc_input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input_size // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_features)
        if self.lstm_type == 'LSTM':
            rnn_out, (hidden, cell) = self.rnn(x)
        else:
            rnn_out, hidden = self.rnn(x)
        
        # Tomar la última salida temporal
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        last_output = self.dropout(last_output)
        
        # Predicción final
        output = self.fc(last_output)
        return output
    
def train_model_with_curves(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']
    return train_losses, val_losses


def prepare_data(df_selected, selected_features, sequence_length, 
                 prediction_horizon=1, train_ratio=0.8):
    data = df_selected[selected_features + ['Cierre']].values
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(data[:, :-1])
    target_scaler.fit(data[:, -1].reshape(-1, 1))
    data_normalized = np.column_stack([
        feature_scaler.transform(data[:, :-1]),
        target_scaler.transform(data[:, -1].reshape(-1, 1))
    ])
    train_size = int(len(data_normalized) * train_ratio)
    train_data = data_normalized[:train_size]
    test_data = data_normalized[train_size:]
    return train_data, test_data, feature_scaler, target_scaler

def evaluate_model(model, data_loader, target_scaler):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(batch_y.cpu().numpy().flatten())

    predictions = target_scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)).flatten()
    actuals = target_scaler.inverse_transform(np.array(
        actuals).reshape(-1, 1)).flatten()

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    r2 = r2_score(actuals, predictions)

    # Directional Accuracy
    direction_pred = np.sign(np.diff(predictions))
    direction_real = np.sign(np.diff(actuals))
    directional_accuracy = np.mean(direction_pred == direction_real)

    # Sharpe Ratio
    returns = np.diff(predictions) / predictions[:-1]
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)

    return predictions, actuals, {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy,
        'Sharpe_Ratio': sharpe_ratio
    }

def train_model(model, train_loader, val_loader, epochs=500, 
                learning_rate=0.001, weight_decay=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, 
                                                     factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15

    epoch_iter = tqdm(range(epochs), desc="Épocas", unit="época", leave=False)
    for epoch in epoch_iter:
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            # Gradient clipping para LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        epoch_iter.set_postfix({
            "Train Loss": f"{train_loss:.5f}",
            "Val Loss": f"{val_loss:.5f}",
            "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(torch.load('best_lstm_model.pth'))
    return train_losses[-1], val_losses[-1], optimizer.param_groups[0]['lr']

# --- PARÁMETROS DEL GRID SEARCH PARA LSTM ---

param_grid = {
    'hidden_size': [32, 64],
    'num_layers': [1, 2, 3],
    'dropout': [0.1, 0.2],
    'bidirectional': [False, True],
    'lstm_type': ['LSTM', 'GRU'],  # Comparar LSTM vs GRU
    'learning_rate': [0.0001, 0.0005],
    'weight_decay': [1e-5, 1e-4],
    'batch_size': [32, 64],
    'sequence_length': [30, 60]
}

# --- EJECUCIÓN PRINCIPAL CON GRID SEARCH ---

# Supongamos que tienes tu DataFrame df_selected ya cargado
# Descomentar la siguiente línea si necesitas cargar datos
# df_selected = pd.read_csv('tu_archivo.csv')

# Obtener automáticamente las columnas y excluir el target
columna_objetivo = 'Cierre'
selected_features = [col for col in list(df_selected.columns) if col != 
                     columna_objetivo]
print("Características seleccionadas:", selected_features)

# Lista para guardar resultados de todos los modelos
resultados_grid = []

# Generar todas las combinaciones de hiperparámetros
keys, values = zip(*param_grid.items())
combinaciones = list(itertools.product(*values))

print(f"Total de combinaciones a evaluar: {len(combinaciones)}")

# Barra de progreso para el grid search
grid_bar = tqdm(combinaciones, desc="Grid Search LSTM", unit="modelo")

for i, comb in enumerate(grid_bar):
    params = dict(zip(keys, comb))
    grid_bar.set_postfix(**{k: str(v)[:10] for k, v in list(params.items(
        ))[:3]})
    
    try:
        # INICIO: medir tiempo
        start_time = time.time()
        
        # Preparar datos con la ventana temporal actual
        train_data, test_data, feature_scaler, target_scaler = prepare_data(
            df_selected, selected_features, params['sequence_length']
        )
        
        # Crear datasets y dataloaders
        train_dataset = FinancialTimeSeriesDataset(train_data, params[
            'sequence_length'])
        test_dataset = FinancialTimeSeriesDataset(test_data, params[
            'sequence_length'])
        
        # Verificar si hay suficientes datos
        if len(train_dataset) < 10 or len(test_dataset) < 5:
            print(f"Saltando configuración {i+1}: datos insuficientes")
            continue
        
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=params['batch_size'], 
                                shuffle=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=params['batch_size'], 
                              shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], 
                               shuffle=False, drop_last=True)
        
        # Inicializar modelo LSTM
        input_features = len(selected_features)
        model = FinancialLSTM(
            input_features=input_features,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            bidirectional=params['bidirectional'],
            lstm_type=params['lstm_type']
        ).to(device)
        
        # Entrenar modelo
        train_loss, val_loss, final_lr = train_model(
            model, train_loader, val_loader,
            epochs=500, 
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Evaluar modelo
        predictions, actuals, metrics = evaluate_model(model, test_loader, 
                                                       target_scaler)
        
        # FINAL: medir tiempo
        elapsed_time = time.time() - start_time
        
        # Guardar resultados
        resultados_grid.append({
            **params,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **metrics,
            'final_learning_rate': final_lr,
            'execution_time_sec': elapsed_time,
            'model_complexity': params[
                'hidden_size'] * params['num_layers'] * (2 if params[
                    'bidirectional'] else 1)
        })
        
    except Exception as e:
        print(f"Error en configuración {i+1}: {str(e)}")
        continue

# Convertir resultados a DataFrame y guardar en CSV
resultados_df = pd.DataFrame(resultados_grid)
resultados_df.to_csv('resultados_grid_search_lstm.csv', index=False,
                     encoding='utf-8-sig')

print(f"\nGrid Search completado. {len(resultados_df)} configuraciones evaluadas.")
print("Resultados guardados en 'resultados_grid_search_lstm.csv'")

# === ANÁLISIS DE RESULTADOS ===

# Top 5 modelos por RMSE
top_5_rmse = resultados_df.nsmallest(5, 'RMSE')
print("\n=== TOP 5 MODELOS POR RMSE ===")
print(top_5_rmse[['lstm_type', 'hidden_size', 'num_layers', 'bidirectional', 
                  'RMSE', 'MAE', 'R2', 'Directional_Accuracy']].to_string())

# Top 5 modelos por R²
top_5_r2 = resultados_df.nlargest(5, 'R2')
print("\n=== TOP 5 MODELOS POR R² ===")
print(top_5_r2[['lstm_type', 'hidden_size', 'num_layers', 'bidirectional', 
               'RMSE', 'MAE', 'R2', 'Directional_Accuracy']].to_string())

# === IDENTIFICAR MEJOR MODELO (por menor RMSE) ===
mejor_fila = resultados_df.loc[resultados_df['MAPE'].idxmin()]
mejor_params = {k: mejor_fila[k] for k in param_grid.keys()}

# Entrenamiento de mejores modelos 

print(f"\n=== MEJORES PARÁMETROS ===")
for param, valor in mejor_params.items():
    print(f"{param}: {valor}")

print(f"\nMétricas del mejor modelo:")
print(f"RMSE: {mejor_fila['RMSE']:.4f}")
print(f"MAE: {mejor_fila['MAE']:.4f}")
print(f"R²: {mejor_fila['R2']:.4f}")
print(f"Precisión Direccional: {mejor_fila['Directional_Accuracy']:.4f}")
print(f"Tiempo de ejecución: {mejor_fila['execution_time_sec']:.2f} segundos")

# Obtener los dos mejores lstm_type según el RMSE promedio
top_lstm_types = resultados_df.groupby('lstm_type')['MAPE'].mean().sort_values().head(2).index.tolist()

model_metrics = {}

for lstm_type in top_lstm_types:
    print(f"\nEntrenando modelo tipo: {lstm_type}")

    mejor_params_tipo = resultados_df[resultados_df['lstm_type'] == lstm_type].sort_values(by='MAPE').iloc[0]

    # Limpiar tipos
    mejor_params_clean = {
        'sequence_length': int(mejor_params_tipo['sequence_length']),
        'batch_size': int(mejor_params_tipo['batch_size']),
        'hidden_size': int(mejor_params_tipo['hidden_size']),
        'num_layers': int(mejor_params_tipo['num_layers']),
        'dropout': float(mejor_params_tipo['dropout']),
        'bidirectional': bool(mejor_params_tipo['bidirectional']),
        'learning_rate': float(mejor_params_tipo['learning_rate']),
        'weight_decay': float(mejor_params_tipo['weight_decay']),
        'lstm_type': mejor_params_tipo['lstm_type']
    }

    # Preparar datos
    train_data, test_data, feature_scaler, target_scaler = prepare_data(
        df_selected, selected_features, mejor_params_clean['sequence_length']
    )

    train_dataset = FinancialTimeSeriesDataset(train_data, mejor_params_clean['sequence_length'])
    test_dataset = FinancialTimeSeriesDataset(test_data, mejor_params_clean['sequence_length'])

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=mejor_params_clean['batch_size'], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=mejor_params_clean['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=mejor_params_clean['batch_size'], shuffle=False)

    # Crear modelo
    model = FinancialLSTM(
        input_features=len(selected_features),
        hidden_size=mejor_params_clean['hidden_size'],
        num_layers=mejor_params_clean['num_layers'],
        dropout=mejor_params_clean['dropout'],
        bidirectional=mejor_params_clean['bidirectional'],
        lstm_type=mejor_params_clean['lstm_type']
    ).to(device)

    # Entrenar
    train_losses, val_losses = train_model_with_curves(
        model, train_loader, val_loader, epochs=500,
        learning_rate=mejor_params_clean['learning_rate'],
        weight_decay=mejor_params_clean['weight_decay']
    )

    # Evaluar
    predictions, actuals, final_metrics = evaluate_model(model, test_loader, target_scaler)

    # Guardar
    model_metrics[lstm_type] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'predictions': predictions,
        'actuals': actuals,
        'metrics': final_metrics
    }

# Más graficos de los modelos

for lstm_type in top_lstm_types:
    predictions = model_metrics[lstm_type]['predictions']
    actuals = model_metrics[lstm_type]['actuals']
    r2 = model_metrics[lstm_type]['metrics']['R2']

    # 1. Línea: reales vs predicciones
    plt.figure(figsize=(10, 4))
    plt.plot(actuals, label='Real', alpha=0.8, linewidth=1.5)
    plt.plot(predictions, label='Predicción', alpha=0.8, linewidth=1.5)
    plt.title(f'{lstm_type} - Predicciones vs Valores Reales')
    plt.xlabel('Índice de tiempo')
    plt.ylabel('Valor')
    plt.xlim(0, len(predictions))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    # 2. Error absoluto
    abs_error = np.abs(actuals - predictions)
    plt.figure(figsize=(10, 4))
    plt.plot(abs_error, color='orange', alpha=0.7)
    plt.title(f'{lstm_type} - Error Absoluto')
    plt.xlabel('Índice Temporal')
    plt.ylabel('|Real - Predicción|')
    plt.xlim(0, len(abs_error))
    plt.grid(True, alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    # 3. Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals, predictions, alpha=0.5, s=5)
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.title(f'{lstm_type} - Scatter Plot\nR² = {r2:.4f}')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
        
#%% BAYESIAN SEARCH MODELOS LSTM - GRU

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import optuna
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# --- REUTILIZAR CLASES DEL GRID SEARCH ---

class FinancialTimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, prediction_horizon=1):
        self.features = data[:, :-1]
        self.targets = data[:, -1]
        self.seq_len = sequence_length
        self.prediction_horizon = prediction_horizon

    def __len__(self):
        return int(len(self.features) - self.seq_len - self.prediction_horizon)

    def __getitem__(self, idx):
        X = self.features[idx:idx+self.seq_len]
        y = self.targets[idx+self.seq_len+self.prediction_horizon-1]
        return torch.FloatTensor(X), torch.FloatTensor([y])

class FinancialRNN(nn.Module):
    def __init__(self, input_features, hidden_size, num_layers, dropout, 
                 bidirectional=False, lstm_type='LSTM'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm_type = lstm_type
        
        if lstm_type == 'LSTM':
            self.rnn = nn.LSTM(input_features, hidden_size, num_layers, 
                              dropout=dropout, bidirectional=bidirectional, 
                              batch_first=True)
        elif lstm_type == 'GRU':
            self.rnn = nn.GRU(input_features, hidden_size, num_layers, 
                             dropout=dropout, bidirectional=bidirectional, 
                             batch_first=True)
        else:
            self.rnn = nn.RNN(input_features, hidden_size, num_layers, 
                             dropout=dropout, bidirectional=bidirectional, 
                             batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, fc_input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input_size // 2, 1)
        )
        
    def forward(self, x):
        if self.lstm_type == 'LSTM':
            rnn_out, (hidden, cell) = self.rnn(x)
        else:
            rnn_out, hidden = self.rnn(x)
        
        last_output = rnn_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output


def prepare_data(df_selected, selected_features, sequence_length, 
                 prediction_horizon=1, train_ratio=0.8):
    data = df_selected[selected_features + ['Cierre']].values
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(data[:, :-1])
    target_scaler.fit(data[:, -1].reshape(-1, 1))
    data_normalized = np.column_stack([
        feature_scaler.transform(data[:, :-1]),
        target_scaler.transform(data[:, -1].reshape(-1, 1))
    ])
    train_size = int(len(data_normalized) * train_ratio)
    train_data = data_normalized[:train_size]
    test_data = data_normalized[train_size:]
    return train_data, test_data, feature_scaler, target_scaler

def evaluate_model(model, data_loader, target_scaler):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(batch_y.cpu().numpy().flatten())

    predictions = target_scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)).flatten()
    actuals = target_scaler.inverse_transform(np.array(
        actuals).reshape(-1, 1)).flatten()

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    r2 = r2_score(actuals, predictions)

    direction_pred = np.sign(np.diff(predictions))
    direction_real = np.sign(np.diff(actuals))
    directional_accuracy = np.mean(direction_pred == direction_real)

    returns = np.diff(predictions) / predictions[:-1]
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)

    return predictions, actuals, {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy,
        'Sharpe_Ratio': sharpe_ratio
    }

def train_model_with_curves(model, train_loader, val_loader, epochs=500, 
                           learning_rate=0.001, weight_decay=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, 
                                                     factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_{model.lstm_type}.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(torch.load(f'best_model_{model.lstm_type}.pth'))
    return train_losses, val_losses

# --- OBTENER MEJORES HIPERPARÃMETROS DEL GRID SEARCH ---

def get_best_hyperparameters_from_grid(resultados_df):
    """
    Extrae los mejores hiperparÃ¡metros de cada tipo de RNN del grid search
    """
    best_lstm = resultados_df[resultados_df['lstm_type'] == 'LSTM'].nsmallest(1, 'MAPE').iloc[0]
    best_gru = resultados_df[resultados_df['lstm_type'] == 'GRU'].nsmallest(1, 'MAPE').iloc[0]
    
    return best_lstm, best_gru

# --- CONFIGURAR BAYESIAN SEARCH CON OPTUNA ---

def create_objective_function(lstm_type, df_selected, selected_features, best_params):
    """
    Crea la funciÃ³n objetivo para Optuna basada en los mejores parÃ¡metros
    """
    def objective(trial):
        # Definir rangos alrededor de los mejores valores
        hidden_size = trial.suggest_int('hidden_size', 
                                       max(16, int(best_params['hidden_size'] * 0.5)), 
                                       int(best_params['hidden_size'] * 1.5))
        num_layers = trial.suggest_int('num_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 
                                     max(0.05, best_params['dropout'] * 0.5), 
                                     min(0.5, best_params['dropout'] * 2))
        bidirectional = trial.suggest_categorical('bidirectional', [True, False])
        learning_rate = trial.suggest_float('learning_rate', 
                                           best_params['learning_rate'] * 0.1, 
                                           best_params['learning_rate'] * 10, 
                                           log=True)
        weight_decay = trial.suggest_float('weight_decay', 
                                          best_params['weight_decay'] * 0.1, 
                                          best_params['weight_decay'] * 10, 
                                          log=True)
        batch_size = trial.suggest_int('batch_size', 
                                      max(16, int(best_params['batch_size'] * 0.5)), 
                                      int(best_params['batch_size'] * 2))
        sequence_length = trial.suggest_int('sequence_length', 
                                           max(10, int(best_params['sequence_length'] * 0.5)), 
                                           int(best_params['sequence_length'] * 1.5))
        
        try:
            # Preparar datos
            train_data, test_data, feature_scaler, target_scaler = prepare_data(
                df_selected, selected_features, sequence_length
            )
            
            train_dataset = FinancialTimeSeriesDataset(train_data, sequence_length)
            test_dataset = FinancialTimeSeriesDataset(test_data, sequence_length)
            
            if len(train_dataset) < 10 or len(test_dataset) < 5:
                return 100.0  # PenalizaciÃ³n alta
            
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_subset, val_subset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, 
                                    shuffle=True, drop_last=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, 
                                  shuffle=False, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                   shuffle=False, drop_last=True)
            
            # Crear modelo
            model = FinancialLSTM(
                input_features=len(selected_features),
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                lstm_type=lstm_type
            ).to(device)
            
            # Entrenar (menos Ã©pocas para acelerar)
            train_losses, val_losses = train_model_with_curves(
                model, train_loader, val_loader, epochs=200,
                learning_rate=learning_rate, weight_decay=weight_decay
            )
            
            # Evaluar
            predictions, actuals, metrics = evaluate_model(model, test_loader, target_scaler)
            
            # Guardar mÃ©tricas adicionales en el trial
            trial.set_user_attr('RMSE', metrics['RMSE'])
            trial.set_user_attr('MAE', metrics['MAE'])
            trial.set_user_attr('R2', metrics['R2'])
            trial.set_user_attr('Directional_Accuracy', metrics['Directional_Accuracy'])
            trial.set_user_attr('Sharpe_Ratio', metrics['Sharpe_Ratio'])
            
            # Retornar MAPE (lo que queremos minimizar)
            return metrics['MAPE']
            
        except Exception as e:
            print(f"Error en optimizaciÃ³n: {str(e)}")
            return 100.0  # PenalizaciÃ³n alta
    
    return objective

# --- EJECUCIÃ“N PRINCIPAL ---

# Cargar resultados del grid search
# Asume que ya tienes resultados_df del grid search
# Si no, carga el archivo CSV:
# resultados_df = pd.read_csv('resultados_grid_search_lstm.csv')

# Obtener caracterÃ­sticas
columna_objetivo = 'Cierre'
selected_features = [col for col in list(df_selected.columns) if col != columna_objetivo]

# Obtener mejores hiperparÃ¡metros de cada tipo
best_lstm_params, best_gru_params = get_best_hyperparameters_from_grid(resultados_df)

print("Mejores parÃ¡metros LSTM del Grid Search:")
print(f"MAPE: {best_lstm_params['MAPE']:.4f}")
print(f"Hidden Size: {best_lstm_params['hidden_size']}")
print(f"Num Layers: {best_lstm_params['num_layers']}")
print(f"Dropout: {best_lstm_params['dropout']}")
print(f"Bidirectional: {best_lstm_params['bidirectional']}")

print("\nMejores parÃ¡metros GRU del Grid Search:")
print(f"MAPE: {best_gru_params['MAPE']:.4f}")
print(f"Hidden Size: {best_gru_params['hidden_size']}")
print(f"Num Layers: {best_gru_params['num_layers']}")
print(f"Dropout: {best_gru_params['dropout']}")
print(f"Bidirectional: {best_gru_params['bidirectional']}")

# Almacenar resultados de Bayesian Search
bayesian_results = {}

# --- BAYESIAN SEARCH PARA LSTM ---
print("\n" + "="*50)
print("BAYESIAN SEARCH PARA LSTM")
print("="*50)

# Crear estudio de Optuna para LSTM
lstm_study = optuna.create_study(direction='minimize', 
                                study_name='LSTM_Bayesian_Search')

# Crear funciÃ³n objetivo para LSTM
lstm_objective = create_objective_function('LSTM', df_selected, selected_features, best_lstm_params)

# Ejecutar optimizaciÃ³n bayesiana para LSTM
lstm_study.optimize(lstm_objective, n_trials=25, show_progress_bar=True)

# Extraer mejores parÃ¡metros LSTM
best_lstm_bayesian = lstm_study.best_params.copy()
best_lstm_bayesian['lstm_type'] = 'LSTM'
best_lstm_bayesian['mape'] = lstm_study.best_value

# Obtener mÃ©tricas adicionales del mejor trial
best_lstm_trial = lstm_study.best_trial
for attr_name, attr_value in best_lstm_trial.user_attrs.items():
    best_lstm_bayesian[attr_name] = attr_value

print(f"Mejor MAPE LSTM (Bayesian): {lstm_study.best_value:.4f}")
print(f"Mejores parÃ¡metros LSTM: {best_lstm_bayesian}")

# --- BAYESIAN SEARCH PARA GRU ---
print("\n" + "="*50)
print("BAYESIAN SEARCH PARA GRU")
print("="*50)

# Crear estudio de Optuna para GRU
gru_study = optuna.create_study(direction='minimize', 
                               study_name='GRU_Bayesian_Search')

# Crear funciÃ³n objetivo para GRU
gru_objective = create_objective_function('GRU', df_selected, selected_features, best_gru_params)

# Ejecutar optimizaciÃ³n bayesiana para GRU
gru_study.optimize(gru_objective, n_trials=25, show_progress_bar=True)

# Extraer mejores parÃ¡metros GRU
best_gru_bayesian = gru_study.best_params.copy()
best_gru_bayesian['lstm_type'] = 'GRU'
best_gru_bayesian['mape'] = gru_study.best_value

# Obtener mÃ©tricas adicionales del mejor trial
best_gru_trial = gru_study.best_trial
for attr_name, attr_value in best_gru_trial.user_attrs.items():
    best_gru_bayesian[attr_name] = attr_value

print(f"Mejor MAPE GRU (Bayesian): {gru_study.best_value:.4f}")
print(f"Mejores parÃ¡metros GRU: {best_gru_bayesian}")

# --- ENTRENAR MODELOS FINALES CON MEJORES PARÃMETROS ---

final_models = {}
model_configs = [best_lstm_bayesian, best_gru_bayesian]

for config in model_configs:
    lstm_type = config['lstm_type']
    print(f"\nEntrenando modelo final {lstm_type}...")
    
    # Preparar datos
    train_data, test_data, feature_scaler, target_scaler = prepare_data(
        df_selected, selected_features, config['sequence_length']
    )
    
    train_dataset = FinancialTimeSeriesDataset(train_data, config['sequence_length'])
    test_dataset = FinancialTimeSeriesDataset(test_data, config['sequence_length'])
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], 
                            shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], 
                          shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                           shuffle=False, drop_last=True)
    
    # Crear modelo
    model = FinancialLSTM(
        input_features=len(selected_features),
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional'],
        lstm_type=lstm_type
    ).to(device)
    
    # Entrenar modelo final con mÃ¡s Ã©pocas
    train_losses, val_losses = train_model_with_curves(
        model, train_loader, val_loader, epochs=500,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Evaluar modelo final
    predictions, actuals, metrics = evaluate_model(model, test_loader, target_scaler)
    
    # Guardar resultados
    final_models[lstm_type] = {
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'predictions': predictions,
        'actuals': actuals,
        'metrics': metrics,
        'model': model
    }

# --- GENERAR GRÃFICOS DE COMPARACIÃ“N ---

# Determinar el mejor modelo por MAPE
best_model_type = min(final_models.keys(), 
                     key=lambda x: final_models[x]['metrics']['MAPE'])

print(f"\n" + "="*60)
print("RESULTADOS FINALES - BAYESIAN SEARCH")
print("="*60)

for model_type in final_models.keys():
    metrics = final_models[model_type]['metrics']
    print(f"\n{model_type} - MÃ©tricas Finales:")
    print(f"  MAPE: {metrics['MAPE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  RÂ²: {metrics['R2']:.4f}")
    print(f"  PrecisiÃ³n Direccional: {metrics['Directional_Accuracy']:.4f}")

print(f"\nMejor modelo por MAPE: {best_model_type}")

# GRÃFICO 1: Curvas de pÃ©rdida
plt.figure(figsize=(12, 6))
colors = ['blue', 'red']
for i, model_type in enumerate(final_models.keys()):
    train_losses = final_models[model_type]['train_losses']
    val_losses = final_models[model_type]['val_losses']
    mape = final_models[model_type]['metrics']['MAPE']
    
    plt.plot(train_losses, color=colors[i], linestyle='-', alpha=0.7, 
             label=f'{model_type} Train (MAPE: {mape:.3f})')
    plt.plot(val_losses, color=colors[i], linestyle='--', alpha=0.7, 
             label=f'{model_type} Val')

plt.title('Curvas de PÃ©rdida - Bayesian Search (Mejores Modelos)', fontsize=14)
plt.xlabel('Ã‰poca')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# GRÃFICO 2: DistribuciÃ³n de MAPE
plt.figure(figsize=(8, 6))
mape_values = [final_models[model_type]['metrics']['MAPE'] for model_type in final_models.keys()]
model_names = list(final_models.keys())

bars = plt.bar(model_names, mape_values, color=['skyblue', 'lightcoral'])
plt.title('ComparaciÃ³n MAPE - Bayesian Search', fontsize=14)
plt.ylabel('MAPE (%)')
plt.grid(True, alpha=0.3, axis='y')

# AÃ±adir valores en las barras
for bar, value in zip(bars, mape_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.show()

# GRÃFICO 3: Predicciones vs Reales para el mejor modelo
best_model_data = final_models[best_model_type]
predictions = best_model_data['predictions']
actuals = best_model_data['actuals']
metrics = best_model_data['metrics']

plt.figure(figsize=(18, 6))

# Subplot 1: Series temporales
plt.subplot(1, 3, 1)
plt.plot(actuals, label='Real', alpha=0.8, linewidth=1.5, color='blue')
plt.plot(predictions, label='PredicciÃ³n', alpha=0.8, linewidth=1.5, color='red')
plt.title(f'{best_model_type} - Predicciones vs Valores Reales\nMAPE: {metrics["MAPE"]:.3f}%')
plt.xlabel('Ãndice de tiempo')
plt.ylabel('Valor')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Error absoluto
plt.subplot(1, 3, 2)
abs_error = np.abs(actuals - predictions)
plt.plot(abs_error, color='orange', alpha=0.7)
plt.title(f'Error Absoluto\nMAE: {metrics["MAE"]:.3f}')
plt.xlabel('Ãndice Temporal')
plt.ylabel('|Real - PredicciÃ³n|')
plt.grid(True, alpha=0.3)

# Subplot 3: Scatter plot
plt.subplot(1, 3, 3)
plt.scatter(actuals, predictions, alpha=0.5, s=2, color='purple')
plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
plt.title(f'Scatter Plot\nRÂ² = {metrics["R2"]:.4f}')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# GRÃFICO 4: ComparaciÃ³n de todas las mÃ©tricas
metrics_names = ['MAPE', 'RMSE', 'MAE', 'R2', 'Directional_Accuracy']
lstm_metrics = [final_models['LSTM']['metrics'][metric] for metric in metrics_names]
gru_metrics = [final_models['GRU']['metrics'][metric] for metric in metrics_names]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, lstm_metrics, width, label='LSTM', color='skyblue')
bars2 = ax.bar(x + width/2, gru_metrics, width, label='GRU', color='lightcoral')

ax.set_xlabel('MÃ©tricas')
ax.set_ylabel('Valores')
ax.set_title('ComparaciÃ³n de MÃ©tricas - LSTM vs GRU (Bayesian Search)')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# AÃ±adir valores en las barras
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

add_value_labels(bars1)
add_value_labels(bars2)

plt.tight_layout()
plt.show()

# Guardar resultados del Bayesian Search
bayesian_results_df = pd.DataFrame([
    {**final_models['LSTM']['config'], **final_models['LSTM']['metrics'], 'search_type': 'Bayesian'},
    {**final_models['GRU']['config'], **final_models['GRU']['metrics'], 'search_type': 'Bayesian'}
])

bayesian_results_df.to_csv('resultados_bayesian_search_lstm_gru.csv', index=False, encoding='utf-8-sig')
print(f"\nResultados del Bayesian Search guardados en 'resultados_bayesian_search_lstm_gru.csv'")

# %% ANALISIS DE SENSIBILIDAD: EMPRESAS CON MAYOR VOLATILIDAD

#%% Análisis de sensibilidad

def calcular_betas_sectoriales(merged_data, sectores, sp500_data):
    """
    Calcula la beta de cada sector en relación con el S&P 500 y lo representa gráficamente.
    Utiliza la media de los retornos de las empresas por sector.
    """
    # Verificar datos antes de procesar
    print(f"Verificación de datos antes de procesar:")
    print(f"merged_data tiene {merged_data.isna().sum().sum()} valores NaN")
    print(f"sp500_data tiene {sp500_data.isna().sum()} valores NaN")
    
    # Asegurar que los datos son numéricos y limpiar NaN
    merged_data = merged_data.apply(pd.to_numeric, errors='coerce')
    
    # Usar una copia para no modificar los datos originales
    sp500_clean = sp500_data.copy()
    
    # Calcular retornos diarios sin eliminar NaN inicialmente
    retornos_empresas = merged_data.pct_change(fill_method=None)
    retorno_sp500 = sp500_clean.pct_change(fill_method=None)
    
    # Mostrar información sobre los retornos calculados
    print(f"Después de pct_change:")
    print(f"retornos_empresas tiene {retornos_empresas.isna().sum().sum()} valores NaN")
    print(f"retorno_sp500 tiene {retorno_sp500.isna().sum()} valores NaN")
    
    # Eliminar la primera fila (que será NaN debido a pct_change)
    retornos_empresas = retornos_empresas.iloc[1:]
    retorno_sp500 = retorno_sp500.iloc[1:]
    
    # Verificar si aún hay datos para procesar
    if retornos_empresas.empty or retorno_sp500.empty:
        print("Error: No hay suficientes datos después de calcular retornos")
        return pd.DataFrame(columns=['Beta Sectorial'])

    # Alinear fechas
    fechas_comunes = retornos_empresas.index.intersection(retorno_sp500.index)
    if len(fechas_comunes) == 0:
        print("Error: No hay fechas comunes entre los datos de empresas y el S&P 500")
        return pd.DataFrame(columns=['Beta Sectorial'])
    
    print(f"Número de fechas comunes: {len(fechas_comunes)}")
    
    retornos_empresas = retornos_empresas.loc[fechas_comunes]
    retorno_sp500 = retorno_sp500.loc[fechas_comunes]

    # Crear un diccionario que mapea empresas a sectores
    sector_dict = sectores.to_dict()
    
    # Verificar el mapeo de sectores
    print(f"Número de empresas con sector asignado: {sum(1 for empresa in sector_dict if empresa in retornos_empresas.columns)}")
    
    # Agrupar empresas por sector y calcular el retorno medio
    # En lugar de sumar, recopilamos los retornos por sector para calcular la media
    empresas_por_sector = {}
    for empresa, sector in sector_dict.items():
        if empresa in retornos_empresas.columns and pd.notna(sector) and sector != "":
            # Convertir cada columna a numérico para asegurar compatibilidad
            retorno_empresa = pd.to_numeric(retornos_empresas[empresa], errors='coerce')
            
            if sector not in empresas_por_sector:
                empresas_por_sector[sector] = [retorno_empresa]
            else:
                empresas_por_sector[sector].append(retorno_empresa)

    # Calcular la media de los retornos por sector
    retornos_sector = {}
    for sector, lista_retornos in empresas_por_sector.items():
        if lista_retornos:  # Verificar que hay datos
            # Convertir lista de series a DataFrame
            df_sector = pd.concat(lista_retornos, axis=1)
            # Calcular la media por fila (para cada fecha)
            retornos_sector[sector] = df_sector.mean(axis=1)
    
    # Verificar sectores
    print(f"Número de sectores calculados: {len(retornos_sector)}")
    
    if not retornos_sector:
        print("Error: No se pudo asignar ninguna empresa a un sector")
        return pd.DataFrame(columns=['Beta Sectorial'])

    # Convertir a DataFrame
    retornos_sector_df = pd.DataFrame(retornos_sector)
    
    # Calcular beta de cada sector
    betas_sector = {}
    for sector in retornos_sector_df.columns:
        # Eliminar NaN para este sector específico
        sector_data = retornos_sector_df[sector].dropna()
        sp500_data_aligned = retorno_sp500.loc[sector_data.index].dropna()
        
        # Encontrar índices comunes después de eliminar NaN
        common_idx = sector_data.index.intersection(sp500_data_aligned.index)
        
        if len(common_idx) > 5:  # Asegurar suficientes datos para una regresión válida
            sector_data = sector_data.loc[common_idx]
            sp500_data_aligned = sp500_data_aligned.loc[common_idx]
            
            # Verificar finalmente que no hay NaN
            if not sector_data.isna().any() and not sp500_data_aligned.isna().any():
                try:
                    slope, _, _, _, _ = stats.linregress(sp500_data_aligned, sector_data)
                    betas_sector[sector] = slope
                    print(f"Beta calculada para sector {sector}: {slope}")
                except Exception as e:
                    print(f"Error al calcular beta para sector {sector}: {e}")
            else:
                print(f"Advertencia: Aún hay NaN en los datos del sector {sector}")
        else:
            print(f"Advertencia: Menos de 5 puntos de datos para el sector {sector} ({len(common_idx)} encontrados)")

    # Verificar si se calculó alguna beta
    if not betas_sector:
        print("Error: No se pudo calcular beta para ningún sector")
        return pd.DataFrame(columns=['Beta Sectorial'])

    # Mostrar información sobre las betas calculadas
    print(f"Número de betas sectoriales calculadas: {len(betas_sector)}")
    
    # Convertir a DataFrame y ordenar
    betas_df = pd.DataFrame.from_dict(betas_sector, orient='index', columns=['Beta Sectorial'])
    betas_df = betas_df.sort_values(by='Beta Sectorial', ascending=False)

    return betas_df

def visualizar_betas_sectoriales(betas_df):
    """
    Genera gráficos para visualizar las betas sectoriales.
    """
    if betas_df.empty:
        print("No hay datos de betas para visualizar")
        return
        
    plt.figure(figsize=(12, 6))
    ax = betas_df.sort_values('Beta Sectorial', ascending=False).plot(kind='bar', color='darkblue', legend=False)
    
    plt.axhline(y=1, color='r', linestyle='--', label='Beta = 1 (S&P 500)')
    plt.title('Beta de Cada Sector respecto al S&P 500')
    plt.ylabel('Beta')
    # plt.ylim(0, max(7, betas_df['Beta Sectorial'].max() * 1.1))  # Ajustar ylim dinámicamente
    plt.xticks(rotation=45, ha="right")
    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Diagnóstico de los datos
print("Información de diagnóstico inicial:")
print(f"Forma de empresas: {empresas.shape if isinstance(empresas, pd.DataFrame) else 'No es un DataFrame'}")
print(f"Número de sectores: {len(sectores) if isinstance(sectores, pd.Series) else 'No es una Series'}")
print(f"Longitud de cierre_sp500: {len(cierre_sp500) if isinstance(cierre_sp500, pd.Series) else 'No es una Series'}")

# Verificar los primeros valores para debug
print("\nPrimeros valores de los datos:")
try:
    print("Primeras filas de empresas:")
    print(empresas.iloc[:3, :3])
    
    print("\nPrimeros valores de sectores:")
    print(sectores.head())
    
    print("\nPrimeros valores de cierre_sp500:")
    print(cierre_sp500.head())
except Exception as e:
    print(f"Error al imprimir valores: {e}")

# Calcular betas sectoriales
betas_sectores = calcular_betas_sectoriales(empresas, sectores, cierre_sp500)

# Mostrar resultados
print("\nBetas Sectoriales:")
print(betas_sectores)

# Mostrar gráfico
visualizar_betas_sectoriales(betas_sectores)


def visualizar_empresas_mayor_beta(resultados_empresas, num_sectores=8, num_empresas=5):
    """
    Genera visualizaciones de las empresas con mayor Beta dentro de cada sector,
    ordenadas de mayor a menor Beta.
    """
    if resultados_empresas.empty:
        print("No hay datos de empresas para visualizar")
        return
    
    # Identificar los sectores principales según la suma de las Betas
    top_sectores = resultados_empresas.groupby('Sector')['Beta'].sum().nlargest(num_sectores).index
    
    # Crear un gráfico para cada sector
    plt.figure(figsize=(15, 10))
    
    for i, sector in enumerate(top_sectores):
        # Obtener las empresas con mayor Beta para este sector
        empresas_sector = resultados_empresas[resultados_empresas['Sector'] == sector].nlargest(num_empresas, 'Beta')
        empresas_sector = empresas_sector.sort_values('Beta')  # Ordenar de menor a mayor para que la barra mayor quede arriba
        
        # Crear subplot
        plt.subplot(len(top_sectores) // 2 + len(top_sectores) % 2, 2, i+1)
        
        # Crear barras horizontales
        bars = plt.barh(empresas_sector.index, empresas_sector['Beta'], color='salmon')
        
        # Añadir etiquetas SOLO con valor de Beta
        for bar in bars:
            plt.text(
                bar.get_width() * 1.01, 
                bar.get_y() + bar.get_height()/2, 
                f'β={bar.get_width():.2f}', 
                va='center'
            )
        
        plt.title(f'Sector: {sector}')
        plt.xlabel('Beta')
        plt.grid(False)
        plt.tight_layout()
    
    plt.suptitle('Empresas con Mayor Beta por Sector', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Crear un gráfico global de las empresas con mayor Beta
    plt.figure(figsize=(14, 10))  # MÁS ESPACIO
    
    # Tomar las top empresas globales con mayor Beta
    top_empresas = resultados_empresas.nlargest(20, 'Beta')
    top_empresas = top_empresas.sort_values('Beta')  # Ordenar de menor a mayor para la gráfica
    
    # Crear un mapa de colores basado en sectores
    sectores_unicos = top_empresas['Sector'].unique()
    color_map = dict(zip(sectores_unicos, sns.color_palette("tab10", len(sectores_unicos))))
    bar_colors = [color_map[sector] for sector in top_empresas['Sector']]
    
    # Crear barras
    bars = plt.barh(top_empresas.index, top_empresas['Beta'], color=bar_colors)
    
    # Añadir etiquetas SOLO con valor de Beta
    for bar in bars:
        plt.text(
            bar.get_width() * 1.01, 
            bar.get_y() + bar.get_height()/2, 
            f'β={bar.get_width():.2f}', 
            va='center'
        )
    
    plt.title('Top 20 Empresas con Mayor Beta en el S&P 500')
    plt.xlabel('Beta')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Crear leyenda FUERA DEL GRÁFICO
    legend_elements = [Patch(facecolor=color_map[sector], label=sector) for sector in sectores_unicos]
    plt.legend(
        handles=legend_elements, 
        loc='center left', 
        bbox_to_anchor=(1, 0.5),
        title="Sectores"
    )
    
    # Eliminar bordes superiores y derechos del gráfico
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Ajustar límites del eje X según el valor máximo
    max_beta = top_empresas['Beta'].max()
    plt.xlim(0, max_beta * 1.05)
    
    plt.tight_layout()
    plt.show()




print("Información de diagnóstico inicial:")
print(f"Forma de empresas: {empresas.shape if isinstance(empresas, pd.DataFrame) else 'No es un DataFrame'}")
print(f"Número de sectores: {len(sectores) if isinstance(sectores, pd.Series) else 'No es una Series'}")
print(f"Longitud de cierre_sp500: {len(cierre_sp500) if isinstance(cierre_sp500, pd.Series) else 'No es una Series'}")

# Verificar los primeros valores para debug
print("\nPrimeros valores de los datos:")
try:
    print("Primeras filas de empresas:")
    print(empresas.iloc[:3, :3])

    print("\nPrimeros valores de sectores:")
    print(sectores.head())

    print("\nPrimeros valores de cierre_sp500:")
    print(cierre_sp500.head())
except Exception as e:
    print(f"Error al imprimir valores: {e}")

# Calcular betas sectoriales y obtener datos adicionales
betas_sectores, empresas_por_sector, retornos_empresas, retorno_sp500 = calcular_betas_sectoriales(empresas, sectores, cierre_sp500)

# Mostrar resultados de betas sectoriales
print("\nBetas Sectoriales:")
print(betas_sectores)

# Visualizar betas sectoriales
visualizar_betas_sectoriales(betas_sectores)

# Calcular la influencia sectorial con capitalización simulada
# datos_sector = calcular_influencia_sectorial(betas_sectores, None)  # None para generar capitalización simulada

# # Mostrar resultados de influencia sectorial
# print("\nInfluencia de Cada Sector en el S&P 500:")
# print(datos_sector)

# # Visualizar influencia sectorial
# visualizar_influencia_sectorial(datos_sector)

# # Calcular y visualizar las empresas con mayor Beta dentro de cada sector
# try:
#     # Crear dict de sector para cada empresa
#     sector_mapping = {}
#     for empresa, lista_series in empresas_por_sector.items():
#         for serie in lista_series:
#             empresa_name = serie.name
#             sector_mapping[empresa_name] = empresa

#     # Calcular empresas influyentes (aquí puedes usar la misma función si ya calcula Beta también)
#     empresas_betas = calcular_empresas_influyentes(
#         empresas_por_sector, 
#         retornos_empresas, 
#         retorno_sp500, 
#         sector_mapping
#     )

#     # Mostrar resultados
#     print("\nEmpresas con Mayor Beta dentro de cada sector:")
#     print(empresas_betas.head(20))

#     # Visualizar empresas con mayor Beta (usando tu función modificada que te pasé antes)
#     visualizar_empresas_mayor_beta(empresas_betas)

# except Exception as e:
#     print(f"Error al calcular la Beta de las empresas: {e}")
#     import traceback

# %% ANÁLISIS DE ESTACIONALIDAD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

def calculate_seasonal_pattern(sp_data):
    """
    Calcula el patrón estacional promedio anual
    sp_data: DataFrame con fechas en el índice y columna 'Cierre'
    """
    # Crear copia del DataFrame
    df = sp_data.copy()
    
    # Extraer día del año del índice
    df['day_of_year'] = df.index.dayofyear
    
    # Calcular promedio por día del año usando la columna 'Cierre'
    seasonal_pattern = df.groupby('day_of_year')['Cierre'].mean()
    
    return seasonal_pattern

def continuous_wavelet_transform(data, scales=None, wavelet='cmor1.5-1.0', sampling_period=1):
    """
    Aplica la transformada wavelet continua usando la ondícula de Morlet
    """
    if scales is None:
        scales = np.arange(1, 51)
    
    # Aplicar CWT
    coefficients, frequencies = pywt.cwt(data, scales, wavelet, sampling_period)
    
    return coefficients, frequencies, scales

def plot_wavelet_seasonality(sp_data):
    """
    Genera el gráfico de estacionalidad usando transformada wavelet continua
    sp_data: DataFrame con fechas en el índice y columna 'Cierre'
    """
    print("Calculando patrón estacional promedio...")
    
    # Calcular patrón estacional promedio
    seasonal_pattern = calculate_seasonal_pattern(sp_data)
    
    # Rellenar días faltantes (para años bisiestos)
    full_year = np.arange(1, 252)
    seasonal_data = np.interp(full_year, seasonal_pattern.index, seasonal_pattern.values)
    
    print("Aplicando transformada wavelet continua...")
    
    # Aplicar transformada wavelet continua
    scales = np.arange(1, 23 )
    coefficients, frequencies, scales = continuous_wavelet_transform(
        seasonal_data, 
        scales=scales,
        wavelet='cmor1.5-1.0'
    )
    
    print("Generando gráfico...")
    
    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Crear meshgrid para el gráfico
    time_axis = np.arange(len(seasonal_data))
    T, S = np.meshgrid(time_axis, scales)
    
    # Plotear el escalograma con más niveles de color
    im = ax.contourf(T, S, np.abs(coefficients), levels=100, cmap='plasma')
    
    # Configurar el gráfico
    ax.set_xlabel('Mes', fontsize=12)
    ax.set_ylabel('Escala', fontsize=12)
    ax.set_title('Wavelet - Estacionalidad Promedio Anual', fontsize=14, fontweight='bold')
    
    # Configurar etiquetas del eje x (meses)
    month_labels = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                   'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    month_positions = np.linspace(0, len(seasonal_data)-1, 12)
    ax.set_xticks(month_positions)
    ax.set_xticklabels(month_labels, rotation=45, ha='right')
    
    # Invertir el eje y para que las escalas menores estén arriba
    ax.invert_yaxis()
    
    # Añadir colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnitud', fontsize=12)
    
    # Ajustar layout
    plt.tight_layout()
    
    print("¡Gráfico generado exitosamente!")
    
    return fig, coefficients, scales, seasonal_data

# Ejecutar el análisis
if __name__ == "__main__":
    # Verificar que sp_data existe
    try:
        # Ejecutar el análisis wavelet
        fig, coefficients, scales, seasonal_data = plot_wavelet_seasonality(sp_data)
        plt.show()
        
        # Información adicional
        print(f"\nInformación del análisis:")
        print(f"- Período analizado: {sp_data.index.min()} a {sp_data.index.max()}")
        print(f"- Total de observaciones: {len(sp_data)}")
        print(f"- Escalas analizadas: {scales.min()} a {scales.max()}")
        print(f"- Dimensiones del coeficiente wavelet: {coefficients.shape}")
        
    except NameError:
        print("Error: No se encontró la variable 'sp_data'")
        print("Asegúrate de que sp_data esté definida en tu entorno con:")
        print("- Índice: fechas (datetime)")
        print("- Columna: 'Cierre' con los precios de cierre")
        
    except Exception as e:
        print(f"Error durante el análisis: {e}")

# Para ejecutar directamente, simplemente copia y pega este código
# Asegúrate de que sp_data esté cargada en tu entorno

#%% MODELOS HÍBRIDOS

# === 1. Datos ===
df_selected = pd.read_csv("tu_dataset.csv")  # Cambia por tu dataset
series = df_selected['Cierre'].values

test_size = 0.2
split = int(len(series) * (1 - test_size))
train_data, test_data = series[:split], series[split:]

# === 2. ARIMA ===
best_score, best_order = float('inf'), None
for p, q in product(range(6), repeat=2):
    try:
        model = ARIMA(train_data, order=(p, 1, q))
        model_fit = model.fit()
        pred = model_fit.forecast(steps=len(test_data))
        rmse = np.sqrt(mean_squared_error(test_data, pred))
        if rmse < best_score:
            best_score, best_order = rmse, (p, 1, q)
    except:
        continue

print(f"Mejor ARIMA: {best_order}")
arima_model = ARIMA(series, order=best_order)
arima_fit = arima_model.fit()
arima_pred = arima_fit.forecast(steps=len(test_data))

# === 3. GARCH ===
returns = pd.Series(series).pct_change().dropna().values
split_garch = int(len(returns) * (1 - test_size))
train_ret, test_ret = returns[:split_garch], returns[split_garch:]

best_score, best_order = float('inf'), None
for p, q in product(range(6), repeat=2):
    if p == 0 and q == 0:
        continue
    try:
        model = arch_model(train_ret, vol='GARCH', p=p, q=q)
        model_fit = model.fit(disp='off')
        forecast = model_fit.forecast(horizon=len(test_ret))
        pred_var = forecast.variance.values[-1, :]
        rmse = np.sqrt(np.mean(pred_var))
        if rmse < best_score:
            best_score, best_order = rmse, (p, q)
    except:
        continue

print(f"Mejor GARCH: {best_order}")
garch_model = arch_model(returns, vol='GARCH', p=best_order[0], q=best_order[1])
garch_fit = garch_model.fit(disp='off')
forecast = garch_fit.forecast(horizon=len(test_ret))
garch_pred = forecast.variance.values[-1, :]

# === 4. Prepara datos para LSTM/GRU ===
scaler = MinMaxScaler()
arima_scaled = scaler.fit_transform(arima_pred.reshape(-1, 1))
garch_scaled = scaler.fit_transform(garch_pred.reshape(-1, 1))
target_scaled = scaler.fit_transform(test_data.reshape(-1, 1))

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 5
# ARIMA → LSTM / GRU
X_arima, y_arima = create_sequences(arima_scaled, window_size)
# GARCH → LSTM / GRU
X_garch, y_garch = create_sequences(garch_scaled, window_size)

# === 5. Definir Modelos Híbridos (LSTM y GRU) ===
def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru(input_shape):
    model = Sequential([
        GRU(50, return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# === 6. Entrenamiento y Evaluación ===
def train_and_evaluate(model, X, y, name):
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    print(f"{name} → MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return pred

# ARIMA → LSTM
print("\n=== ARIMA → LSTM ===")
arima_lstm = build_lstm((window_size, 1))
train_and_evaluate(arima_lstm, X_arima, y_arima, "ARIMA-LSTM")

# GARCH → LSTM
print("\n=== GARCH → LSTM ===")
garch_lstm = build_lstm((window_size, 1))
train_and_evaluate(garch_lstm, X_garch, y_garch, "GARCH-LSTM")

# ARIMA → GRU
print("\n=== ARIMA → GRU ===")
arima_gru = build_gru((window_size, 1))
train_and_evaluate(arima_gru, X_arima, y_arima, "ARIMA-GRU")

# GARCH → GRU
print("\n=== GARCH → GRU ===")
garch_gru = build_gru((window_size, 1))
train_and_evaluate(garch_gru, X_garch, y_garch, "GARCH-GRU")

# %% GRID SEARCH MODELOS TRANSFORMER
import numpy as np
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# --- CLASES Y FUNCIONES ---

class FinancialTimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, prediction_horizon=1):
        self.features = data[:, :-1]
        self.targets = data[:, -1]
        self.seq_len = sequence_length
        self.prediction_horizon = prediction_horizon

    def __len__(self):
        return int(len(self.features) - self.seq_len - self.prediction_horizon)

    def __getitem__(self, idx):
        X = self.features[idx:idx+self.seq_len]
        y = self.targets[idx+self.seq_len+self.prediction_horizon-1]
        return torch.FloatTensor(X), torch.FloatTensor([y])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # Positional encoding suma por secuencia
        return self.dropout(x)

class FinancialTransformer(nn.Module):
    def __init__(self, input_features, d_model, nhead, num_layers, dropout, sequence_length):
        super(FinancialTransformer, self).__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length

        # Capa de embedding para convertir las características de entrada a d_model dimensiones
        self.embedding = nn.Linear(input_features, d_model)

        # Codificador de posición
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Capa TransformerEncoderLayer con batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True  # Muy importante para usar (batch_size, seq_len, d_model)
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Capa final para producir la salida
        self.fc = nn.Linear(d_model, 1)  # O ajusta según el número de salidas

    def forward(self, x):
        # Esperado: x de forma (batch_size, sequence_length, input_features)
        if x.dim() == 4:
            # Por si entra como (batch_size, channels, sequence_length, features), colapsamos
            x = x.view(x.size(0), x.size(2), -1)

        x = self.embedding(x) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)  # Mantiene la forma
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        x = x[:, -1, :]  # Tomamos el último paso temporal
        x = self.fc(x)  # (batch_size, 1)
        return x


    def forward(self, x):
        # Esperado: x de forma (batch_size, sequence_length, input_features)
        if x.dim() == 4:
            # Por si entra como (batch_size, channels, sequence_length, features), colapsamos
            x = x.view(x.size(0), x.size(2), -1)

        x = self.embedding(x) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)  # Mantiene la forma
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        x = x[:, -1, :]  # Tomamos el último paso temporal
        x = self.fc(x)  # (batch_size, 1)
        return x



def prepare_data(df_selected, selected_features, sequence_length,
                 prediction_horizon=10, train_ratio=0.8):
    data = df_selected[selected_features + ['Cierre']].values
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(data[:, :-1])
    target_scaler.fit(data[:, -1].reshape(-1, 1))
    data_normalized = np.column_stack([
        feature_scaler.transform(data[:, :-1]),
        target_scaler.transform(data[:, -1].reshape(-1, 1))
    ])
    train_size = int(len(data_normalized) * train_ratio)
    train_data = data_normalized[:train_size]
    test_data = data_normalized[train_size:]
    return train_data, test_data, feature_scaler, target_scaler

def evaluate_model(model, data_loader, target_scaler):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(batch_y.cpu().numpy().flatten())

    predictions = target_scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)).flatten()
    actuals = target_scaler.inverse_transform(np.array(
        actuals).reshape(-1, 1)).flatten()

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    r2 = r2_score(actuals, predictions)

    # Directional Accuracy
    direction_pred = np.sign(np.diff(predictions))
    direction_real = np.sign(np.diff(actuals))
    directional_accuracy = np.mean(direction_pred == direction_real)

    # Sharpe Ratio
    returns = np.diff(predictions) / predictions[:-1]
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)

    return predictions, actuals, {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy,
        'Sharpe_Ratio': sharpe_ratio
    }


def train_model(model, train_loader, val_loader, epochs=500,
                learning_rate=0.001):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, 
                                                     factor=0.5)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20

    epoch_iter = tqdm(range(epochs), desc="Épocas", unit="época", leave=False)
    for epoch in epoch_iter:
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        epoch_iter.set_postfix({
            "Train Loss": f"{train_loss:.5f}",
            "Val Loss": f"{val_loss:.5f}",
            "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_transformer_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(torch.load('best_transformer_model.pth'))
    return train_losses[-1], val_losses[-1], optimizer.param_groups[0]['lr']

# --- PARÁMETROS DEL GRID SEARCH ---

param_grid = {
    'd_model': [32, 64],
    'nhead': [2, 4],
    'num_layers': [2,3, 4],
    'dropout': [0.2],
    'learning_rate': [ 0.0005, 0.00001],
    'batch_size': [64],
    'sequence_length': [60, 120]
}

# --- EJECUCIÓN PRINCIPAL CON GRID SEARCH ---

# Obtener automáticamente las columnas y excluir el target
columna_objetivo = 'Cierre'
selected_features = [col for col in list(df_selected.columns
                                         ) if col != columna_objetivo]
print("Características seleccionadas:", selected_features)

# Lista para guardar resultados de todos los modelos
resultados_grid = []

# Generar todas las combinaciones de hiperparámetros
keys, values = zip(*param_grid.items())
combinaciones = list(itertools.product(*values))

# Barra de progreso para el grid search
grid_bar = tqdm(combinaciones, desc="Grid Search", unit="modelo")

for comb in grid_bar:
    params = dict(zip(keys, comb))
    grid_bar.set_postfix(**{k: str(v) for k, v in params.items()})
    
    # INICIO: medir tiempo
    start_time = time.time()
    
    # Preparar datos con la ventana temporal actual
    train_data, test_data, feature_scaler, target_scaler = prepare_data(
        df_selected, selected_features, params['sequence_length']
    )
    
    # Crear datasets y dataloaders
    train_dataset = FinancialTimeSeriesDataset(train_data, params[
        'sequence_length'])
    test_dataset = FinancialTimeSeriesDataset(test_data, params[
        'sequence_length'])
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=params[
        'batch_size'], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=params[
        'batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params[
        'batch_size'], shuffle=False)
    
    # Inicializar modelo
    input_features = len(selected_features)
    model = FinancialTransformer(
        input_features=input_features,
        d_model=params['d_model'],
        nhead=params['nhead'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        sequence_length=params['sequence_length']
    ).to(device)
    
    # Entrenar modelo
    train_loss, val_loss, final_lr = train_model(
        model, train_loader, val_loader,
        epochs=500, learning_rate=params['learning_rate']
    )
    
    # Evaluar modelo
    predictions, actuals, metrics = evaluate_model(model, test_loader, 
                                                   target_scaler)
    
    # FINAL: medir tiempo
    elapsed_time = time.time() - start_time
    
    # Guardar resultados
    resultados_grid.append({
    **params,
    'train_loss': train_loss,
    'val_loss': val_loss,
    **metrics,
    'final_learning_rate': final_lr,
    'execution_time_sec': elapsed_time  # <-- Tiempo en segundos
    })

# Convertir resultados a DataFrame y guardar en CSV
resultados_df = pd.DataFrame(resultados_grid)
resultados_df.to_csv('resultados_grid_search2.csv', index=False, 
                     encoding='utf-8-sig')

selected_features = [col for col in list(df_selected.columns
                                         ) if col != columna_objetivo]
print("Características seleccionadas:", selected_features)

# === IDENTIFICAR MEJOR MODELO (por menor RMSE) ===
mejor_fila = resultados_df.loc[resultados_df['MAPE'].idxmin()]
mejor_params = {k: mejor_fila[k] for k in param_grid.keys()}

# === PREPARAR DATOS DEL MEJOR MODELO ===
train_data, test_data, feature_scaler, target_scaler = prepare_data(
    df_selected, selected_features, mejor_params['sequence_length']
)

train_dataset = FinancialTimeSeriesDataset(train_data, 
                                           int(mejor_params['sequence_length']))
test_dataset = FinancialTimeSeriesDataset(test_data, 
                                          int(mejor_params['sequence_length']))


train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [
    train_size, val_size])

batch_size = int(mejor_params['batch_size'])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# === REENTRENAR MEJOR MODELO PARA OBTENER CURVAS DE ENTRENAMIENTO ===
model = FinancialTransformer(
    input_features=int(len(selected_features)),
    d_model=int(mejor_params['d_model']),
    nhead=int(mejor_params['nhead']),
    num_layers=int(mejor_params['num_layers']),
    dropout=float(mejor_params['dropout']),  # dropout sí puede ser float (entre 0 y 1)
    sequence_length=int(mejor_params['sequence_length'])
).to(device)


# Entrenar modelo y guardar pérdidas por época
def train_model_with_curves(model, train_loader, val_loader, epochs, 
                            learning_rate):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, 
                                                     factor=0.5)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20

    for epoch in tqdm(range(epochs), desc="Entrenando mejor modelo"):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_losses.append(total_train_loss / len(train_loader))

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
        val_losses.append(total_val_loss / len(val_loader))
        scheduler.step(val_losses[-1])

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_retrained.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(torch.load('best_model_retrained.pth'))
    return train_losses, val_losses

# === Ejecutar entrenamiento final ===
train_losses, val_losses = train_model_with_curves(
    model, train_loader, val_loader, epochs=500, learning_rate=20**-4)

# === GRAFICO 1: Curvas de pérdida ===
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Curvas de pérdida del mejor modelo')
plt.xlabel('Época')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === GRAFICO 2: Predicciones vs valores reales ===
predictions, actuals, _ = evaluate_model(model, test_loader, target_scaler)

plt.figure(figsize=(10, 5))
plt.plot(actuals, label='Real', alpha=0.7)
plt.plot(predictions, label='Predicción', alpha=0.7)
plt.title('Predicciones vs Valores reales (Mejor modelo)')
plt.xlabel('Índice de tiempo')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === GRAFICO 3: Error absoluto ===
plt.figure(figsize=(10, 5))
abs_error = np.abs(actuals - predictions)
plt.plot(abs_error, label='Error Absoluto', color='orange')
plt.title('Error Absoluto a lo Largo del Tiempo')
plt.xlabel('Índice Temporal')
plt.ylabel('Error |Real - Predicción|')
plt.grid(True, alpha=0.3)
plt.legend()

# %% BAYESIAN SEARCH MODELOS TRANSFORMER

#### DA significa DIRECTIONAL ACCURACY

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import optuna

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# --- CLASES Y FUNCIONES ---

class FinancialTimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, prediction_horizon=1):
        self.features = data[:, :-1]
        self.targets = data[:, -1]
        self.seq_len = sequence_length
        self.prediction_horizon = prediction_horizon

    def __len__(self):
        return int(len(self.features) - self.seq_len - self.prediction_horizon)

    def __getitem__(self, idx):
        X = self.features[idx:idx+self.seq_len]
        y = self.targets[idx+self.seq_len+self.prediction_horizon-1]
        return torch.FloatTensor(X), torch.FloatTensor([y])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calcular div_term solo para las posiciones pares disponibles
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
            ) * (-np.log(10000.0) / d_model))
        
        # Aplicar seno a posiciones pares (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Aplicar coseno a posiciones impares (1, 3, 5, ...)
        # Solo si hay suficientes posiciones impares
        if d_model > 1:
            # Para d_model impar, pe[:, 1::2] tiene un elemento menos que div_term
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class FinancialTransformer(nn.Module):
    def __init__(self, input_features, d_model, nhead, num_layers, dropout, 
                 sequence_length):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead,
                                                    dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 
                                                         num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sequence_length = sequence_length

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[-1, :, :]
        x = self.fc(x)
        return x

def prepare_data(df_selected, selected_features, sequence_length,
                 prediction_horizon=1, train_ratio=0.8):
    data = df_selected[selected_features + ['Cierre']].values
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(data[:, :-1])
    target_scaler.fit(data[:, -1].reshape(-1, 1))
    data_normalized = np.column_stack([
        feature_scaler.transform(data[:, :-1]),
        target_scaler.transform(data[:, -1].reshape(-1, 1))
    ])
    train_size = int(len(data_normalized) * train_ratio)
    train_data = data_normalized[:train_size]
    test_data = data_normalized[train_size:]
    return train_data, test_data, feature_scaler, target_scaler

def directional_accuracy(actuals, predictions):
    if len(actuals) < 2:
        return 0.0
    actual_directions = np.sign(np.diff(actuals))
    pred_directions = np.sign(np.diff(predictions))
    return np.mean(actual_directions == pred_directions)

def evaluate_model(model, data_loader, target_scaler):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(batch_y.cpu().numpy().flatten())
    predictions = target_scaler.inverse_transform(np.array(
        predictions).reshape(-1, 1)).flatten()
    actuals = target_scaler.inverse_transform(np.array(
        actuals).reshape(-1, 1)).flatten()
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    da = directional_accuracy(actuals, predictions)
    return predictions, actuals, {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'DA': da
    }

def train_model(model, train_loader, val_loader, epochs=500,
                learning_rate=0.001, params=None):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, 
                                                     factor=0.5)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20

    epoch_iter = tqdm(range(epochs), desc="Épocas", unit="época", leave=False)
    for epoch in epoch_iter:
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # Evaluar DA en validación
        predictions, actuals, metrics = evaluate_model(model, val_loader, 
                                                       target_scaler)
        da = metrics['DA']

        # Mostrar parámetros y métricas
        postfix = {
            "Train Loss": f"{train_loss:.5f}",
            "Val Loss": f"{val_loss:.5f}",
            "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
            "DA": f"{da:.4f}"
        }
        if params is not None:
            for k, v in params.items():
                postfix[k] = str(v)
        epoch_iter.set_postfix(postfix)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_transformer_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(torch.load('best_transformer_model.pth'))
    return train_losses[-1], val_losses[-1], optimizer.param_groups[0
                                                                    ]['lr'], da

# --- PARÁMETROS DEL MEJOR MODELO ---

best_params = {
    'd_model': 64,
    'nhead': 2,
    'num_layers': 1,
    'dropout': 0.2,
    'learning_rate': 0.0005,
    'batch_size': 32,
    'sequence_length': 60
}

# Obtener automáticamente las columnas y excluir el target
columna_objetivo = 'Cierre'
selected_features = [col for col in list(df_selected.columns
                                         ) if col != columna_objetivo]
print("Características seleccionadas:", selected_features)

def get_param_ranges(best, factor=0.5):
    ranges = {}
    for k, v in best.items():
        if isinstance(v, float):
            min_val = v * (1 - factor)
            max_val = v * (1 + factor)
            if k == 'dropout':
                min_val = max(0.1, min_val)
            if min_val > max_val:
                min_val, max_val = max_val, min_val
        else:
            min_val = max(1, int(v * (1 - factor)))
            max_val = int(v * (1 + factor))
        ranges[k] = (min_val, max_val)
    return ranges

param_ranges = get_param_ranges(best_params)

def objective(trial):
    params = {}
    for k, (min_val, max_val) in param_ranges.items():
        if k in ['d_model', 'nhead', 'num_layers', 'batch_size', 
                 'sequence_length']:
            params[k] = trial.suggest_int(k, min_val, max_val)
        elif k in ['dropout', 'learning_rate']:
            params[k] = trial.suggest_float(k, min_val, max_val)
    
    # Asegurar que d_model sea divisible por nhead
    params['d_model'] = max(1, (params['d_model'] // params[
        'nhead']) * params['nhead'])
    
    train_data, test_data, feature_scaler, target_scaler = prepare_data(
        df_selected, selected_features, params['sequence_length']
    )
    train_dataset = FinancialTimeSeriesDataset(train_data,
                                               params['sequence_length'])
    test_dataset = FinancialTimeSeriesDataset(test_data, 
                                              params['sequence_length'])
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=params[
        'batch_size'], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=params[
        'batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params[
        'batch_size'], shuffle=False)

    input_features = len(selected_features)
    model = FinancialTransformer(
        input_features=input_features,
        d_model=params['d_model'],
        nhead=params['nhead'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        sequence_length=params['sequence_length']
    ).to(device)

    train_loss, val_loss, final_lr, da = train_model(
        model, train_loader, val_loader,
        epochs=500, learning_rate=params['learning_rate'],
        params=params
    )

    predictions, actuals, metrics = evaluate_model(model, test_loader,
                                                   target_scaler)

    trial.set_user_attr('train_loss', train_loss)
    trial.set_user_attr('val_loss', val_loss)
    trial.set_user_attr('final_learning_rate', final_lr)
    trial.set_user_attr('MAE', metrics['MAE'])
    trial.set_user_attr('RMSE', metrics['RMSE'])
    trial.set_user_attr('MAPE', metrics['MAPE'])
    trial.set_user_attr('R2', metrics['R2'])
    trial.set_user_attr('DA', metrics['DA'])

    return metrics['RMSE']  # Maximizar RMSE

# --- EJECUCIÓN PRINCIPAL ---

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

# Guardar resultados en CSV
resultados = []
for trial in study.trials:
    params = trial.params
    metrics = {
        'train_loss': trial.user_attrs['train_loss'],
        'val_loss': trial.user_attrs['val_loss'],
        'final_learning_rate': trial.user_attrs['final_learning_rate'],
        'MAE': trial.user_attrs['MAE'],
        'RMSE': trial.user_attrs['RMSE'],
        'MAPE': trial.user_attrs['MAPE'],
        'R2': trial.user_attrs['R2'],
        'DA': trial.user_attrs['DA']
    }
    resultados.append({**params, **metrics})
resultados_df = pd.DataFrame(resultados)
resultados_df.to_csv('resultados_bayesian_search.csv', index=False, 
                     encoding='utf-8-sig')

# --- VISUALIZACIÓN DEL MEJOR MODELO ---

best_trial = study.best_trial
best_params = best_trial.params
print("\n🎯 MEJOR MODELO BAYESIANO:")
print("="*50)
print(best_params)
print("R²:", best_trial.user_attrs['R2'])
print("RMSE:", best_trial.user_attrs['RMSE'])
print("MAPE:", best_trial.user_attrs['MAPE'])
print("DA:", best_trial.user_attrs['DA'])

# Volver a entrenar/evaluar el mejor para obtener predicciones y gráficos
train_data, test_data, feature_scaler, target_scaler = prepare_data(
    df_selected, selected_features, best_params['sequence_length']
)
train_dataset = FinancialTimeSeriesDataset(train_data, best_params[
    'sequence_length'])
test_dataset = FinancialTimeSeriesDataset(test_data, best_params[
    'sequence_length'])
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size])
train_loader = DataLoader(train_subset, batch_size=best_params[
    'batch_size'], shuffle=True)
val_loader = DataLoader(val_subset, batch_size=best_params[
    'batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=best_params
                         ['batch_size'], shuffle=False)
model = FinancialTransformer(
    input_features=input_features,
    d_model=best_params['d_model'],
    nhead=best_params['nhead'],
    num_layers=best_params['num_layers'],
    dropout=best_params['dropout'],
    sequence_length=best_params['sequence_length']
).to(device)
train_loss, val_loss, final_lr, da = train_model(
    model, train_loader, val_loader,
    epochs=500, learning_rate=best_params['learning_rate'],
    params=best_params
)
predictions, actuals, metrics = evaluate_model(model, test_loader, 
                                               target_scaler)

# Visualización de resultados
plt.figure(figsize=(15, 10))

# Subplot 1: Train/Val Loss
plt.subplot(2, 2, 1)
plt.plot([train_loss], 'o-', label='Train Loss', alpha=0.7)
plt.plot([val_loss], 'o-', label='Validation Loss', alpha=0.7)
plt.title('Pérdidas del mejor modelo')
plt.xlabel('Época (solo resultado final)')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Error Absoluto
plt.subplot(2, 2, 2)
abs_error = np.abs(actuals - predictions)
plt.plot(abs_error, label='Error Absoluto', color='orange')
plt.title('Error Absoluto a lo Largo del Tiempo')
plt.xlabel('Índice Temporal')
plt.ylabel('Error |Real - Predicción|')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 3 y 4 combinados: Serie temporal de valores reales vs predichos
plt.subplot(2, 2, (3, 4))
indices = range(len(predictions))
plt.plot(indices, actuals, label='Real', alpha=0.8)
plt.plot(indices, predictions, label='Predicción', alpha=0.8)
plt.title('Predicciones vs Valores Reales - Índice S&P 500')
plt.xlabel('Tiempo')
plt.ylabel('Precio de Cierre')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Resumen final del mejor modelo
print("\n🎯 RESUMEN FINAL DEL MEJOR MODELO:")
print("="*50)
print(f"Características utilizadas: {len(selected_features)}")
print(f"Ventana temporal: {best_params['sequence_length']} períodos")
print(f"Mejor R²: {metrics['R2']:.4f}")
print(f"RMSE: {metrics['RMSE']:.4f}")
print(f"MAPE: {metrics['MAPE']:.2f}%")
print(f"DA: {metrics['DA']:.4f}")

#%% GRID Y BAYESIAN SEARCH MODELOS LIQUID NEURAL NETWORKS

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ncps.torch import CfC
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

# ===============================
# 1. Preprocesado de datos
# ===============================

# df_selected debe tener índice datetime y columna 'Cierre'
df_selected_numeric = df_selected.select_dtypes(include=[np.number])

train_df = df_selected_numeric[df_selected_numeric.index < '2023-01-01'].copy()
test_df = df_selected_numeric[df_selected_numeric.index >= '2023-01-01'].copy()

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

target_idx = train_df.columns.get_loc('Cierre')

def create_sequences(data, target_idx, lookback=20):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback, target_idx])
    return np.array(X), np.array(y)

lookback = 20

X_train, y_train = create_sequences(train_scaled, target_idx, lookback)
X_test, y_test = create_sequences(test_scaled, target_idx, lookback)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ===============================
# 2. Modelo CfC apilado
# ===============================

class CfCTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(CfCTimeSeriesModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(CfC(input_size=layer_input_dim, units=hidden_dim))
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        for layer in self.layers:
            output, _ = layer(x, h)
            x = output  # salida de una capa es entrada para la siguiente
        out = self.fc(output[:, -1, :])
        return out

# ===============================
# 3. Early stopping helper
# ===============================

class EarlyStopping:
    def __init__(self, patience=20, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf  # Cambiado aquí
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop

# ===============================
# 4. Funciones auxiliares
# ===============================

def invert_scale(scaled_preds, scaler, target_idx, original_shape):
    zeros = np.zeros((scaled_preds.shape[0], original_shape[1]))
    zeros[:, target_idx] = scaled_preds.flatten()
    inversed = scaler.inverse_transform(zeros)[:, target_idx]
    return inversed

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ===============================
# 5. Grid search con tqdm y early stopping (MODIFICADO)
# ===============================

param_grid = {
    "hidden_dim": [16, 32, 64, 128],
    "learning_rate": [0.001, 0.0001, 0.0005],
    "batch_size": [16, 32, 64],
    "num_layers": [1, 2, 3],
    "epochs": [500]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_mape = np.inf  # Cambiado aquí
best_params = None
best_model = None

combinations = list(itertools.product(
    param_grid['hidden_dim'],
    param_grid['learning_rate'],
    param_grid['batch_size'],
    param_grid['num_layers'],
    param_grid['epochs']
))

outer_pbar = tqdm(combinations, desc="Grid Search", position=0)

for hidden_dim, lr, batch_size, num_layers, epochs in outer_pbar:
    # Configurar descripción del modelo actual
    model_desc = f"h_dim={hidden_dim}, lr={lr}, bs={batch_size}, layers={num_layers}"
    outer_pbar.set_description(f"Probando: {model_desc}")
    
    # Preparar datos
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Modelo y optimizador
    model = CfCTimeSeriesModel(X_train.shape[2], hidden_dim, 1, num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    early_stopping = EarlyStopping(patience=20, verbose=False)

    # Barra de progreso para épocas
    epoch_pbar = tqdm(range(epochs), desc="Épocas", leave=False, position=1)
    
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        epoch_pbar.set_postfix({"Loss": f"{epoch_loss:.8f}"})

        # Validación
        model.eval()
        with torch.no_grad():
            preds_val = model(X_test_tensor.to(device)).cpu().numpy()
            y_val = y_test_tensor.cpu().numpy()

        preds_val_rescaled = invert_scale(preds_val, scaler, target_idx, test_scaled.shape)
        y_val_rescaled = invert_scale(y_val, scaler, target_idx, test_scaled.shape)

        val_mape = mean_absolute_percentage_error(y_val_rescaled, preds_val_rescaled)
        
        if early_stopping(val_mape):
            epoch_pbar.close()
            break

    # Cerrar la barra de épocas
    epoch_pbar.close()

    # Actualizar mejores parámetros
    if val_mape < best_mape:
        best_mape = val_mape
        best_params = {
            "hidden_dim": hidden_dim,
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_layers": num_layers,
            "epochs_trained": epoch + 1
        }
        best_model = model

    outer_pbar.set_postfix({
        "Best MAPE": f"{best_mape:.3f}%",
        "Current MAPE": f"{val_mape:.3f}%"
    })

print("\nMejor modelo encontrado:")
print(best_params)

# ===============================
# 6. Visualizar resultados del mejor modelo
# ===============================

best_model.eval()
with torch.no_grad():
    preds = best_model(X_test_tensor.to(device)).cpu().numpy()
    y_true = y_test_tensor.cpu().numpy()

preds_rescaled = invert_scale(preds, scaler, target_idx, test_scaled.shape)
y_true_rescaled = invert_scale(y_true, scaler, target_idx, test_scaled.shape)

plt.figure(figsize=(12,6))
plt.plot(test_df.index[lookback:], y_true_rescaled, label='Real')
plt.plot(test_df.index[lookback:], preds_rescaled, label='Predicción')
plt.title('Predicciones vs Valores Reales (Mejor Modelo)')
plt.xlabel('Fecha')
plt.ylabel('Cierre')
plt.legend()
plt.show()

# Entrenamiento de mejores modelos 

print(f"\n=== MEJORES PARÁMETROS ===")
for param, valor in mejor_params.items():
    print(f"{param}: {valor}")

print(f"\nMétricas del mejor modelo:")
print(f"RMSE: {mejor_fila['RMSE']:.4f}")
print(f"MAE: {mejor_fila['MAE']:.4f}")
print(f"R²: {mejor_fila['R2']:.4f}")
print(f"Precisión Direccional: {mejor_fila['Directional_Accuracy']:.4f}")
print(f"Tiempo de ejecución: {mejor_fila['execution_time_sec']:.2f} segundos")
print(f"Parámetros del modelo: {mejor_fila['model_complexity']:,}")

# Obtener las dos mejores configuraciones según el MAPE
top_configs = resultados_df.nsmallest(2, 'MAPE')

model_metrics = {}

for idx, config in top_configs.iterrows():
    config_name = f"Config_{idx}"
    print(f"\nEntrenando configuración: {config_name}")

    # Limpiar tipos
    config_params = {
        'sequence_length': int(config['sequence_length']),
        'batch_size': int(config['batch_size']),
        'd_model': int(config['d_model']),
        'nhead': int(config['nhead']),
        'num_layers': int(config['num_layers']),
        'dim_feedforward': int(config['dim_feedforward']),
        'dropout': float(config['dropout']),
        'learning_rate': float(config['learning_rate']),
        'weight_decay': float(config['weight_decay']),
        'warmup_steps': int(config['warmup_steps'])
    }

    # Preparar datos
    train_data, test_data, feature_scaler, target_scaler = prepare_data(
        df_selected, selected_features, config_params['sequence_length']
    )

    train_dataset = FinancialTimeSeriesDataset(train_data, config_params['sequence_length'])
    test_dataset = FinancialTimeSeriesDataset(test_data, config_params['sequence_length'])

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=config_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config_params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config_params['batch_size'], shuffle=False)

    # Crear modelo
    model = FinancialTransformer(
        input_features=len(selected_features),
        d_model=config_params['d_model'],
        nhead=config_params['nhead'],
        num_layers=config_params['num_layers'],
        dim_feedforward=config_params['dim_feedforward'],
        dropout=config_params['dropout'],
        max_len=config_params['sequence_length'] + 50
    ).to(device)

    # Entrenar
    train_losses, val_losses = train_model_with_curves(
        model, train_loader, val_loader, epochs=300,
        learning_rate=config_params['learning_rate'],
        weight_decay=config_params['weight_decay'],
        warmup_steps=config_params['warmup_steps']
    )

    # Evaluar
    predictions, actuals, final_metrics = evaluate_model(model, test_loader, target_scaler)

    # Guardar
    model_metrics[config_name] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'predictions': predictions,
        'actuals': actuals,
        'metrics': final_metrics,
        'params': config_params
    }

# Gráficos de los mejores modelos
for config_name in model_metrics.keys():
    predictions = model_metrics[config_name]['predictions']
    actuals = model_metrics[config_name]['actuals']
    r2 = model_metrics[config_name]['metrics']['R2']
    mae = model_metrics[config_name]['metrics']['MAE']
    rmse = model_metrics[config_name]['metrics']['RMSE']
    mape = model_metrics[config_name]['metrics']['MAPE']
    directional_acc = model_metrics[config_name]['metrics']['Directional_Accuracy']
    params = model_metrics[config_name]['params']

    print(f"\n=== {config_name} ===")
    print(f"d_model: {params['d_model']}, nhead: {params['nhead']}, layers: {params['num_layers']}")

    # 1. GRÁFICO PRINCIPAL DE PREDICCIONES - PROFESIONAL
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Serie temporal completa
    time_idx = range(len(predictions))
    ax1.plot(time_idx, actuals, label='Valores Reales', color='#2E86AB', linewidth=2, alpha=0.8)
    ax1.plot(time_idx, predictions, label='Predicciones Transformer', color='#A23B72', linewidth=2, alpha=0.8)
    ax1.fill_between(time_idx, actuals, predictions, alpha=0.2, color='gray', label='Error')
    
    ax1.set_title(f'{config_name} - Predicciones Transformer vs Valores Reales\n'
                  f'R² = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f} | MAPE = {mape:.2f}%', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Índice Temporal', fontsize=12)
    ax1.set_ylabel('Precio de Cierre', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Zoom en últimos 100 puntos
    zoom_size = min(100, len(predictions))
    zoom_idx = range(len(predictions) - zoom_size, len(predictions))
    ax2.plot(zoom_idx, actuals[-zoom_size:], label='Valores Reales', color='#2E86AB', linewidth=2.5, marker='o', markersize=3)
    ax2.plot(zoom_idx, predictions[-zoom_size:], label='Predicciones', color='#A23B72', linewidth=2.5, marker='s', markersize=3)
    
    ax2.set_title(f'Zoom - Últimos {zoom_size} Puntos | Precisión Direccional = {directional_acc:.4f}', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Índice Temporal', fontsize=12)
    ax2.set_ylabel('Precio de Cierre', fontsize=12)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

    # 2. Error absoluto con estadísticas
    abs_error = np.abs(actuals - predictions)
    rel_error = abs_error / np.abs(actuals) * 100
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(abs_error, color='#F18F01', alpha=0.8, linewidth=1.5)
    plt.axhline(y=np.mean(abs_error), color='red', linestyle='--', alpha=0.8, 
                label=f'Error Promedio: {np.mean(abs_error):.4f}')
    plt.fill_between(range(len(abs_error)), 0, abs_error, alpha=0.3, color='#F18F01')
    plt.title(f'{config_name} - Error Absoluto\nMAE: {mae:.4f} | Std: {np.std(abs_error):.4f}', fontweight='bold')
    plt.xlabel('Índice Temporal')
    plt.ylabel('|Real - Predicción|')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Error relativo (%)
    plt.subplot(1, 2, 2)
    plt.plot(rel_error, color='#C73E1D', alpha=0.8, linewidth=1.5)
    plt.axhline(y=np.mean(rel_error), color='darkred', linestyle='--', alpha=0.8,
                label=f'Error Relativo Promedio: {np.mean(rel_error):.2f}%')
    plt.fill_between(range(len(rel_error)), 0, rel_error, alpha=0.3, color='#C73E1D')
    plt.title(f'{config_name} - Error Relativo (%)\nMAPE: {mape:.2f}%', fontweight='bold')
    plt.xlabel('Índice Temporal')
    plt.ylabel('Error Relativo (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

    # 3. Scatter plot mejorado
    plt.figure(figsize=(10, 8))
    
    # Colorear puntos por densidad
    from scipy.stats import pearsonr
    correlation, p_value = pearsonr(actuals, predictions)
    
    plt.scatter(actuals, predictions, alpha=0.6, s=15, c=range(len(actuals)), 
                cmap='viridis', edgecolors='white', linewidth=0.5)
    
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, alpha=0.8, label='Línea Perfecta')
    
    # Línea de regresión
    z = np.polyfit(actuals, predictions, 1)
    p = np.poly1d(z)
    plt.plot(actuals, p(actuals), "b-", alpha=0.8, linewidth=2, label=f'Regresión (y = {z[0]:.3f}x + {z[1]:.3f})')
    
    plt.title(f'{config_name} - Scatter Plot\nR² = {r2:.4f} | Correlación = {correlation:.4f} (p = {p_value:.4f})', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Valores Reales', fontsize=12)
    plt.ylabel('Predicciones', fontsize=12)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.colorbar(label='Orden Temporal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    # 4. Curvas de pérdida con suavizado
    plt.figure(figsize=(12, 6))
    
    train_losses_smooth = pd.Series(model_metrics[config_name]['train_losses']).rolling(window=5, center=True).mean()
    val_losses_smooth = pd.Series(model_metrics[config_name]['val_losses']).rolling(window=5, center=True).mean()
    
    epochs = range(len(model_metrics[config_name]['train_losses']))
    
    plt.plot(epochs, model_metrics[config_name]['train_losses'], alpha=0.3, color='blue', linewidth=1)
    plt.plot(epochs, train_losses_smooth, label='Train Loss (suavizado)', color='blue', linewidth=2)
    
    plt.plot(epochs, model_metrics[config_name]['val_losses'], alpha=0.3, color='red', linewidth=1)
    plt.plot(epochs, val_losses_smooth, label='Validation Loss (suavizado)', color='red', linewidth=2)
    
    # Marcar mejor época
    best_epoch = np.argmin(model_metrics[config_name]['val_losses'])
    best_val_loss = min(model_metrics[config_name]['val_losses'])
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, 
                label=f'Mejor época: {best_epoch} (Val Loss: {best_val_loss:.5f})')
    
    plt.title(f'{config_name} - Curvas de Entrenamiento', fontsize=14, fontweight='bold')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida (MSE)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Escala logarítmica para mejor visualización
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    # 5. NUEVO: Análisis de residuos
    residuals = actuals - predictions
    plt.figure(figsize=(15, 5))
    
    # Histograma de residuos
    plt.subplot(1, 3, 1)
    plt.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=np.mean(residuals), color='red', linestyle='--', 
                label=f'Media: {np.mean(residuals):.4f}')
    plt.axvline(x=np.median(residuals), color='green', linestyle='--', 
                label=f'Mediana: {np.median(residuals):.4f}')
    plt.title('Distribución de Residuos', fontweight='bold')
    plt.xlabel('Residuos (Real - Predicción)')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot para normalidad
    from scipy import stats
    plt.subplot(1, 3, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normalidad)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Residuos vs predicciones
    plt.subplot(1, 3, 3)
    plt.scatter(predictions, residuals, alpha=0.6, s=10, color='purple')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    plt.title('Residuos vs Predicciones', fontweight='bold')
    plt.xlabel('Predicciones')
    plt.ylabel('Residuos')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("\n=== RESUMEN FINAL ===")
for config_name in model_metrics.keys():
    metrics = model_metrics[config_name]['metrics']
    print(f"\n{config_name}:")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  R²: {metrics['R2']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  Precisión Direccional: {metrics['Directional_Accuracy']:.4f}")

# Extraer hiperparámetros
best_params = mejor_config.to_dict()
sequence_length = int(best_params['sequence_length'])
batch_size = int(best_params['batch_size'])

# Preparar datos nuevamente
train_data, test_data, feature_scaler, target_scaler = prepare_data(
    df_selected, selected_features, sequence_length
)

# Dataset y DataLoader para test
test_dataset = FinancialTimeSeriesDataset(test_data, sequence_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Cargar el modelo y pesos entrenados
input_features = len(selected_features)
best_model = FinancialTransformer(
    input_features=input_features,
    d_model=int(best_params['d_model']),
    nhead=int(best_params['nhead']),
    num_layers=int(best_params['num_layers']),
    dim_feedforward=int(best_params['dim_feedforward']),
    dropout=best_params['dropout'],
    max_len=sequence_length + 50
).to(device)

# Cargar los pesos entrenados
best_model.load_state_dict(torch.load('best_transformer_model.pth'))

# Evaluar el modelo
predictions, actuals, metrics = evaluate_model(best_model, test_loader, target_scaler)

# Crear índice temporal para las predicciones
start_idx = sequence_length + int(best_params['prediction_horizon']) - 1
test_index = df_selected.iloc[len(train_data) + start_idx : len(train_data) + start_idx + len(actuals)].index

# Graficar predicciones vs valores reales
plt.figure(figsize=(14,6))
plt.plot(test_index, actuals, label='Real', color='blue')
plt.plot(test_index, predictions, label='Predicción', color='orange')
plt.title('Predicción vs Real (Mejor Transformer)')
plt.xlabel('Fecha')
plt.ylabel('Precio de cierre')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# === ANÁLISIS DE RESULTADOS ===

# Top 5 modelos por RMSE
top_5_rmse = resultados_df.nsmallest(5, 'RMSE')
print("\n=== TOP 5 MODELOS POR RMSE ===")
print(top_5_rmse[['d_model', 'nhead', 'num_layers', 'dim_feedforward', 
                  'RMSE', 'MAE', 'R2', 'Directional_Accuracy']].to_string())

# Top 5 modelos por R²
top_5_r2 = resultados_df.nlargest(5, 'R2')
print("\n=== TOP 5 MODELOS POR R² ===")
print(top_5_r2[['d_model', 'nhead', 'num_layers', 'dim_feedforward', 
               'RMSE', 'MAE', 'R2', 'Directional_Accuracy']].to_string())

# === IDENTIFICAR MEJOR MODELO (por menor MAPE) ===
mejor_fila = resultados_df.loc[resultados_df['MAPE'].idxmin()]
mejor_params = {k: mejor_fila[k] for k in param_grid.keys()}

# Entrenamiento de mejores modelos 

print(f"\n=== MEJORES PARÁMETROS ===")
for param, valor in mejor_params.items():
    print(f"{param}: {valor}")

print(f"\nMétricas del mejor modelo:")
print(f"RMSE: {mejor_fila['RMSE']:.4f}")
print(f"MAE: {mejor_fila['MAE']:.4f}")
print(f"R²: {mejor_fila['R2']:.4f}")
print(f"Precisión Direccional: {mejor_fila['Directional_Accuracy']:.4f}")
print(f"Tiempo de ejecución: {mejor_fila['execution_time_sec']:.2f} segundos")
print(f"Parámetros del modelo: {mejor_fila['model_complexity']:,}")

# Obtener las dos mejores configuraciones según el MAPE
top_configs = resultados_df.nsmallest(2, 'MAPE')

model_metrics = {}

for idx, config in top_configs.iterrows():
    config_name = f"Config_{idx}"
    print(f"\nEntrenando configuración: {config_name}")

    # Limpiar tipos
    config_params = {
        'sequence_length': int(config['sequence_length']),
        'batch_size': int(config['batch_size']),
        'd_model': int(config['d_model']),
        'nhead': int(config['nhead']),
        'num_layers': int(config['num_layers']),
        'dim_feedforward': int(config['dim_feedforward']),
        'dropout': float(config['dropout']),
        'learning_rate': float(config['learning_rate']),
        'weight_decay': float(config['weight_decay']),
        'warmup_steps': int(config['warmup_steps'])
    }

    # Preparar datos
    train_data, test_data, feature_scaler, target_scaler = prepare_data(
        df_selected, selected_features, config_params['sequence_length']
    )

    train_dataset = FinancialTimeSeriesDataset(train_data, config_params['sequence_length'])
    test_dataset = FinancialTimeSeriesDataset(test_data, config_params['sequence_length'])

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=config_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config_params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config_params['batch_size'], shuffle=False)

    # Crear modelo
    model = FinancialTransformer(
        input_features=len(selected_features),
        d_model=config_params['d_model'],
        nhead=config_params['nhead'],
        num_layers=config_params['num_layers'],
        dim_feedforward=config_params['dim_feedforward'],
        dropout=config_params['dropout'],
        max_len=config_params['sequence_length'] + 50
    ).to(device)

    # Entrenar
    train_losses, val_losses = train_model_with_curves(
        model, train_loader, val_loader, epochs=300,
        learning_rate=config_params['learning_rate'],
        weight_decay=config_params['weight_decay'],
        warmup_steps=config_params['warmup_steps']
    )

    # Evaluar
    predictions, actuals, final_metrics = evaluate_model(model, test_loader, target_scaler)

    # Guardar
    model_metrics[config_name] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'predictions': predictions,
        'actuals': actuals,
        'metrics': final_metrics,
        'params': config_params
    }

# Gráficos de los mejores modelos
for config_name in model_metrics.keys():
    predictions = model_metrics[config_name]['predictions']
    actuals = model_metrics[config_name]['actuals']
    r2 = model_metrics[config_name]['metrics']['R2']
    params = model_metrics[config_name]['params']

    print(f"\n=== {config_name} ===")
    print(f"d_model: {params['d_model']}, nhead: {params['nhead']}, layers: {params['num_layers']}")

    # 1. Línea: reales vs predicciones
    plt.figure(figsize=(12, 4))
    plt.plot(actuals, label='Real', alpha=0.8, linewidth=1.5, color='blue')
    plt.plot(predictions, label='Predicción', alpha=0.8, linewidth=1.5, color='red')
    plt.title(f'{config_name} - Predicciones vs Valores Reales\nR² = {r2:.4f}')
    plt.xlabel('Índice de tiempo')
    plt.ylabel('Valor')
    plt.xlim(0, len(predictions))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    # 2. Error absoluto
    abs_error = np.abs(actuals - predictions)
    plt.figure(figsize=(12, 4))
    plt.plot(abs_error, color='orange', alpha=0.7)
    plt.title(f'{config_name} - Error Absoluto')
    plt.xlabel('Índice Temporal')
    plt.ylabel('|Real - Predicción|')
    plt.xlim(0, len(abs_error))
    plt.grid(True, alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    # 3. Scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.6, s=8, color='green')
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.title(f'{config_name} - Scatter Plot\nR² = {r2:.4f}')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    # 4. Curvas de pérdida
    plt.figure(figsize=(10, 6))
    plt.plot(model_metrics[config_name]['train_losses'], label='Train Loss', alpha=0.8)
    plt.plot(model_metrics[config_name]['val_losses'], label='Validation Loss', alpha=0.8)
    plt.title(f'{config_name} - Curvas de Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

print("\n=== RESUMEN FINAL ===")
for config_name in model_metrics.keys():
    metrics = model_metrics[config_name]['metrics']
    print(f"\n{config_name}:")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  R²: {metrics['R2']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  Precisión Direccional: {metrics['Directional_Accuracy']:.4f}")

#%% GRID/BAYESIAN SEARCH MODELO TRANSFORMERS 

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# --- CLASES Y FUNCIONES ---

class FinancialTimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, prediction_horizon=1):
        self.features = data[:, :-1]
        self.targets = data[:, -1]
        self.seq_len = sequence_length
        self.prediction_horizon = prediction_horizon

    def __len__(self):
        return int(len(self.features) - self.seq_len - self.prediction_horizon)

    def __getitem__(self, idx):
        X = self.features[idx:idx+self.seq_len]
        y = self.targets[idx+self.seq_len+self.prediction_horizon-1]
        return torch.FloatTensor(X), torch.FloatTensor([y])

class PositionalEncoding(nn.Module):
    """Codificación posicional para Transformers"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class FinancialTransformer(nn.Module):
    def __init__(self, input_features, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=256, dropout=0.1, max_len=200):
        super(FinancialTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_features = input_features
        
        # Proyección de entrada
        self.input_projection = nn.Linear(input_features, d_model)
        
        # Codificación posicional
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Capas del Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Capas de salida
        self.output_layers = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Inicialización de pesos
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()
        
        for layer in self.output_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_features)
        
        # Proyección a d_model
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Añadir codificación posicional
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer
        transformer_out = self.transformer(x)
        
        # Usar solo la última salida temporal para predicción
        last_output = transformer_out[:, -1, :]  # (batch_size, d_model)
        
        # Predicción final
        output = self.output_layers(last_output)
        
        return output

def prepare_data(df_selected, selected_features, sequence_length, 
                 prediction_horizon=1, train_ratio=0.8):
    data = df_selected[selected_features + ['Cierre']].values
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(data[:, :-1])
    target_scaler.fit(data[:, -1].reshape(-1, 1))
    data_normalized = np.column_stack([
        feature_scaler.transform(data[:, :-1]),
        target_scaler.transform(data[:, -1].reshape(-1, 1))
    ])
    train_size = int(len(data_normalized) * train_ratio)
    train_data = data_normalized[:train_size]
    test_data = data_normalized[train_size:]
    return train_data, test_data, feature_scaler, target_scaler

def evaluate_model(model, data_loader, target_scaler):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(batch_y.cpu().numpy().flatten())

    predictions = target_scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)).flatten()
    actuals = target_scaler.inverse_transform(np.array(
        actuals).reshape(-1, 1)).flatten()

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    r2 = r2_score(actuals, predictions)

    # Directional Accuracy
    direction_pred = np.sign(np.diff(predictions))
    direction_real = np.sign(np.diff(actuals))
    directional_accuracy = np.mean(direction_pred == direction_real)

    # Sharpe Ratio
    returns = np.diff(predictions) / predictions[:-1]
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)

    return predictions, actuals, {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy,
        'Sharpe_Ratio': sharpe_ratio
    }

def train_model(model, train_loader, val_loader, epochs=300, 
                learning_rate=0.001, weight_decay=1e-5, warmup_steps=50):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                          weight_decay=weight_decay, betas=(0.9, 0.999))
    
    # Scheduler con warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (epochs - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20

    epoch_iter = tqdm(range(epochs), desc="Épocas", unit="época", leave=False)
    for epoch in epoch_iter:
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            # Gradient clipping para Transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        epoch_iter.set_postfix({
            "Train Loss": f"{train_loss:.5f}",
            "Val Loss": f"{val_loss:.5f}",
            "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_transformer_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(torch.load('best_transformer_model.pth'))
    return train_losses[-1], val_losses[-1], optimizer.param_groups[0]['lr']

def train_model_with_curves(model, train_loader, val_loader, epochs=300, 
                           learning_rate=0.001, weight_decay=1e-5, warmup_steps=50):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                          weight_decay=weight_decay, betas=(0.9, 0.999))
    
    # Scheduler con warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (epochs - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_transformer_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(torch.load('best_transformer_model.pth'))
    return train_losses, val_losses

# --- PARÁMETROS DEL GRID SEARCH PARA TRANSFORMER ---

param_grid = {
    'd_model': [16, 32],           # Dimensión del modelo
    'nhead': [2,4],                # Número de cabezas de atención
    'num_layers': [1, 2, 3],           # Número de capas del encoder
    'dim_feedforward': [128, 64], # Dimensión de la capa feedforward
    'dropout': [0.1, 0.2],        # Dropout
    'learning_rate': [0.0005], # Tasa de aprendizaje
    'weight_decay': [1e-5], # Regularización L2
    'batch_size': [128, 256],         # Tamaño del batch
    'sequence_length': [90, 180],    # Longitud de secuencia
    'warmup_steps': [25, 50]      # Pasos de warmup
    }

# --- EJECUCIÓN PRINCIPAL CON GRID SEARCH ---

# Supongamos que tienes tu DataFrame df_selected ya cargado
# Descomentar la siguiente línea si necesitas cargar datos
# df_selected = pd.read_csv('tu_archivo.csv')

# Obtener automáticamente las columnas y excluir el target
columna_objetivo = 'Cierre'
selected_features = [col for col in list(df_selected.columns) if col != 
                     columna_objetivo]
print("Características seleccionadas:", selected_features)

# Lista para guardar resultados de todos los modelos
resultados_grid = []

# Generar todas las combinaciones de hiperparámetros
keys, values = zip(*param_grid.items())
combinaciones = list(itertools.product(*values))

print(f"Total de combinaciones a evaluar: {len(combinaciones)}")

# Barra de progreso para el grid search
grid_bar = tqdm(combinaciones, desc="Grid Search Transformer", unit="modelo")

for i, comb in enumerate(grid_bar):
    params = dict(zip(keys, comb))
    
    # Validar compatibilidad d_model % nhead == 0
    if params['d_model'] % params['nhead'] != 0:
        continue
    
    grid_bar.set_postfix(**{k: str(v)[:10] for k, v in list(params.items(
        ))[:3]})
    
    try:
        # INICIO: medir tiempo
        start_time = time.time()
        
        # Preparar datos con la ventana temporal actual
        train_data, test_data, feature_scaler, target_scaler = prepare_data(
            df_selected, selected_features, params['sequence_length']
        )
        
        # Crear datasets y dataloaders
        train_dataset = FinancialTimeSeriesDataset(train_data, params[
            'sequence_length'])
        test_dataset = FinancialTimeSeriesDataset(test_data, params[
            'sequence_length'])
        
        # Verificar si hay suficientes datos
        if len(train_dataset) < 10 or len(test_dataset) < 5:
            print(f"Saltando configuración {i+1}: datos insuficientes")
            continue
        
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=params['batch_size'], 
                                shuffle=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=params['batch_size'], 
                              shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], 
                               shuffle=False, drop_last=True)
        
        # Inicializar modelo Transformer
        input_features = len(selected_features)
        model = FinancialTransformer(
            input_features=input_features,
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_layers=params['num_layers'],
            dim_feedforward=params['dim_feedforward'],
            dropout=params['dropout'],
            max_len=params['sequence_length'] + 50
        ).to(device)
        
        # Entrenar modelo
        train_loss, val_loss, final_lr = train_model(
            model, train_loader, val_loader,
            epochs=500, 
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            warmup_steps=params['warmup_steps']
        )
        
        # Evaluar modelo
        predictions, actuals, metrics = evaluate_model(model, test_loader, 
                                                       target_scaler)
        
        # FINAL: medir tiempo
        elapsed_time = time.time() - start_time
        
        # Calcular complejidad del modelo
        total_params = sum(p.numel() for p in model.parameters())
        
        # Guardar resultados
        resultados_grid.append({
            **params,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **metrics,
            'final_learning_rate': final_lr, 
            'execution_time_sec': elapsed_time,
            'model_complexity': total_params
        })
        
    except Exception as e:
        print(f"Error en configuración {i+1}: {str(e)}")
        continue

# Convertir resultados a DataFrame y guardar en CSV
resultados_df = pd.DataFrame(resultados_grid)
resultados_df.to_csv('resultados_grid_search_transformer4.csv', index=False,
                     encoding='utf-8-sig')

print(f"\nGrid Search completado. {len(resultados_df)} configuraciones evaluadas.")
print("Resultados guardados en 'resultados_grid_search_transformer.csv'")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os

results_df = pd.DataFrame(resultados_grid)

# === 1. Selección automática del mejor modelo ===
best_row = results_df.sort_values(by="MAPE", ascending=True).iloc[0]

best_params = {
    "d_model": int(best_row["d_model"]),
    "nhead": int(best_row["nhead"]),
    "num_layers": int(best_row["num_layers"]),
    "dim_feedforward": int(best_row["dim_feedforward"]),
    "dropout": float(best_row["dropout"]),
    "learning_rate": float(best_row["learning_rate"]),
    "weight_decay": float(best_row["weight_decay"]),
    "batch_size": int(best_row["batch_size"]),
    "sequence_length": int(best_row["sequence_length"]),
    "warmup_steps": int(best_row["warmup_steps"])
}

print("Mejores hiperparámetros:", best_params)

# === 2. Preparar los datos ===
sequence_length = best_params["sequence_length"]
batch_size = best_params["batch_size"]

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# === 3. Definición del modelo Transformer ===
class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.input_layer = nn.Linear(1, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_layer(src)
        src = src + self.positional_encoding[:, :src.size(1), :]
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output

# === 4. Entrenamiento del modelo ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerModel(**{k: best_params[k] for k in ['d_model', 'nhead', 'num_layers', 'dim_feedforward', 'dropout']}).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["learning_rate"], weight_decay=best_params["weight_decay"])

num_epochs = 50
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    train_losses.append(epoch_train_loss / len(train_loader))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

# === 5. Guardar el modelo entrenado ===
model_path = "mejor_modelo_transformer.pth"
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en: {model_path}")

# === 6. Evaluar el modelo en test ===
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.squeeze().cpu().numpy())
        actuals.extend(targets.numpy())

predictions = np.array(predictions)
actuals = np.array(actuals)

# === 7. Graficar las predicciones vs valores reales ===
plt.figure(figsize=(12, 6))
plt.plot(actuals, label="Real")
plt.plot(predictions, label="Predicción")
plt.title("Predicción vs Real")
plt.xlabel("Timestep")
plt.ylabel("Valor")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()




# %%


# Parte 1: Espectrogramas por periodo mejorados
periodos = {
    'Crisis 2008': sp_data.loc['2007-07':'2009-06'],
    'COVID-19': sp_data.loc['2020-01':'2021-12'],
    'Recuperación': sp_data.loc['2022-01':'2024-12']
}

for nombre, df_periodo in periodos.items():
    # Suavizar y reducir resolución si es muy larga
    serie_raw = df_periodo['Cierre'].rolling(window=30).mean().dropna()[::2]
    serie = serie_raw.values - np.mean(serie_raw.values)
    fechas = serie_raw.index
    
    coef, freqs = pywt.cwt(serie, np.arange(1, 32), 'cmor', sampling_period=1)
    
    plt.figure(figsize=(10, 4))
    vmax = np.percentile(np.abs(coef), 95)  # eliminar saturación por valores extremos
    extent = [0, len(serie), 32, 1]
    plt.imshow(np.abs(coef), extent=extent, cmap='viridis', aspect='auto', vmax=vmax)
    
    ticks = np.arange(0, len(fechas), max(1, len(fechas) // 6))
    tick_labels = [fechas[i].strftime('%Y-%m') for i in ticks]
    plt.xticks(ticks, tick_labels, rotation=45)
    plt.title(f"Wavelet - {nombre}")
    plt.xlabel("Fecha")
    plt.ylabel("Escala")
    plt.colorbar(label='Magnitud')
    plt.tight_layout()
    plt.show()


#%% Influcencia de empresas en el S&P 500
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def calcular_betas_sectoriales(merged_data, sectores, sp500_data):
    """
    Calcula la beta de cada sector en relación con el S&P 500 y lo representa gráficamente.
    Utiliza la media de los retornos de las empresas por sector.
    """
    # Verificar datos antes de procesar
    print(f"Verificación de datos antes de procesar:")
    print(f"merged_data tiene {merged_data.isna().sum().sum()} valores NaN")
    print(f"sp500_data tiene {sp500_data.isna().sum()} valores NaN")
    
    # Asegurar que los datos son numéricos y limpiar NaN
    merged_data = merged_data.apply(pd.to_numeric, errors='coerce')
    
    # Usar una copia para no modificar los datos originales
    sp500_clean = sp500_data.copy()
    
    # Calcular retornos diarios sin eliminar NaN inicialmente
    retornos_empresas = merged_data.pct_change(fill_method=None)
    retorno_sp500 = sp500_clean.pct_change(fill_method=None)
    
    # Mostrar información sobre los retornos calculados
    print(f"Después de pct_change:")
    print(f"retornos_empresas tiene {retornos_empresas.isna().sum().sum()} valores NaN")
    print(f"retorno_sp500 tiene {retorno_sp500.isna().sum()} valores NaN")
    
    # Eliminar la primera fila (que será NaN debido a pct_change)
    retornos_empresas = retornos_empresas.iloc[1:]
    retorno_sp500 = retorno_sp500.iloc[1:]
    
    # Verificar si aún hay datos para procesar
    if retornos_empresas.empty or retorno_sp500.empty:
        print("Error: No hay suficientes datos después de calcular retornos")
        return pd.DataFrame(columns=['Beta Sectorial'])

    # Alinear fechas
    fechas_comunes = retornos_empresas.index.intersection(retorno_sp500.index)
    if len(fechas_comunes) == 0:
        print("Error: No hay fechas comunes entre los datos de empresas y el S&P 500")
        return pd.DataFrame(columns=['Beta Sectorial'])
    
    print(f"Número de fechas comunes: {len(fechas_comunes)}")
    
    retornos_empresas = retornos_empresas.loc[fechas_comunes]
    retorno_sp500 = retorno_sp500.loc[fechas_comunes]

    # Crear un diccionario que mapea empresas a sectores
    sector_dict = sectores.to_dict()
    
    # Verificar el mapeo de sectores
    print(f"Número de empresas con sector asignado: {sum(1 for empresa in sector_dict if empresa in retornos_empresas.columns)}")
    
    # Agrupar empresas por sector y calcular el retorno medio
    # En lugar de sumar, recopilamos los retornos por sector para calcular la media
    empresas_por_sector = {}
    for empresa, sector in sector_dict.items():
        if empresa in retornos_empresas.columns and pd.notna(sector) and sector != "":
            # Convertir cada columna a numérico para asegurar compatibilidad
            retorno_empresa = pd.to_numeric(retornos_empresas[empresa], errors='coerce')
            
            if sector not in empresas_por_sector:
                empresas_por_sector[sector] = [retorno_empresa]
            else:
                empresas_por_sector[sector].append(retorno_empresa)

    # Calcular la media de los retornos por sector
    retornos_sector = {}
    for sector, lista_retornos in empresas_por_sector.items():
        if lista_retornos:  # Verificar que hay datos
            # Convertir lista de series a DataFrame
            df_sector = pd.concat(lista_retornos, axis=1)
            # Calcular la media por fila (para cada fecha)
            retornos_sector[sector] = df_sector.mean(axis=1)
    
    # Verificar sectores
    print(f"Número de sectores calculados: {len(retornos_sector)}")
    
    if not retornos_sector:
        print("Error: No se pudo asignar ninguna empresa a un sector")
        return pd.DataFrame(columns=['Beta Sectorial'])

    # Convertir a DataFrame
    retornos_sector_df = pd.DataFrame(retornos_sector)
    
    # Calcular beta de cada sector
    betas_sector = {}
    for sector in retornos_sector_df.columns:
        # Eliminar NaN para este sector específico
        sector_data = retornos_sector_df[sector].dropna()
        sp500_data_aligned = retorno_sp500.loc[sector_data.index].dropna()
        
        # Encontrar índices comunes después de eliminar NaN
        common_idx = sector_data.index.intersection(sp500_data_aligned.index)
        
        if len(common_idx) > 5:  # Asegurar suficientes datos para una regresión válida
            sector_data = sector_data.loc[common_idx]
            sp500_data_aligned = sp500_data_aligned.loc[common_idx]
            
            # Verificar finalmente que no hay NaN
            if not sector_data.isna().any() and not sp500_data_aligned.isna().any():
                try:
                    slope, _, _, _, _ = stats.linregress(sp500_data_aligned, sector_data)
                    betas_sector[sector] = slope
                    print(f"Beta calculada para sector {sector}: {slope}")
                except Exception as e:
                    print(f"Error al calcular beta para sector {sector}: {e}")
            else:
                print(f"Advertencia: Aún hay NaN en los datos del sector {sector}")
        else:
            print(f"Advertencia: Menos de 5 puntos de datos para el sector {sector} ({len(common_idx)} encontrados)")

    # Verificar si se calculó alguna beta
    if not betas_sector:
        print("Error: No se pudo calcular beta para ningún sector")
        return pd.DataFrame(columns=['Beta Sectorial'])

    # Mostrar información sobre las betas calculadas
    print(f"Número de betas sectoriales calculadas: {len(betas_sector)}")
    
    # Convertir a DataFrame y ordenar
    betas_df = pd.DataFrame.from_dict(betas_sector, orient='index', columns=['Beta Sectorial'])
    betas_df = betas_df.sort_values(by='Beta Sectorial', ascending=False)

    return betas_df, empresas_por_sector, retornos_empresas, retorno_sp500

def visualizar_betas_sectoriales(betas_df):
    """
    Genera gráficos para visualizar las betas sectoriales.
    """
    if betas_df.empty:
        print("No hay datos de betas para visualizar")
        return
        
    plt.figure(figsize=(12, 6))
    ax = betas_df.sort_values('Beta Sectorial', ascending=False).plot(kind='bar', color='darkblue', legend=False)
    
    plt.axhline(y=1, color='r', linestyle='--', label='Beta = 1 (S&P 500)')
    plt.title('Beta de Cada Sector respecto al S&P 500')
    plt.ylabel('Beta')
    plt.xticks(rotation=45, ha="right")
    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def calcular_influencia_sectorial(betas_df, capitalizacion=None):
    """
    Calcula la influencia de cada sector en el S&P 500 basado en su beta y capitalización de mercado.
    Si no se proporciona capitalización, crea una simulada basada en betas.
    
    Parámetros:
    - betas_df: DataFrame con las betas sectoriales.
    - capitalizacion: Serie con la capitalización de cada sector (opcional).
    
    Retorna:
    - DataFrame con la contribución de cada sector al movimiento del índice.
    """
    # Si no hay capitalización, crear una simulada basada en la beta
    if capitalizacion is None or len(capitalizacion) == 0:
        print("Creando capitalizaciones simuladas para los sectores basadas en el índice S&P 500 real")
        # Capitalización simulada por sector
        sectores_sp500 = {
            'Tecnologia de la informacion': 26.8,
            'Salud': 13.1,
            'Financieros': 12.8,
            'Bienes de Consumo Discrecional': 9.8,
            'Servicios de Comunicacion': 8.7,
            'Industriales': 8.4,
            'Bienes de Consumo Básico': 6.9,
            'Energia': 4.5,
            'Servicios publicos': 2.7,
            'Bienes Raices': 2.5,
            'Materiales': 2.4,
            'Semiconductores': 1.4  # No es un sector GICS pero está en tus datos
        }
        
        # Crear Series con la capitalización de cada sector
        capitalizacion = pd.Series(sectores_sp500)
        
        # Filtrar para mantener solo sectores que estén en betas_df
        capitalizacion = capitalizacion[capitalizacion.index.isin(betas_df.index)]
        
        print("Capitalización de mercado simulada por sector:")
        print(capitalizacion)
        
        # Si aún no tenemos datos, crear valores aleatorios
        if len(capitalizacion) == 0:
            print("Usando capitalización aleatoria para los sectores")
            cap_dict = {sector: np.random.uniform(1, 10) for sector in betas_df.index}
            capitalizacion = pd.Series(cap_dict)
    
    # Asegurar que la capitalización sea numérica
    capitalizacion = capitalizacion.apply(pd.to_numeric, errors='coerce')
    
    # Normalizar la capitalización (proporción del total)
    capitalizacion_total = capitalizacion.sum()
    if capitalizacion_total == 0:
        print("Error: La capitalización total es 0")
        return pd.DataFrame()
    
    peso_sector = capitalizacion / capitalizacion_total  # Ponderaciones de cada sector en el índice
    
    # Crear DataFrame con los sectores de betas_df
    datos_sector = pd.DataFrame(index=betas_df.index)
    datos_sector["Beta Sectorial"] = betas_df["Beta Sectorial"]
    
    # Añadir pesos, asegurando que todos los sectores están presentes
    for sector in datos_sector.index:
        if sector in peso_sector.index:
            datos_sector.loc[sector, "Peso"] = peso_sector[sector]
        else:
            # Asignar un peso pequeño si no hay datos
            datos_sector.loc[sector, "Peso"] = 0.001
    
    # Calcular la contribución de cada sector al índice
    datos_sector["Influencia Relativa"] = datos_sector["Beta Sectorial"] * datos_sector["Peso"]
    
    # Ordenar por influencia relativa
    datos_sector = datos_sector.sort_values("Influencia Relativa", ascending=False)
    
    return datos_sector

def visualizar_influencia_sectorial(datos_sector):
    """
    Genera un gráfico de barras con la influencia relativa de cada sector en el S&P 500.
    
    Parámetros:
    - datos_sector: DataFrame con betas, peso en el índice y la influencia relativa.
    """
    if datos_sector.empty:
        print("No hay datos de influencia para visualizar")
        return
    
    plt.figure(figsize=(12, 6))
    # Crear un gráfico de barras con colores basados en el valor
    ax = datos_sector["Influencia Relativa"].plot(
        kind="bar", 
        color=plt.cm.viridis(np.linspace(0, 1, len(datos_sector)))
    )
    
    # Añadir etiquetas con el valor de beta y peso
    for i, (idx, row) in enumerate(datos_sector.iterrows()):
        ax.annotate(
            f"β={row['Beta Sectorial']:.2f}\nPeso={row['Peso']:.2%}", 
            xy=(i, row["Influencia Relativa"]), 
            xytext=(0, 5),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=8
        )
    
    plt.axhline(y=0, color='black', linestyle='-')
    plt.title("Influencia Relativa de Cada Sector en el S&P 500")
    plt.ylabel("Influencia Relativa")
    plt.xticks(rotation=45, ha="right")
    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def calcular_empresas_influyentes(empresas_por_sector, retornos_empresas, retorno_sp500, sector_dict):
    """
    Identifica las empresas más influyentes dentro de cada sector basándose en su beta
    y en cuánto contribuyen a la volatilidad del sector.
    
    Parámetros:
    - empresas_por_sector: Diccionario que mapea sectores a listas de empresas
    - retornos_empresas: DataFrame con retornos de todas las empresas
    - retorno_sp500: Serie con retornos del S&P 500
    - sector_dict: Diccionario que mapea empresas a sectores
    
    Retorna:
    - DataFrame con las empresas más influyentes por sector
    """
    print("Calculando influencia de empresas individuales dentro de cada sector...")
    
    # Calcular betas individuales de cada empresa respecto al S&P 500
    betas_empresas = {}
    r2_empresas = {}  # También calcularemos el R² para medir la calidad del ajuste
    
    for empresa in retornos_empresas.columns:
        try:
            # Limpiar datos
            retorno_empresa = pd.to_numeric(retornos_empresas[empresa], errors='coerce').dropna()
            sp500_aligned = retorno_sp500.loc[retorno_empresa.index].dropna()
            
            # Encontrar índices comunes
            common_idx = retorno_empresa.index.intersection(sp500_aligned.index)
            
            if len(common_idx) > 10:  # Asegurar suficientes datos
                retorno_empresa = retorno_empresa.loc[common_idx]
                sp500_aligned = sp500_aligned.loc[common_idx]
                
                if not retorno_empresa.isna().any() and not sp500_aligned.isna().any():
                    slope, intercept, r_value, p_value, std_err = stats.linregress(sp500_aligned, retorno_empresa)
                    betas_empresas[empresa] = slope
                    r2_empresas[empresa] = r_value**2
        except Exception as e:
            print(f"Error al calcular beta para empresa {empresa}: {e}")
    
    # Convertir a DataFrame
    resultados_empresas = pd.DataFrame({
        'Beta': pd.Series(betas_empresas),
        'R²': pd.Series(r2_empresas)
    })
    
    # Añadir sector a cada empresa
    resultados_empresas['Sector'] = resultados_empresas.index.map(lambda x: sector_dict.get(x, 'Desconocido'))
    
    # Calcular la volatilidad de cada empresa (desviación estándar de los retornos)
    volatilidades = {}
    for empresa in retornos_empresas.columns:
        retorno_empresa = pd.to_numeric(retornos_empresas[empresa], errors='coerce').dropna()
        if len(retorno_empresa) > 10:
            volatilidades[empresa] = retorno_empresa.std()
    
    resultados_empresas['Volatilidad'] = pd.Series(volatilidades)
    
    # Calcular un score de influencia para cada empresa
    # El score combina beta, R² y volatilidad
    resultados_empresas['Score_Influencia'] = (
        resultados_empresas['Beta'].abs() * 
        resultados_empresas['R²'] * 
        resultados_empresas['Volatilidad']
    )
    
    # Normalizar el score por sector
    for sector in resultados_empresas['Sector'].unique():
        if sector == 'Desconocido':
            continue
            
        sector_mask = resultados_empresas['Sector'] == sector
        max_score = resultados_empresas.loc[sector_mask, 'Score_Influencia'].max()
        
        if max_score > 0:
            resultados_empresas.loc[sector_mask, 'Score_Normalizado'] = (
                resultados_empresas.loc[sector_mask, 'Score_Influencia'] / max_score
            )
    
    # Ordenar resultados
    resultados_ordenados = resultados_empresas.sort_values('Score_Influencia', ascending=False)
    
    return resultados_ordenados

import matplotlib.pyplot as plt
import seaborn as sns

def visualizar_empresas_mayor_beta(resultados_empresas, num_sectores=8, num_empresas=5):
    """
    Genera visualizaciones de las empresas con mayor Beta dentro de cada sector.
    
    Parámetros:
    - resultados_empresas: DataFrame con datos de Beta de cada empresa
    - num_sectores: Número de sectores a mostrar
    - num_empresas: Número de empresas por sector a mostrar
    """
    if resultados_empresas.empty:
        print("No hay datos de empresas para visualizar")
        return
    
    # Identificar los sectores principales (según la suma de las Betas, o puedes mantener Score_Influencia si lo prefieres)
    top_sectores = resultados_empresas.groupby('Sector')['Beta'].sum().nlargest(num_sectores).index
    
    # Crear un gráfico para cada sector
    plt.figure(figsize=(15, 10))
    
    for i, sector in enumerate(top_sectores):
        # Obtener las empresas con mayor Beta para este sector
        empresas_sector = resultados_empresas[resultados_empresas['Sector'] == sector].nlargest(num_empresas, 'Beta')
        
        # Crear subplot
        plt.subplot(len(top_sectores) // 2 + len(top_sectores) % 2, 2, i+1)
        
        # Crear barras horizontales
        bars = plt.barh(empresas_sector.index, empresas_sector['Beta'], color='salmon')
        
        # Añadir etiquetas con valor de Score de Influencia
        for bar, score in zip(bars, empresas_sector['Score_Influencia']):
            plt.text(
                bar.get_width() * 1.01, 
                bar.get_y() + bar.get_height()/2, 
                f'Score={score:.2f}', 
                va='center'
            )
        
        plt.title(f'Sector: {sector}')
        plt.xlabel('Beta')
        plt.grid(False)
        plt.tight_layout()
    
    plt.suptitle('Empresas con Mayor Beta por Sector', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Crear un gráfico global de las empresas con mayor Beta
    plt.figure(figsize=(12, 8))
    
    # Tomar las top empresas globales con mayor Beta
    top_empresas = resultados_empresas.nlargest(20, 'Beta')
    
    # Crear un mapa de colores basado en sectores
    sectores_unicos = top_empresas['Sector'].unique()
    color_map = dict(zip(sectores_unicos, sns.color_palette("tab10", len(sectores_unicos))))
    bar_colors = [color_map[sector] for sector in top_empresas['Sector']]
    
    # Crear barras
    bars = plt.barh(top_empresas.index, top_empresas['Beta'], color=bar_colors)
    
    # Añadir etiquetas con valor de Score de Influencia y sector
    for bar, score, sector in zip(bars, top_empresas['Score_Influencia'], top_empresas['Sector']):
        plt.text(
            bar.get_width() * 1.01, 
            bar.get_y() + bar.get_height()/2, 
            f'Score={score:.2f} ({sector})', 
            va='center'
        )
    
    plt.title('Top 20 Empresas con Mayor Beta en el S&P 500')
    plt.xlabel('Beta')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Crear leyenda
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[sector], label=sector) for sector in sectores_unicos]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.show()


# Diagnóstico de los datos
print("Información de diagnóstico inicial:")
print(f"Forma de empresas: {empresas.shape if isinstance(empresas, pd.DataFrame) else 'No es un DataFrame'}")
print(f"Número de sectores: {len(sectores) if isinstance(sectores, pd.Series) else 'No es una Series'}")
print(f"Longitud de cierre_sp500: {len(cierre_sp500) if isinstance(cierre_sp500, pd.Series) else 'No es una Series'}")

# Verificar los primeros valores para debug
print("\nPrimeros valores de los datos:")
try:
    print("Primeras filas de empresas:")
    print(empresas.iloc[:3, :3])
    
    print("\nPrimeros valores de sectores:")
    print(sectores.head())
    
    print("\nPrimeros valores de cierre_sp500:")
    print(cierre_sp500.head())
except Exception as e:
    print(f"Error al imprimir valores: {e}")

# Calcular betas sectoriales y obtener datos adicionales
betas_sectores, empresas_por_sector, retornos_empresas, retorno_sp500 = calcular_betas_sectoriales(empresas, sectores, cierre_sp500)

# Mostrar resultados de betas sectoriales
print("\nBetas Sectoriales:")
print(betas_sectores)

# Visualizar betas sectoriales
visualizar_betas_sectoriales(betas_sectores)

# Calcular la influencia sectorial con capitalización simulada
datos_sector = calcular_influencia_sectorial(betas_sectores, None)  # None para generar capitalización simulada

# Mostrar resultados de influencia sectorial
print("\nInfluencia de Cada Sector en el S&P 500:")
print(datos_sector)
