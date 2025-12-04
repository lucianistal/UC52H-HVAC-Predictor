# UC52H HVAC Predictor - Sistema de Predicción ML

Sistema de aprendizaje automático basado en Random Forest para la predicción de temperatura de salida de aire en sistemas HVAC.

##  Descripción

Este trabajo presenta el desarrollo e implementación de un modelo predictivo mediante técnicas de Machine Learning para estimar la temperatura de salida de aire (UCAOT) en un sistema de climatización UC52H. El modelo ha sido entrenado utilizando datos reales de operación y alcanza niveles de precisión superiores al 99% en ambos modos operacionales.

El sistema implementa dos modelos independientes especializados en las condiciones de verano e invierno, reconociendo que las dinámicas térmicas y las variables relevantes difieren significativamente entre ambos períodos.

##  Resultados

### Modelo Verano
El modelo de verano procesa 8 datasets correspondientes a diferentes condiciones operacionales, totalizando 43,707 registros. Tras el proceso de limpieza y filtrado de valores atípicos, se utilizaron 27,006 registros para el entrenamiento y validación del modelo.

**Métricas de rendimiento:**
- Coeficiente de determinación (R²): 0.9995 (99.95%)
- Error cuadrático medio (RMSE): 0.057°C
- Error absoluto medio (MAE): 0.034°C

**Variables de entrada:** UCWIT, UCAIT, UCWF, UCAF, UCFDP

**Análisis de importancia:** La temperatura de entrada del aire (UCAIT) resulta ser la variable más influyente con un 76.5% de importancia, seguida por la temperatura del agua de entrada (UCWIT) con 16.8%.

### Modelo Invierno
El modelo de invierno analiza 7 datasets con un total de 20,087 registros, de los cuales 17,642 superaron los criterios de validación para su uso en el entrenamiento.

**Métricas de rendimiento:**
- Coeficiente de determinación (R²): 0.9918 (99.18%)
- Error cuadrático medio (RMSE): 0.592°C
- Error absoluto medio (MAE): 0.278°C

**Variables de entrada:** UCAIT, UCAF, UCHV, UCHC, UCFDP

##  Estructura del Proyecto

```
HVAC/
├── src/
│   └── modelo_uc52h.py           # Entrenamiento del modelo
├── simulador.py                   # Predictor independiente
├── data/
│   ├── DS01_sum_*.csv            # Datasets modo verano
│   ├── DS03_win_*.csv            # Datasets modo invierno
│   └── unit_cooler_data_COMPLETE_2023-2024-2025.csv
├── output/
│   ├── RESULTADOS_UC52H.xlsx     # Resultados del entrenamiento
│   ├── PREDICCIONES_VERANO_SIMULADAS.csv
│   └── PREDICCIONES_INVIERNO_SIMULADAS.csv
├── docs/
│   └── MEMORIA.txt
├── DATOS_SIMULACION_VERANO.csv   # Datos de entrada para simular
├── DATOS_SIMULACION_INVIERNO.csv # Datos de entrada para simular
├── guia_simulador.txt            # Guía de uso del simulador
├── interpretacion_resultados.txt # Cómo interpretar resultados
├── README.md
└── requirements.txt
```

##  Instalación

### Requisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalar dependencias

```bash
pip install -r requirements.txt
```

O instalar manualmente:
```bash
pip install pandas numpy scikit-learn openpyxl
```

Verificar instalación:
```bash
python -c "import pandas, numpy, sklearn, openpyxl; print('Instalacion correcta')"
```

##  Uso

### 1. Entrenar el modelo

Ejecuta el script principal para entrenar los modelos:

```bash
python src/modelo_uc52h.py
```

**Tiempo de ejecución:** ~10 segundos

El sistema realizará:
- Búsqueda y carga de datasets
- Preprocesamiento (limpieza, filtrado outliers)
- Entrenamiento de modelos verano/invierno
- Evaluación con métricas R², RMSE, MAE
- Generación de archivo Excel con resultados

**Salida:** `output/RESULTADOS_UC52H.xlsx`

### 2. Hacer predicciones con nuevos datos

Usa el simulador para predecir con datos nuevos:

```bash
python simulador.py
```

El simulador:
1. Carga los modelos entrenados
2. Lee los archivos CSV de entrada:
   - `DATOS_SIMULACION_VERANO.csv`
   - `DATOS_SIMULACION_INVIERNO.csv`
3. Genera predicciones de temperatura de salida (UCAOT)
4. Guarda resultados en:
   - `output/PREDICCIONES_VERANO_SIMULADAS.csv`
   - `output/PREDICCIONES_INVIERNO_SIMULADAS.csv`

**Consulta `guia_simulador.txt` para más detalles sobre cómo usar el simulador.**

##  Variables del Sistema

### Variables de Entrada - Modo Verano

| Variable | Descripción | Unidad | Rango típico |
|----------|-------------|--------|--------------|
| UCWIT | Temperatura entrada agua | °C | 15-25 |
| UCAIT | Temperatura entrada aire | °C | 20-35 |
| UCWF | Caudal de agua | L/min | 5-25 |
| UCAF | Caudal de aire | m³/h | 300-400 |
| UCFDP | Presión diferencial ventilador | Pa | 0-150 |

### Variables de Entrada - Modo Invierno

| Variable | Descripción | Unidad | Rango típico |
|----------|-------------|--------|--------------|
| UCAIT | Temperatura entrada aire | °C | 15-25 |
| UCAF | Caudal de aire | m³/h | 1200-1500 |
| UCHV | Voltaje resistencia calefacción | V | 380-420 |
| UCHC | Corriente resistencia calefacción | A | 0-5 |
| UCFDP | Presión diferencial ventilador | Pa | 90-110 |

### Variable de Salida

| Variable | Descripción | Unidad |
|----------|-------------|--------|
| UCAOT | Temperatura salida aire | °C |

##  Archivo Excel de Resultados

`output/RESULTADOS_UC52H.xlsx` contiene 4 hojas:

1. **Resumen:** Métricas generales, importancia de variables
2. **Predicciones Verano:** 1000 predicciones con valores reales vs predichos
3. **Predicciones Invierno:** 1000 predicciones con valores reales vs predichos
4. **Estadísticas:** Análisis descriptivo de todas las variables

##  Modelo Técnico

**Algoritmo:** Random Forest (Bosque Aleatorio)

**Hiperparámetros:**
- Número de árboles: 200
- Profundidad máxima: 20
- Muestras mínimas por división: 5
- Muestras mínimas por hoja: 2
- Estado aleatorio: 42 (reproducibilidad)
- Procesamiento paralelo: activado

**Preprocesamiento:**
- Limpieza de valores nulos
- Filtrado de outliers (método IQR, percentil 1-99)
- División train-test: 80%-20%

##  Documentación Adicional

- **`guia_simulador.txt`:** Guía completa para usar el simulador
- **`interpretacion_resultados.txt`:** Cómo interpretar los resultados
- **`docs/MEMORIA.txt`:** Memoria técnica completa del proyecto
