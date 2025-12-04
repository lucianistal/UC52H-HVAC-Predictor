# Modelo Predictivo para Sistema de Climatización UC52H

Sistema de aprendizaje automático basado en Random Forest para la predicción de temperatura de salida de aire en sistemas HVAC.

---

## Descripción del Proyecto

Este trabajo presenta el desarrollo e implementación de un modelo predictivo mediante técnicas de Machine Learning para estimar la temperatura de salida de aire (UCAOT) en un sistema de climatización UC52H. El modelo ha sido entrenado utilizando datos reales de operación y alcanza niveles de precisión superiores al 99% en ambos modos operacionales.

El sistema implementa dos modelos independientes especializados en las condiciones de verano e invierno, reconociendo que las dinámicas térmicas y las variables relevantes difieren significativamente entre ambos períodos.

---

## Resultados Principales

### Modo Verano

El modelo de verano procesa 8 datasets correspondientes a diferentes condiciones operacionales, totalizando 43,707 registros. Tras el proceso de limpieza y filtrado de valores atípicos, se utilizaron 27,006 registros para el entrenamiento y validación del modelo.

**Métricas de rendimiento:**
- Coeficiente de determinación (R²): 0.9995 (99.95%)
- Error cuadrático medio (RMSE): 0.057°C
- Error absoluto medio (MAE): 0.034°C

**Variables de entrada:** UCWIT, UCAIT, UCWF, UCAF, UCFDP

**Análisis de importancia:**
La temperatura de entrada del aire (UCAIT) resulta ser la variable más influyente con un 76.5% de importancia, seguida por la temperatura del agua de entrada (UCWIT) con 16.8%. Esta distribución es consistente con los principios físicos que rigen el intercambio térmico en el sistema.

### Modo Invierno

El modelo de invierno analiza 7 datasets con un total de 20,087 registros, de los cuales 17,642 superaron los criterios de validación para su uso en el entrenamiento.

**Métricas de rendimiento:**
- Coeficiente de determinación (R²): 0.9918 (99.18%)
- Error cuadrático medio (RMSE): 0.592°C
- Error absoluto medio (MAE): 0.278°C

**Variables de entrada:** UCAIT, UCAF, UCHV, UCHC, UCFDP

**Análisis de importancia:**
La temperatura de entrada del aire (UCAIT) domina el modelo con una importancia del 94.5%, lo cual refleja el cambio de modo operacional del sistema de enfriamiento a calefacción.

---

## Estructura del Proyecto

```
HVAC/
├── src/
│   └── modelo_uc52h.py            Código principal del modelo
├── data/
│   ├── DS01_sum_*.csv             Datasets modo verano
│   ├── DS03_win_*.csv             Datasets modo invierno
│   └── unit_cooler_data_*.csv     Dataset completo (opcional)
├── output/
│   └── RESULTADOS_UC52H.xlsx      Archivo Excel con resultados
├── docs/
│   └── MEMORIA.txt                Memoria técnica completa
├── README.md                      Este archivo
├── requirements.txt               Dependencias del proyecto
└── INSTRUCCIONES.txt              Guía de uso
```

---

## Requisitos del Sistema

### Software necesario

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Librerías requeridas

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
openpyxl>=3.0.0
```

---

## Instalación

### Paso 1: Clonar o descargar el proyecto

Descargar la carpeta completa del proyecto manteniendo la estructura de directorios.

### Paso 2: Instalar dependencias

Abrir una terminal en la carpeta raíz del proyecto y ejecutar:

```bash
pip install -r requirements.txt
```

Alternativamente, instalar cada librería individualmente:

```bash
pip install pandas numpy scikit-learn openpyxl
```

### Paso 3: Verificar la instalación

```bash
python -c "import pandas, numpy, sklearn, openpyxl; print('Instalacion correcta')"
```

---

## Uso del Sistema

### Ejecución básica

Desde la carpeta raíz del proyecto:

```bash
python src/modelo_uc52h_FINAL.py
```

### Proceso de ejecución

El sistema realiza automáticamente las siguientes operaciones:

1. **Búsqueda de archivos:** Localiza todos los archivos CSV en la carpeta data/
2. **Clasificación:** Identifica archivos de verano (sum) e invierno (win)
3. **Carga de datos:** Lee y combina los datasets con manejo de múltiples codificaciones
4. **Preprocesamiento:** Limpia valores nulos y filtra outliers mediante método IQR
5. **Entrenamiento:** Genera dos modelos Random Forest independientes
6. **Evaluación:** Calcula métricas de rendimiento (R², RMSE, MAE)
7. **Predicciones:** Genera predicciones sobre los datos de prueba
8. **Exportación:** Crea archivo Excel con análisis completo

### Tiempo de ejecución estimado

- Carga de datos: 2 segundos
- Preprocesamiento: 1 segundo
- Entrenamiento modelo verano: 3 segundos
- Entrenamiento modelo invierno: 2 segundos
- Generación de resultados: 2 segundos

**Total aproximado:** 10 segundos

---

## Descripción de Variables

### Variables de entrada (Verano)

| Variable | Descripción | Unidad | Rango típico |
|----------|-------------|--------|--------------|
| UCWIT | Temperatura entrada agua | °C | 15-25 |
| UCAIT | Temperatura entrada aire | °C | 20-35 |
| UCWF | Caudal de agua | L/min | 5-25 |
| UCAF | Caudal de aire | m³/h | 300-400 |
| UCFDP | Presión diferencial ventilador | Pa | 0-150 |

### Variables de entrada (Invierno)

| Variable | Descripción | Unidad | Rango típico |
|----------|-------------|--------|--------------|
| UCAIT | Temperatura entrada aire | °C | 15-25 |
| UCAF | Caudal de aire | m³/h | 1200-1500 |
| UCHV | Voltaje resistencia calefacción | V | 380-420 |
| UCHC | Corriente resistencia calefacción | A | 0-5 |
| UCFDP | Presión diferencial ventilador | Pa | 90-110 |

### Variable de salida

| Variable | Descripción | Unidad |
|----------|-------------|--------|
| UCAOT | Temperatura salida aire | °C |

---

## Archivo de Resultados

El sistema genera automáticamente el archivo `output/RESULTADOS_UC52H.xlsx` con las siguientes hojas:

### Hoja 1: Resumen

Contiene la información general del modelo incluyendo fecha de ejecución, algoritmo utilizado, métricas de rendimiento para ambos modos, e importancia de las variables predictoras.

### Hoja 2: Predicciones Verano

Muestra las primeras 1000 predicciones del modelo de verano, incluyendo valores reales, valores predichos, error absoluto de cada predicción, y capacidad de enfriamiento calculada.

### Hoja 3: Predicciones Invierno

Presenta las primeras 1000 predicciones del modelo de invierno con valores reales, predichos, errores, y capacidad de calentamiento.

### Hoja 4: Estadísticas

Análisis estadístico descriptivo de todas las variables (media, mediana, desviación estándar, valores mínimos y máximos) para ambos modos operacionales.

---

## Metodología

### Algoritmo utilizado

Random Forest (Bosque Aleatorio) con los siguientes hiperparámetros:

- Número de árboles: 200
- Profundidad máxima: 20
- Muestras mínimas por división: 5
- Muestras mínimas por hoja: 2
- Estado aleatorio: 42 (reproducibilidad)
- Procesamiento paralelo: activado

### Preprocesamiento de datos

**Limpieza de valores nulos:**
Se eliminan registros con valores faltantes en las variables de entrada o salida.

**Filtrado de outliers:**
Método de rango intercuartílico (IQR) con factor 1.5. Se calcula el percentil 1 y 99 para cada variable y se descartan valores fuera del rango aceptable.

**División train-test:**
80% datos de entrenamiento, 20% datos de prueba. División aleatoria con semilla fija para reproducibilidad.

### Métricas de evaluación

**R² (Coeficiente de determinación):**
Indica el porcentaje de variabilidad explicada por el modelo. Valores cercanos a 1 representan mejor ajuste.

**RMSE (Root Mean Square Error):**
Error cuadrático medio que penaliza más los errores grandes. Se expresa en las mismas unidades que la variable objetivo (°C).

**MAE (Mean Absolute Error):**
Error absoluto medio que representa la desviación promedio de las predicciones. Más robusto ante outliers que RMSE.

---

## Interpretación de Resultados

### Precisión del modelo

Los valores de R² superiores a 0.99 indican que el modelo captura prácticamente toda la variabilidad de los datos. Esto se traduce en predicciones altamente confiables para ambos modos operacionales.

### Magnitud del error

Los errores MAE de 0.034°C (verano) y 0.278°C (invierno) son excepcionalmente bajos para sistemas de climatización, donde errores de 1-2°C se consideran aceptables en aplicaciones industriales.

### Importancia de variables

La dominancia de UCAIT en ambos modelos es físicamente consistente, ya que la temperatura del aire de entrada determina en gran medida la carga térmica del sistema y, por tanto, la temperatura de salida.

---

## Solución de Problemas

### No se encuentran archivos CSV

**Causa:** Los archivos no están en la carpeta data/ o tienen nombres no reconocidos.

**Solución:** Verificar que los archivos CSV están ubicados en `HVAC/data/` y contienen "sum" (verano) o "win" (invierno) en sus nombres.

### Error de módulos no encontrados

**Causa:** Las librerías necesarias no están instaladas.

**Solución:** Ejecutar `pip install -r requirements.txt` o instalar cada librería individualmente.

### Pocos registros procesados

**Causa:** Alta presencia de valores atípicos o datos faltantes.

**Solución:** Esto es normal. El sistema filtra automáticamente datos de baja calidad. Los resultados siguen siendo válidos con los datos limpios.

### Error al generar Excel

**Causa:** El archivo Excel está abierto o no hay permisos de escritura.

**Solución:** Cerrar el archivo `RESULTADOS_UC52H.xlsx` si está abierto y verificar permisos en la carpeta output/.
