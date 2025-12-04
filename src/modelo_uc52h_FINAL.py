#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
MODELO PREDICTIVO UC52H - VERSION FINAL
Sistema de Prediccion de Temperatura de Salida de Aire
===============================================================================
VARIABLES:
    Entrada Verano: UCWIT, UCAIT, UCWF, UCAF, UCFDP
    Entrada Invierno: UCAIT, UCAF, UCHV, UCHC, UCFDP
    Salida: UCAOT (Temperatura salida aire)

===============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
import sys
import os
import glob
from datetime import datetime

warnings.filterwarnings('ignore')

class ModeloUC52H:
    """Modelo predictivo para sistema UC52H"""
    
    def __init__(self):
        """Inicializar modelo"""
        self.modelo_verano = None
        self.modelo_invierno = None
        self.datos_verano = None
        self.datos_invierno = None
        self.metricas_verano = {}
        self.metricas_invierno = {}
        
        # Variables para cada modo
        self.vars_entrada_verano = ['UCWIT', 'UCAIT', 'UCWF', 'UCAF', 'UCFDP']
        self.vars_entrada_invierno = ['UCAIT', 'UCAF', 'UCHV', 'UCHC', 'UCFDP']
        self.var_salida = 'UCAOT'
        
        # Parametros del modelo
        self.params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def separador(self, texto=""):
        """Imprimir separador visual"""
        print("\n" + "="*80)
        if texto:
            print(f"{texto:^80}")
        print("="*80 + "\n")
    
    def subseparador(self, texto):
        """Imprimir subseparador"""
        print("\n" + "-"*80)
        print(f"{texto:^80}")
        print("-"*80 + "\n")
    
    def cargar_csv(self, ruta):
        """Cargar CSV con multiples codificaciones y separadores"""
        separadores = [';', ',', '\t']
        codificaciones = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']
        
        for encoding in codificaciones:
            for sep in separadores:
                try:
                    df = pd.read_csv(ruta, sep=sep, encoding=encoding)
                    if len(df.columns) > 10:  # Debe tener muchas columnas
                        return df
                except:
                    continue
        
        return None
    
    def buscar_y_cargar_datos(self, carpeta_data):
        """Buscar y cargar todos los archivos CSV"""
        self.subseparador("BUSCANDO ARCHIVOS DE DATOS")
        
        # Buscar archivos
        archivos_verano = []
        archivos_invierno = []
        
        # Buscar en carpeta data
        if os.path.exists(carpeta_data):
            todos_csv = glob.glob(os.path.join(carpeta_data, "*.csv"))
            
            for archivo in todos_csv:
                nombre = os.path.basename(archivo).lower()
                if 'sum' in nombre or 'DS01' in archivo or 'DS02' in archivo or \
                   'DS06' in archivo or 'DS07' in archivo or 'DS08' in archivo or \
                   'DS09' in archivo or 'DS10' in archivo or 'DS11' in archivo:
                    if 'test' not in nombre and 'complete' not in nombre:
                        archivos_verano.append(archivo)
                elif 'win' in nombre or 'DS03' in archivo or 'DS04' in archivo or \
                     'DS05' in archivo or 'DS12' in archivo or 'DS13' in archivo or \
                     'DS14' in archivo or 'DS15' in archivo:
                    if 'complete' not in nombre:
                        archivos_invierno.append(archivo)
        
        print(f"Archivos de VERANO encontrados: {len(archivos_verano)}")
        print(f"Archivos de INVIERNO encontrados: {len(archivos_invierno)}")
        
        # Cargar datos de verano
        if archivos_verano:
            print("\nCargando datos de VERANO...")
            dfs_verano = []
            for archivo in archivos_verano:
                df = self.cargar_csv(archivo)
                if df is not None:
                    # Verificar que tiene las columnas necesarias
                    columnas_requeridas = self.vars_entrada_verano + [self.var_salida]
                    if all(col in df.columns for col in columnas_requeridas):
                        df_limpio = df[columnas_requeridas].dropna()
                        dfs_verano.append(df_limpio)
                        print(f"  [OK] {os.path.basename(archivo)}: {len(df_limpio)} registros")
                    else:
                        print(f"  [OMITIDO] {os.path.basename(archivo)}: columnas faltantes")
            
            if dfs_verano:
                self.datos_verano = pd.concat(dfs_verano, ignore_index=True)
                print(f"\nTotal registros VERANO: {len(self.datos_verano)}")
        
        # Cargar datos de invierno
        if archivos_invierno:
            print("\nCargando datos de INVIERNO...")
            dfs_invierno = []
            for archivo in archivos_invierno:
                df = self.cargar_csv(archivo)
                if df is not None:
                    # Verificar columnas necesarias
                    columnas_requeridas = self.vars_entrada_invierno + [self.var_salida]
                    if all(col in df.columns for col in columnas_requeridas):
                        df_limpio = df[columnas_requeridas].dropna()
                        dfs_invierno.append(df_limpio)
                        print(f"  [OK] {os.path.basename(archivo)}: {len(df_limpio)} registros")
                    else:
                        print(f"  [OMITIDO] {os.path.basename(archivo)}: columnas faltantes")
            
            if dfs_invierno:
                self.datos_invierno = pd.concat(dfs_invierno, ignore_index=True)
                print(f"\nTotal registros INVIERNO: {len(self.datos_invierno)}")
        
        return (self.datos_verano is not None), (self.datos_invierno is not None)
    
    def limpiar_datos(self, df, vars_entrada):
        """Limpiar y filtrar datos"""
        # Eliminar duplicados
        df = df.drop_duplicates()
        
        # Filtrar outliers usando IQR
        df_limpio = df.copy()
        for col in vars_entrada + [self.var_salida]:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            IQR = Q3 - Q1
            filtro = (df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)
            df_limpio = df_limpio[filtro]
        
        return df_limpio
    
    def entrenar_modelo(self, df, vars_entrada, nombre_modo):
        """Entrenar modelo Random Forest"""
        self.subseparador(f"ENTRENANDO MODELO {nombre_modo}")
        
        # Limpiar datos
        df_limpio = self.limpiar_datos(df, vars_entrada)
        print(f"Registros despues de limpieza: {len(df_limpio)}")
        
        # Separar X e y
        X = df_limpio[vars_entrada]
        y = df_limpio[self.var_salida]
        
        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Entrenamiento: {len(X_train)} registros")
        print(f"Prueba: {len(X_test)} registros")
        
        # Entrenar modelo
        print("\nEntrenando Random Forest...")
        modelo = RandomForestRegressor(**self.params)
        modelo.fit(X_train, y_train)
        
        # Predecir y evaluar
        y_pred = modelo.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\nMETRICAS:")
        print(f"  R² = {r2:.6f} ({r2*100:.2f}%)")
        print(f"  RMSE = {rmse:.4f} °C")
        print(f"  MAE = {mae:.4f} °C")
        
        # Importancia de variables
        importancias = dict(zip(vars_entrada, modelo.feature_importances_))
        print("\nIMPORTANCIA DE VARIABLES:")
        for var, imp in sorted(importancias.items(), key=lambda x: x[1], reverse=True):
            print(f"  {var}: {imp*100:.2f}%")
        
        # Guardar métricas
        metricas = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'importancias': importancias
        }
        
        return modelo, metricas
    
    def generar_predicciones(self, modelo, df_entrada, vars_entrada):
        """Generar predicciones sobre nuevos datos"""
        X = df_entrada[vars_entrada]
        predicciones = modelo.predict(X)
        return predicciones
    
    def crear_excel_resultados(self, ruta_salida):
        """Crear archivo Excel con todos los resultados"""
        self.subseparador("GENERANDO ARCHIVO EXCEL")
        
        try:
            with pd.ExcelWriter(ruta_salida, engine='openpyxl') as writer:
                
                # HOJA 1: Resumen General
                resumen_data = []
                resumen_data.append(['MODELO PREDICTIVO UC52H - RESULTADOS'])
                resumen_data.append([''])
                resumen_data.append(['Fecha de ejecución:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                resumen_data.append(['Algoritmo:', 'Random Forest'])
                resumen_data.append([''])
                
                if self.modelo_verano is not None:
                    resumen_data.append(['=== RESULTADOS VERANO ==='])
                    resumen_data.append(['Registros entrenamiento:', self.metricas_verano['n_train']])
                    resumen_data.append(['Registros prueba:', self.metricas_verano['n_test']])
                    resumen_data.append(['R² (Precisión):', f"{self.metricas_verano['r2']*100:.2f}%"])
                    resumen_data.append(['RMSE (Error):', f"{self.metricas_verano['rmse']:.4f} °C"])
                    resumen_data.append(['MAE (Error medio):', f"{self.metricas_verano['mae']:.4f} °C"])
                    resumen_data.append([''])
                    resumen_data.append(['Importancia de Variables:'])
                    for var, imp in sorted(self.metricas_verano['importancias'].items(), 
                                         key=lambda x: x[1], reverse=True):
                        resumen_data.append([var, f"{imp*100:.2f}%"])
                    resumen_data.append([''])
                
                if self.modelo_invierno is not None:
                    resumen_data.append(['=== RESULTADOS INVIERNO ==='])
                    resumen_data.append(['Registros entrenamiento:', self.metricas_invierno['n_train']])
                    resumen_data.append(['Registros prueba:', self.metricas_invierno['n_test']])
                    resumen_data.append(['R² (Precisión):', f"{self.metricas_invierno['r2']*100:.2f}%"])
                    resumen_data.append(['RMSE (Error):', f"{self.metricas_invierno['rmse']:.4f} °C"])
                    resumen_data.append(['MAE (Error medio):', f"{self.metricas_invierno['mae']:.4f} °C"])
                    resumen_data.append([''])
                    resumen_data.append(['Importancia de Variables:'])
                    for var, imp in sorted(self.metricas_invierno['importancias'].items(), 
                                         key=lambda x: x[1], reverse=True):
                        resumen_data.append([var, f"{imp*100:.2f}%"])
                
                df_resumen = pd.DataFrame(resumen_data)
                df_resumen.to_excel(writer, sheet_name='Resumen', index=False, header=False)
                
                # HOJA 2: Predicciones Verano (muestra)
                if self.datos_verano is not None and self.modelo_verano is not None:
                    muestra_verano = self.datos_verano.head(1000).copy()
                    predicciones = self.generar_predicciones(
                        self.modelo_verano, muestra_verano, self.vars_entrada_verano
                    )
                    muestra_verano['UCAOT_Predicho'] = predicciones
                    muestra_verano['Error'] = abs(muestra_verano[self.var_salida] - predicciones)
                    muestra_verano['Enfriamiento'] = muestra_verano['UCAIT'] - predicciones
                    muestra_verano.to_excel(writer, sheet_name='Predicciones Verano', index=False)
                
                # HOJA 3: Predicciones Invierno (muestra)
                if self.datos_invierno is not None and self.modelo_invierno is not None:
                    muestra_invierno = self.datos_invierno.head(1000).copy()
                    predicciones = self.generar_predicciones(
                        self.modelo_invierno, muestra_invierno, self.vars_entrada_invierno
                    )
                    muestra_invierno['UCAOT_Predicho'] = predicciones
                    muestra_invierno['Error'] = abs(muestra_invierno[self.var_salida] - predicciones)
                    muestra_invierno['Calentamiento'] = predicciones - muestra_invierno['UCAIT']
                    muestra_invierno.to_excel(writer, sheet_name='Predicciones Invierno', index=False)
                
                # HOJA 4: Análisis Estadístico
                stats_data = []
                stats_data.append(['ANALISIS ESTADISTICO'])
                stats_data.append([''])
                
                if self.datos_verano is not None:
                    stats_data.append(['=== VERANO ==='])
                    stats_data.append(['Variable', 'Media', 'Mediana', 'Desv.Std', 'Min', 'Max'])
                    for col in self.vars_entrada_verano + [self.var_salida]:
                        stats_data.append([
                            col,
                            f"{self.datos_verano[col].mean():.2f}",
                            f"{self.datos_verano[col].median():.2f}",
                            f"{self.datos_verano[col].std():.2f}",
                            f"{self.datos_verano[col].min():.2f}",
                            f"{self.datos_verano[col].max():.2f}"
                        ])
                    stats_data.append([''])
                
                if self.datos_invierno is not None:
                    stats_data.append(['=== INVIERNO ==='])
                    stats_data.append(['Variable', 'Media', 'Mediana', 'Desv.Std', 'Min', 'Max'])
                    for col in self.vars_entrada_invierno + [self.var_salida]:
                        stats_data.append([
                            col,
                            f"{self.datos_invierno[col].mean():.2f}",
                            f"{self.datos_invierno[col].median():.2f}",
                            f"{self.datos_invierno[col].std():.2f}",
                            f"{self.datos_invierno[col].min():.2f}",
                            f"{self.datos_invierno[col].max():.2f}"
                        ])
                
                df_stats = pd.DataFrame(stats_data)
                df_stats.to_excel(writer, sheet_name='Estadisticas', index=False, header=False)
            
            print(f"Excel generado exitosamente: {ruta_salida}")
            print(f"Hojas creadas: Resumen, Predicciones Verano, Predicciones Invierno, Estadisticas")
            
        except Exception as e:
            print(f"ERROR al generar Excel: {e}")
            import traceback
            traceback.print_exc()
    
    def ejecutar(self, carpeta_data='data', carpeta_output='output'):
        """Ejecutar pipeline completo"""
        self.separador("MODELO PREDICTIVO UC52H - INICIO")
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Crear carpeta output si no existe
        if not os.path.exists(carpeta_output):
            os.makedirs(carpeta_output)
        
        # Buscar y cargar datos
        tiene_verano, tiene_invierno = self.buscar_y_cargar_datos(carpeta_data)
        
        if not tiene_verano and not tiene_invierno:
            print("\nERROR: No se encontraron datos de verano ni invierno")
            return False
        
        # Entrenar modelo de verano
        if tiene_verano:
            self.modelo_verano, self.metricas_verano = self.entrenar_modelo(
                self.datos_verano, self.vars_entrada_verano, "VERANO"
            )
        
        # Entrenar modelo de invierno
        if tiene_invierno:
            self.modelo_invierno, self.metricas_invierno = self.entrenar_modelo(
                self.datos_invierno, self.vars_entrada_invierno, "INVIERNO"
            )
        
        # Generar Excel
        ruta_excel = os.path.join(carpeta_output, 'RESULTADOS_UC52H.xlsx')
        self.crear_excel_resultados(ruta_excel)
        
        self.separador("PROCESO COMPLETADO EXITOSAMENTE")
        
        # Resumen final
        print("RESUMEN FINAL:")
        if self.modelo_verano:
            print(f"\nVERANO:")
            print(f"  Precision: {self.metricas_verano['r2']*100:.2f}%")
            print(f"  Error MAE: {self.metricas_verano['mae']:.3f} °C")
        
        if self.modelo_invierno:
            print(f"\nINVIERNO:")
            print(f"  Precision: {self.metricas_invierno['r2']*100:.2f}%")
            print(f"  Error MAE: {self.metricas_invierno['mae']:.3f} °C")
        
        print(f"\nArchivo generado: {ruta_excel}")
        print("\n" + "="*80)
        
        return True


def main():
    """Funcion principal"""
    try:
        modelo = ModeloUC52H()
        exito = modelo.ejecutar()
        
        if exito:
            print("\n\nProceso finalizado correctamente")
            sys.exit(0)
        else:
            print("\n\nProceso no completado")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()