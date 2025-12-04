"""
===============================================================================
SIMULADOR UC52H
Predicción de Temperatura de Salida con Modelo Entrenado
===============================================================================

Este script utiliza el modelo ya entrenado para predecir temperaturas de
salida (UCAOT) a partir de datos de entrada en archivos CSV.

===============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
import sys
import os

warnings.filterwarnings('ignore')


class SimuladorUC52H:
    """Simulador de temperatura de salida usando modelo entrenado"""
    
    def __init__(self):
        """Inicializar simulador"""
        self.modelo_verano = None
        self.modelo_invierno = None
        self.vars_entrada_verano = ['UCWIT', 'UCAIT', 'UCWF', 'UCAF', 'UCFDP']
        self.vars_entrada_invierno = ['UCAIT', 'UCAF', 'UCHV', 'UCHC', 'UCFDP']
        
    def separador(self, texto=""):
        """Imprimir separador"""
        print("\n" + "="*80)
        if texto:
            print(f"{texto:^80}")
        print("="*80 + "\n")
    
    def subseparador(self, texto):
        """Imprimir subseparador"""
        print("\n" + "-"*80)
        print(f"{texto:^80}")
        print("-"*80 + "\n")
    
    def entrenar_modelos_rapido(self, carpeta_data='data'):
        """Entrenar modelos con datos existentes (versión rápida)"""
        self.subseparador("ENTRENANDO MODELOS")
        
        print("Buscando archivos de entrenamiento...")
        
        # Buscar archivos
        archivos_verano = []
        archivos_invierno = []
        
        if os.path.exists(carpeta_data):
            import glob
            todos_csv = glob.glob(os.path.join(carpeta_data, "*.csv"))
            
            for archivo in todos_csv:
                nombre = os.path.basename(archivo).lower()
                if 'sum' in nombre or any(f'DS0{i}' in archivo for i in [1,2,6,7,8,9]) or 'DS10' in archivo or 'DS11' in archivo:
                    if 'test' not in nombre and 'complete' not in nombre and 'simulacion' not in nombre:
                        archivos_verano.append(archivo)
                elif 'win' in nombre or any(f'DS0{i}' in archivo for i in [3,4,5]) or any(f'DS{i}' in archivo for i in [12,13,14,15]):
                    if 'complete' not in nombre and 'simulacion' not in nombre:
                        archivos_invierno.append(archivo)
        
        print(f"Archivos verano encontrados: {len(archivos_verano)}")
        print(f"Archivos invierno encontrados: {len(archivos_invierno)}")
        
        # Entrenar verano
        if archivos_verano:
            print("\nEntrenando modelo VERANO...")
            datos_verano = []
            for archivo in archivos_verano:
                try:
                    df = pd.read_csv(archivo, sep=';', encoding='utf-8')
                    if len(df.columns) < 10:
                        df = pd.read_csv(archivo, sep=',', encoding='utf-8')
                    cols_necesarias = self.vars_entrada_verano + ['UCAOT']
                    if all(col in df.columns for col in cols_necesarias):
                        datos_verano.append(df[cols_necesarias].dropna())
                except:
                    pass
            
            if datos_verano:
                df_verano = pd.concat(datos_verano, ignore_index=True)
                X = df_verano[self.vars_entrada_verano]
                y = df_verano['UCAOT']
                
                self.modelo_verano = RandomForestRegressor(
                    n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
                )
                self.modelo_verano.fit(X, y)
                print(f"Modelo verano entrenado con {len(X)} registros")
        
        # Entrenar invierno
        if archivos_invierno:
            print("\nEntrenando modelo INVIERNO...")
            datos_invierno = []
            for archivo in archivos_invierno:
                try:
                    df = pd.read_csv(archivo, sep=';', encoding='utf-8-sig')
                    if len(df.columns) < 10:
                        df = pd.read_csv(archivo, sep=',', encoding='utf-8-sig')
                    cols_necesarias = self.vars_entrada_invierno + ['UCAOT']
                    if all(col in df.columns for col in cols_necesarias):
                        datos_invierno.append(df[cols_necesarias].dropna())
                except:
                    pass
            
            if datos_invierno:
                df_invierno = pd.concat(datos_invierno, ignore_index=True)
                X = df_invierno[self.vars_entrada_invierno]
                y = df_invierno['UCAOT']
                
                self.modelo_invierno = RandomForestRegressor(
                    n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
                )
                self.modelo_invierno.fit(X, y)
                print(f"Modelo invierno entrenado con {len(X)} registros")
        
        print("\nModelos listos para simulación")
    
    def simular(self, archivo_entrada, modo='auto', archivo_salida=None):
        """
        Simular temperaturas de salida
        
        Args:
            archivo_entrada: CSV con datos de entrada
            modo: 'verano', 'invierno' o 'auto' (detecta automáticamente)
            archivo_salida: Archivo donde guardar resultados (opcional)
        """
        self.subseparador(f"SIMULANDO: {os.path.basename(archivo_entrada)}")
        
        # Leer archivo
        try:
            df = pd.read_csv(archivo_entrada)
            print(f"Archivo cargado: {len(df)} registros")
        except Exception as e:
            print(f"ERROR al leer archivo: {e}")
            return None
        
        # Detectar modo
        if modo == 'auto':
            if 'UCWIT' in df.columns:
                modo = 'verano'
            elif 'UCHV' in df.columns:
                modo = 'invierno'
            else:
                print("ERROR: No se puede detectar el modo automáticamente")
                return None
        
        print(f"Modo detectado: {modo.upper()}")
        
        # Verificar que existe el modelo
        if modo == 'verano' and self.modelo_verano is None:
            print("ERROR: Modelo de verano no está entrenado")
            return None
        elif modo == 'invierno' and self.modelo_invierno is None:
            print("ERROR: Modelo de invierno no está entrenado")
            return None
        
        # Seleccionar variables y modelo
        if modo == 'verano':
            vars_entrada = self.vars_entrada_verano
            modelo = self.modelo_verano
        else:
            vars_entrada = self.vars_entrada_invierno
            modelo = self.modelo_invierno
        
        # Verificar columnas
        columnas_faltantes = [col for col in vars_entrada if col not in df.columns]
        if columnas_faltantes:
            print(f"ERROR: Faltan columnas: {columnas_faltantes}")
            return None
        
        # Preparar datos
        X = df[vars_entrada]
        
        # Predecir
        print("\nGenerando predicciones...")
        predicciones = modelo.predict(X)
        
        # Añadir predicciones al dataframe
        df['UCAOT_Predicho'] = predicciones
        
        # Calcular diferencia térmica
        if modo == 'verano':
            df['Enfriamiento'] = df['UCAIT'] - df['UCAOT_Predicho']
        else:
            df['Calentamiento'] = df['UCAOT_Predicho'] - df['UCAIT']
        
        # Mostrar estadísticas
        print("\nESTADISTICAS DE PREDICCIONES:")
        print(f"  Temperatura predicha media: {predicciones.mean():.2f} °C")
        print(f"  Temperatura predicha min: {predicciones.min():.2f} °C")
        print(f"  Temperatura predicha max: {predicciones.max():.2f} °C")
        print(f"  Desviación estándar: {predicciones.std():.2f} °C")
        
        if modo == 'verano':
            print(f"  Enfriamiento promedio: {df['Enfriamiento'].mean():.2f} °C")
        else:
            print(f"  Calentamiento promedio: {df['Calentamiento'].mean():.2f} °C")
        
        # Mostrar primeras predicciones
        print("\nPRIMERAS 5 PREDICCIONES:")
        if modo == 'verano':
            cols_mostrar = vars_entrada + ['UCAOT_Predicho', 'Enfriamiento']
        else:
            cols_mostrar = vars_entrada + ['UCAOT_Predicho', 'Calentamiento']
        
        print(df[cols_mostrar].head().to_string(index=False))
        
        # Guardar resultados
        if archivo_salida:
            df.to_csv(archivo_salida, index=False)
            print(f"\nResultados guardados en: {archivo_salida}")
        
        return df


def main():
    """Función principal"""
    
    print("""
===============================================================================
                          SIMULADOR UC52H - INICIO                          
===============================================================================
    
Este script utiliza el modelo entrenado para predecir temperaturas de salida
a partir de datos de entrada.
    """)
    
    # Crear simulador
    simulador = SimuladorUC52H()
    
    # Entrenar modelos con datos existentes
    print("\nPaso 1: Entrenando modelos con datos históricos...")
    simulador.entrenar_modelos_rapido(carpeta_data='data')
    
    # Simular con archivos de ejemplo
    print("\n\nPaso 2: Ejecutando simulaciones...")
    
    # Simulación VERANO
    if os.path.exists('DATOS_SIMULACION_VERANO.csv'):
        resultado_verano = simulador.simular(
            archivo_entrada='DATOS_SIMULACION_VERANO.csv',
            modo='verano',
            archivo_salida='output/PREDICCIONES_VERANO_SIMULADAS.csv'
        )
    
    # Simulación INVIERNO
    if os.path.exists('DATOS_SIMULACION_INVIERNO.csv'):
        resultado_invierno = simulador.simular(
            archivo_entrada='DATOS_SIMULACION_INVIERNO.csv',
            modo='invierno',
            archivo_salida='output/PREDICCIONES_INVIERNO_SIMULADAS.csv'
        )
    
    print("""

===============================================================================
                        SIMULACION COMPLETADA                        
===============================================================================

Archivos generados:
  - output/PREDICCIONES_VERANO_SIMULADAS.csv
  - output/PREDICCIONES_INVIERNO_SIMULADAS.csv

Estos archivos contienen las predicciones de temperatura de salida (UCAOT)
para las condiciones de entrada especificadas.

===============================================================================
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)