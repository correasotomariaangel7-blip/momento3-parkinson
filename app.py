"""
Aplicación Streamlit para clasificación de severidad de Parkinson
Comparación de Red Neuronal vs Regresión Logística
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Clasificación de Parkinson",
    page_icon="🏥",
    layout="wide"
)

NOMBRES_FEATURES = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "PPE"
]

RANGOS_FEATURES = [
    (0.000830, 0.099990),
    (0.000002, 0.000446),
    (0.000330, 0.057540),
    (0.000430, 0.069560),
    (0.000980, 0.172630),
    (0.003060, 0.268630),
    (0.026000, 2.107000),
    (0.001610, 0.162670),
    (0.001940, 0.167020),
    (0.002490, 0.275460),
    (0.004840, 0.488020),
    (0.000286, 0.748260),
    (1.659000, 37.875000),
    (0.151020, 0.966080),
    (0.514040, 0.865600),
    (0.021983, 0.731730)
]

ETIQUETAS = {0: 'Leve', 1: 'Moderado', 2: 'Severo'}

@st.cache_data
def cargar_datos():
    dataset = np.loadtxt('parkinsons/telemonitoring/parkinsons_updrs.data', 
                         delimiter=',', skiprows=1)
    return dataset

def crear_variable_objetivo(dataset):
    total_UPDRS = dataset[:, 5]
    severidad = np.zeros(len(total_UPDRS), dtype=int)
    severidad[(total_UPDRS > 20) & (total_UPDRS <= 50)] = 1
    severidad[total_UPDRS > 50] = 2
    return severidad

def obtener_features(dataset):
    return dataset[:, 6:22]

@st.cache_resource
def entrenar_modelos(): # Entrenamos ambos modelos y devolvemos sus precisiones
    dataset = cargar_datos()
    # aqui separamos los datos de entrenaminto con los de prueba 
    X = obtener_features(dataset) 
    y = crear_variable_objetivo(dataset)
    # Dividimos el dataset utilizando un 70% para entrenamiento y un 30% para prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # es para escalar los datos para que el modelo funcione mejor
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Crear el modelo de red neuronal (MLPClassifier)
    mlp = MLPClassifier (max_iter=15000)
    # Definir los hiperparámetros a probar en la búsqueda aleatoria, para encontrar la mejor configuración del modelo
    param_grid = {"hidden_layer_sizes": [(8,), (16,), (32,), (64,)], "activation" : ["relu", "tanh"],"alpha": [0.001, 0.01, 0.1]}
    # Realizar la búsqueda aleatoria con validación cruzada, utilizando todos los núcleos disponibles y evaluando con la métrica de precisión
    grid_search = RandomizedSearchCV(mlp, param_grid, cv=3, n_jobs= -1, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    # Obtener el mejor modelo y sus hiperparámetros
    best_model = grid_search.best_estimator_
    grid_search.best_params_
    # aqui se entrena el modelo con los mejores hiperparámetros encontrados
    mlp.fit(X_train_scaled, y_train)    
    pred_mlp = best_model.predict(X_test_scaled)
    acc_mlp = accuracy_score(y_test, pred_mlp)
    
    #Regresión logistica, se entrena el modelo con los datos escalados y se evalua su precisión con los datos de prueba, para comparar con el modelo de red neuronal
    logreg = LogisticRegression() 
    logreg.fit(X_train_scaled, y_train)
    # aqui se calcula la tasa de efectividad del modelo
    pred_logreg = logreg.predict(X_test_scaled)
    acc_logreg = accuracy_score(y_test, pred_logreg)
    
    return {
        'mlp': {'modelo': mlp, 'scaler': scaler, 'accuracy': acc_mlp},
        'logreg': {'modelo': logreg, 'scaler': scaler, 'accuracy': acc_logreg}
    }, acc_mlp, acc_logreg

def predecir(modelo, scaler, datos):
    datos_scaled = scaler.transform([datos])
    prediccion = modelo.predict(datos_scaled)[0]
    return ETIQUETAS[prediccion]

def main():
    st.title("Clasificacion de Severidad de Parkinson")
    st.markdown("---")
    
    with st.spinner('Entrenando modelos...'):
        modelos, acc_mlp, acc_logreg = entrenar_modelos()
    
    dataset = cargar_datos()
    X = obtener_features(dataset)
    y = crear_variable_objetivo(dataset)
    
    with st.sidebar:
        st.header("Informacion del Dataset")
        st.write(f"**Total de registros:** {len(X)}")
        st.write(f"**Features:** {X.shape[1]}")
        
        st.markdown("---")
        st.subheader("Distribucion de clases:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, cnt in zip(unique, counts):
            pct = cnt / len(y) * 100
            st.write(f"**{ETIQUETAS[cls]}:** {cnt} ({pct:.1f}%)")
        
        st.markdown("---")
        st.subheader("Categorias UPDRS:")
        st.write("**Leve:** 0-20")  
        st.write("**Moderado:** 21-50")
        st.write("**Severo:** 51+")
        
        st.markdown("---")
        st.subheader("Precision de modelos:")
        st.write(f"**Red Neuronal:** {acc_mlp*100:.2f}%")
        st.write(f"**Regresion Logistica:** {acc_logreg*100:.2f}%")
    
    st.subheader("Seleccione el modelo:")
    modelo_seleccionado = st.radio(
        "Modelo:",
        ["Red Neuronal", "Regresion Logistica"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.subheader("Ingrese los 16 valores de caracteristicas de voz:")
    
    cols = st.columns(4)
    valores = []
    
    for i, (feature, (min_val, max_val)) in enumerate(zip(NOMBRES_FEATURES, RANGOS_FEATURES)):
        with cols[i % 4]:
            st.markdown(f"**{i+1}. {feature}**")
            st.caption(f"Rango: [{min_val:.6f} - {max_val:.6f}]")
            valor = st.number_input(
                f"feat_{i}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float((min_val + max_val) / 2),
                format="%.6f",
                label_visibility="collapsed"
            )
            valores.append(valor)
    
    st.markdown("---")
    col_centrada = st.columns([1, 2, 1])
    with col_centrada[1]:
        if st.button("Predecir", type="primary", use_container_width=True):
            key = 'mlp' if modelo_seleccionado == "Red Neuronal" else 'logreg'
            resultado = predecir(
                modelos[key]['modelo'],
                modelos[key]['scaler'],
                valores
            )
            
            st.markdown("---")
            st.subheader("Resultado:")
            if resultado == "Leve":
                st.success(f"## {resultado}")
            elif resultado == "Moderado":
                st.warning(f"## {resultado}")
            else:
                st.error(f"## {resultado}")

    st.markdown("---")
    st.subheader("Predicción por Lote")
    
    with st.expander("Instrucciones del formato requerido"):
        st.markdown("""
        **El archivo CSV debe tener las siguientes 17 columnas:**
        
        | Columna | Nombre | Descripción |
        |---------|--------|-------------|
        | 1 | `total_UPDRS` | Puntaje total UPDRS del paciente (numérico) |
        | 2-17 | 16 features de voz | `Jitter(%)`, `Jitter(Abs)`, `Jitter:RAP`, `Jitter:PPQ5`, `Jitter:DDP`, `Shimmer`, `Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `Shimmer:APQ11`, `Shimmer:DDA`, `NHR`, `HNR`, `RPDE`, `DFA`, `PPE` |
        
        - El archivo **debe ser CSV** separado por comas
        - **Sin encabezados** (el archivo debe comenzar directamente con los datos)
        - El `total_UPDRS` se usa para crear automáticamente la etiqueta de severidad
        - Las 16 features de voz se usan para la predicción del modelo
        """)
    
    archivo_subido = st.file_uploader("Sube tu archivo CSV", type=["csv"], key="batch_upload")
    
    if archivo_subido is not None:
        try:
            df_nuevo = pd.read_csv(archivo_subido, header=None, delimiter=',')
            
            if df_nuevo.shape[1] != 17:
                st.error(f"El archivo debe tener **17 columnas**. Actualmente tiene {df_nuevo.shape[1]} columnas.")
            else:
                datos_nuevo = df_nuevo.values
                total_updrs_nuevo = datos_nuevo[:, 0]
                features_nuevo = datos_nuevo[:, 1:]
                severidad_nueva = np.zeros(len(total_updrs_nuevo), dtype=int)
                severidad_nueva[(total_updrs_nuevo > 20) & (total_updrs_nuevo <= 50)] = 1
                severidad_nueva[total_updrs_nuevo > 50] = 2
                
                dataset_original = cargar_datos()
                X_original = obtener_features(dataset_original)
                y_original = crear_variable_objetivo(dataset_original)
                
                with st.spinner('Generando predicciones...'):
                    key = 'mlp' if modelo_seleccionado == "Red Neuronal" else 'logreg'
                    modelo = modelos[key]['modelo']
                    scaler = modelos[key]['scaler']
                    
                    features_nuevo_scaled = scaler.transform(features_nuevo)
                    predicciones = modelo.predict(features_nuevo_scaled)
                    etiquetas_pred = np.array([ETIQUETAS[p] for p in predicciones])
                
                dataset_combinado = np.vstack((dataset_original, datos_nuevo))
                with open('parkinsons/telemonitoring/parkinsons_updrs.data', 'a') as f:
                    np.savetxt(f, datos_nuevo, delimiter=',')
                
                st.success(f"Se cargaron **{len(features_nuevo)}** registros exitosamente.")
                
                st.subheader("Resultados de la predicción:")
                
                df_resultado = pd.DataFrame()
                df_resultado['total_UPDRS'] = total_updrs_nuevo
                df_resultado['Severidad_Real'] = [ETIQUETAS[s] for s in severidad_nueva]
                df_resultado['Prediccion'] = etiquetas_pred
                
                for idx, row in df_resultado.iterrows():
                    col1, col2, col3 = st.columns([2, 2, 2])
                    with col1:
                        st.metric(f"Registro {idx+1}", f"UPDRS: {row['total_UPDRS']:.2f}")
                    with col2:
                        st.write(f"Severidad: **{row['Severidad_Real']}**")
                    with col3:
                        pred = row['Prediccion']
                        if pred == "Leve":
                            st.success(f"Predicción: **{pred}**")
                        elif pred == "Moderado":
                            st.warning(f"Predicción: **{pred}**")
                        else:
                            st.error(f"Predicción: **{pred}**")
                    
                    if idx < len(df_resultado) - 1:
                        st.divider()
                
                st.markdown("---")
                st.info(f"Los {len(features_nuevo)} nuevos registros fueron añadidos al dataset original.")
                
                cargar_datos.clear()
                if st.button("Recargar la app para re-entrenar con los nuevos datos"):
                    st.rerun()
                
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

if __name__ == "__main__":
    main()