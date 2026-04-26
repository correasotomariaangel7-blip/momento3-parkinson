"""
Aplicación Streamlit para clasificación de severidad de Parkinson
Comparación de Red Neuronal vs Regresión Logística
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
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
    dataset = np.loadtxt('parkinsons\\telemonitoring\\parkinsons_updrs.data', 
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
def entrenar_modelos():
    dataset = cargar_datos()
    X = obtener_features(dataset)
    y = crear_variable_objetivo(dataset)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler_mlp = StandardScaler()
    X_train_scaled = scaler_mlp.fit_transform(X_train)
    X_test_scaled = scaler_mlp.transform(X_test)
    
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu', max_iter=1000, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    pred_mlp = mlp.predict(X_test_scaled)
    acc_mlp = accuracy_score(y_test, pred_mlp)
    
    scaler_logreg = StandardScaler()
    X_train_scaled_lr = scaler_logreg.fit_transform(X_train)
    X_test_scaled_lr = scaler_logreg.transform(X_test)
    
    logreg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    logreg.fit(X_train_scaled_lr, y_train)
    pred_logreg = logreg.predict(X_test_scaled_lr)
    acc_logreg = accuracy_score(y_test, pred_logreg)
    
    return {
        'mlp': {'modelo': mlp, 'scaler': scaler_mlp, 'accuracy': acc_mlp},
        'logreg': {'modelo': logreg, 'scaler': scaler_logreg, 'accuracy': acc_logreg}
    }, acc_mlp, acc_logreg, X, y

def predecir(modelo, scaler, datos):
    datos_scaled = scaler.transform([datos])
    prediccion = modelo.predict(datos_scaled)[0]
    return ETIQUETAS[prediccion]

def main():
    st.title("Clasificacion de Severidad de Parkinson")
    st.markdown("---")
    
    with st.spinner('Entrenando modelos...'):
        modelos, acc_mlp, acc_logreg, X, y = entrenar_modelos()
    
    dataset = cargar_datos()
    y_variable = crear_variable_objetivo(dataset)
    
    with st.sidebar:
        st.header("Informacion del Dataset")
        st.write(f"**Total de registros:** {len(X)}")
        st.write(f"**Features:** {X.shape[1]}")
        
        st.markdown("---")
        st.subheader("Distribucion de clases:")
        unique, counts = np.unique(y_variable, return_counts=True)
        for cls, cnt in zip(unique, counts):
            pct = cnt / len(y_variable) * 100
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
    
    modo = st.radio(
        "Seleccione el modo:",
        ["Individual", "Por Lotes"],
        horizontal=True
    )
    
    if modo == "Individual":
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
    
    else:
        st.markdown("---")
        st.subheader("Matriz de Confusion - Prediccion por Lotes")
        st.write("Usando todo el dataset para prediccion")
        
        col_mlp, col_logreg = st.columns(2)
        
        with col_mlp:
            st.write("### Red Neuronal")
            pred_mlp = modelos['mlp']['modelo'].predict(
                modelos['mlp']['scaler'].transform(X)
            )
            cm_mlp = confusion_matrix(y, pred_mlp)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm_mlp, display_labels=['Leve', 'Moderado', 'Severo']).plot(ax=ax)
            st.pyplot(fig)
            
            report_mlp = classification_report(y, pred_mlp, target_names=['Leve', 'Moderado', 'Severo'], output_dict=True)
            st.write(f"**Accuracy:** {report_mlp['accuracy']*100:.2f}%")
            st.write(f"**Precision (Macro):** {report_mlp['macro avg']['precision']*100:.2f}%")
            st.write(f"**Recall (Macro):** {report_mlp['macro avg']['recall']*100:.2f}%")
            st.write(f"**F1-Score (Macro):** {report_mlp['macro avg']['f1-score']*100:.2f}%")
        
        with col_logreg:
            st.write("### Regresion Logistica")
            pred_logreg = modelos['logreg']['modelo'].predict(
                modelos['logreg']['scaler'].transform(X)
            )
            cm_logreg = confusion_matrix(y, pred_logreg)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm_logreg, display_labels=['Leve', 'Moderado', 'Severo']).plot(ax=ax)
            st.pyplot(fig)
            
            report_logreg = classification_report(y, pred_logreg, target_names=['Leve', 'Moderado', 'Severo'], output_dict=True)
            st.write(f"**Accuracy:** {report_logreg['accuracy']*100:.2f}%")
            st.write(f"**Precision (Macro):** {report_logreg['macro avg']['precision']*100:.2f}%")
            st.write(f"**Recall (Macro):** {report_logreg['macro avg']['recall']*100:.2f}%")
            st.write(f"**F1-Score (Macro):** {report_logreg['macro avg']['f1-score']*100:.2f}%")
        
        st.markdown("---")
        st.subheader("Tabla Comparativa de Metricas")
        
        datos_tabla = pd.DataFrame({
            'Modelo': ['Red Neuronal', 'Regresion Logistica'],
            'Accuracy': [f"{report_mlp['accuracy']*100:.2f}%", f"{report_logreg['accuracy']*100:.2f}%"],
            'Precision': [f"{report_mlp['macro avg']['precision']*100:.2f}%", f"{report_logreg['macro avg']['precision']*100:.2f}%"],
            'Recall': [f"{report_mlp['macro avg']['recall']*100:.2f}%", f"{report_logreg['macro avg']['recall']*100:.2f}%"],
            'F1-Score': [f"{report_mlp['macro avg']['f1-score']*100:.2f}%", f"{report_logreg['macro avg']['f1-score']*100:.2f}%"]
        })
        st.table(datos_tabla)

if __name__ == "__main__":
    main()