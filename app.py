"""
Aplicación Streamlit para clasificación de severidad de Parkinson
Comparación de Red Neuronal vs Regresión Logística
Versión mejorada: diseño visual, métricas completas, matriz de confusión
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Paleta y tipografía ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan · Parkinson",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0b0f1a;
    --surface:   #111827;
    --surface2:  #1a2236;
    --border:    #1e2d45;
    --accent:    #00d4ff;
    --accent2:   #7c3aed;
    --leve:      #10b981;
    --moderado:  #f59e0b;
    --severo:    #ef4444;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

.stApp { background: var(--bg); color: var(--text); font-family: var(--font-body); }
.main .block-container { padding: 2rem 2.5rem; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a1040 50%, #0d2233 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    font-family: var(--font-mono);
    font-size: 0.72rem; letter-spacing: 0.2em;
    color: var(--accent); text-transform: uppercase; margin-bottom: 0.8rem;
}
.hero h1 {
    font-family: var(--font-head); font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.6rem 0; line-height: 1.1;
}
.hero p { color: var(--muted); font-size: 1rem; max-width: 560px; margin: 0; line-height: 1.6; }

/* Metric cards */
.metric-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.2rem 1.5rem;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-label { font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.15em; color: var(--muted); text-transform: uppercase; margin-bottom: 0.4rem; }
.metric-value { font-family: var(--font-head); font-size: 1.9rem; font-weight: 700; color: var(--accent); line-height: 1; }
.metric-sub { font-size: 0.78rem; color: var(--muted); margin-top: 0.3rem; }

/* Section title */
.section-title { font-family: var(--font-head); font-size: 1.1rem; font-weight: 700; color: var(--text); letter-spacing: 0.03em; margin: 0 0 1.2rem 0; display: flex; align-items: center; gap: 0.6rem; }
.section-title span.dot { width: 8px; height: 8px; background: var(--accent); border-radius: 50%; display: inline-block; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #0099bb) !important;
    color: #000 !important; font-family: var(--font-head) !important;
    font-weight: 700 !important; border: none !important;
    border-radius: 10px !important; padding: 0.6rem 1.6rem !important;
    letter-spacing: 0.04em !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 20px rgba(0,212,255,0.25) !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }

/* Number inputs */
.stNumberInput > div > div > input {
    background: var(--surface2) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text) !important;
    font-family: var(--font-mono) !important; font-size: 0.82rem !important;
}
.stNumberInput > div > div > input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important; }

/* Feature label */
.feat-label { font-family: var(--font-mono); font-size: 0.74rem; color: var(--accent); font-weight: 500; margin-bottom: 2px; }
.feat-range { font-size: 0.68rem; color: var(--muted); margin-bottom: 4px; }

/* Result badge */
.result-badge { display: inline-block; padding: 0.5rem 1.4rem; border-radius: 50px; font-family: var(--font-head); font-size: 1.4rem; font-weight: 800; letter-spacing: 0.05em; margin-top: 0.5rem; }
.badge-leve     { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid #10b981; }
.badge-moderado { background: rgba(245,158,11,0.15);  color: #f59e0b; border: 1px solid #f59e0b; }
.badge-severo   { background: rgba(239,68,68,0.15);   color: #ef4444; border: 1px solid #ef4444; }

/* Batch row */
.batch-row {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 12px; padding: 0.9rem 1.4rem; margin-bottom: 0.6rem;
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;
}

/* Example banner */
.example-banner {
    background: rgba(0,212,255,0.06); border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px; padding: 0.5rem 1rem;
    font-family: var(--font-mono); font-size: 0.78rem; color: #00d4ff; margin-bottom: 1rem;
}

hr { border-color: var(--border) !important; }
[data-testid="stFileUploader"] { background: var(--surface2) !important; border: 1px dashed var(--border) !important; border-radius: 12px !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Constantes ─────────────────────────────────────────────────────────────────
NOMBRES_FEATURES = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "Shimmer:APQ11", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE"
]
RANGOS_FEATURES = [
    (0.000830, 0.099990), (0.000002, 0.000446), (0.000330, 0.057540),
    (0.000430, 0.069560), (0.000980, 0.172630), (0.003060, 0.268630),
    (0.026000, 2.107000), (0.001610, 0.162670), (0.001940, 0.167020),
    (0.002490, 0.275460), (0.004840, 0.488020), (0.000286, 0.748260),
    (1.659000, 37.875000),(0.151020, 0.966080), (0.514040, 0.865600),
    (0.021983, 0.731730)
]
# Valores de ejemplo — paciente moderado típico
EJEMPLO_VALORES = [
    0.00317, 0.000020, 0.001634, 0.002003, 0.004902,
    0.02694, 0.246,   0.01342,  0.01539,  0.02463,
    0.04026, 0.01767, 22.085,   0.414783, 0.815285, 0.266482
]
ETIQUETAS   = {0: 'Leve', 1: 'Moderado', 2: 'Severo'}
COLOR_CLASE = {'Leve': '#10b981', 'Moderado': '#f59e0b', 'Severo': '#ef4444'}
DATA_PATH   = 'parkinsons/telemonitoring/parkinsons_updrs.data'

# ── Datos ──────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    """
    Carga el dataset de forma robusta usando pandas.
    - Descarta filas corruptas o con valores no numéricos automáticamente
    - Reescribe el archivo limpio para evitar errores futuros
    - Siempre usa las primeras 22 columnas (estructura original)
    """
    # pandas tolera filas con distinto nº de columnas y las omite
    df = pd.read_csv(DATA_PATH, header=0, sep=",",
                     usecols=range(22), on_bad_lines="skip",
                     engine="python")
    # Forzar numérico: celdas inválidas → NaN → fila descartada
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    # Reescribir archivo limpio manteniendo el encabezado original
    with open(DATA_PATH, "r") as f:
        header = f.readline().strip()
    with open(DATA_PATH, "w", newline="") as f:
        f.write(header + "\n")
        df.to_csv(f, index=False, header=False, float_format="%.6f")
    return df.values

def crear_variable_objetivo(dataset):
    u = dataset[:, 5]
    s = np.zeros(len(u), dtype=int)
    s[(u > 20) & (u <= 50)] = 1
    s[u > 50] = 2
    return s

def obtener_features(dataset):
    return dataset[:, 6:22]

# ── Entrenamiento ──────────────────────────────────────────────────────────────
@st.cache_resource
def entrenar_modelos():
    dataset = cargar_datos()
    X = obtener_features(dataset)
    y = crear_variable_objetivo(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # Red Neuronal con búsqueda de hiperparámetros
    gs = RandomizedSearchCV(
        MLPClassifier(max_iter=15000, random_state=42),
        {"hidden_layer_sizes": [(8,),(16,),(32,),(64,),(32,16)],
         "activation": ["relu","tanh"], "alpha": [0.001,0.01,0.1]},
        n_iter=12, cv=3, n_jobs=-1, scoring='accuracy', random_state=42
    )
    gs.fit(X_tr, y_train)
    best_mlp  = gs.best_estimator_
    pred_mlp  = best_mlp.predict(X_te)

    # Regresión Logística
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_tr, y_train)
    pred_lr = logreg.predict(X_te)

    def metricas(y_true, y_pred):
        return {
            'accuracy':  accuracy_score(y_true, y_pred),
            'f1_macro':  f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall':    recall_score(y_true, y_pred, average='macro', zero_division=0),
            'cm':        confusion_matrix(y_true, y_pred),
            'report':    classification_report(y_true, y_pred,
                             target_names=['Leve','Moderado','Severo'],
                             output_dict=True, zero_division=0),
        }

    return {
        'mlp':    {'modelo': best_mlp, 'scaler': scaler,
                   'params': gs.best_params_, **metricas(y_test, pred_mlp)},
        'logreg': {'modelo': logreg,   'scaler': scaler,
                   **metricas(y_test, pred_lr)},
    }

def predecir(modelo, scaler, datos):
    return ETIQUETAS[modelo.predict(scaler.transform([datos]))[0]]

# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    labels = ['Leve', 'Moderado', 'Severo']
    colors = ['#10b981', '#f59e0b', '#ef4444']
    mask_diag = np.eye(len(cm), dtype=bool)

    # Fondo off-diagonal
    off = np.where(mask_diag, 0, cm).astype(float)
    vmax = off.max() if off.max() > 0 else 1
    ax.imshow(off, cmap=plt.cm.Blues, aspect='auto', vmin=0, vmax=vmax, alpha=0.4)

    # Celdas diagonales
    for i in range(len(cm)):
        rect = plt.Rectangle([i-0.5, i-0.5], 1, 1, color=colors[i], alpha=0.2, zorder=2)
        ax.add_patch(rect)

    # Valores
    for i in range(len(cm)):
        for j in range(len(cm)):
            col = colors[i] if mask_diag[i, j] else '#e2e8f0'
            wt  = 'bold' if mask_diag[i, j] else 'normal'
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color=col, fontsize=14, fontweight=wt, fontfamily='monospace')

    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, color='#64748b', fontsize=9)
    ax.set_yticklabels(labels, color='#64748b', fontsize=9)
    ax.set_xlabel('Predicción', color='#64748b', fontsize=9, labelpad=8)
    ax.set_ylabel('Real',       color='#64748b', fontsize=9, labelpad=8)
    ax.set_title(title, color='#e2e8f0', fontsize=11, fontweight='bold', pad=14)
    ax.tick_params(colors='#64748b')
    for sp in ax.spines.values(): sp.set_edgecolor('#1e2d45')
    plt.tight_layout()
    return fig

def plot_metrics_bar(m_mlp, m_lr):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')

    names = ['Accuracy', 'F1 Macro', 'Precisión', 'Recall']
    v_mlp = [m_mlp['accuracy'], m_mlp['f1_macro'], m_mlp['precision'], m_mlp['recall']]
    v_lr  = [m_lr['accuracy'],  m_lr['f1_macro'],  m_lr['precision'],  m_lr['recall']]

    x = np.arange(len(names)); w = 0.35
    b1 = ax.bar(x - w/2, v_mlp, w, color='#00d4ff', alpha=0.85, label='Red Neuronal',    zorder=3)
    b2 = ax.bar(x + w/2, v_lr,  w, color='#7c3aed', alpha=0.85, label='Reg. Logística', zorder=3)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom',
                color='#e2e8f0', fontsize=8, fontfamily='monospace')

    ax.set_xticks(x); ax.set_xticklabels(names, color='#64748b', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0,.25,.5,.75,1.0])
    ax.set_yticklabels(['0','.25','.50','.75','1.0'], color='#64748b', fontsize=8)
    ax.tick_params(colors='#64748b')
    ax.yaxis.grid(True, color='#1e2d45', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_edgecolor('#1e2d45')
    ax.legend(fontsize=8, labelcolor='#e2e8f0', facecolor='#1a2236', edgecolor='#1e2d45', loc='lower right')
    ax.set_title('Comparación de métricas', color='#e2e8f0', fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    if 'usar_ejemplo' not in st.session_state:
        st.session_state.usar_ejemplo = False

    with st.spinner('Inicializando modelos…'):
        modelos = entrenar_modelos()

    m_mlp = modelos['mlp']
    m_lr  = modelos['logreg']

    dataset = cargar_datos()
    X = obtener_features(dataset)
    y = crear_variable_objetivo(dataset)

    # ── SIDEBAR ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:1rem 0 0.5rem'>
            <div style='font-size:2.5rem'>🧠</div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#e2e8f0;margin-top:0.3rem'>NeuroScan</div>
            <div style='font-family:"DM Mono",monospace;font-size:0.68rem;color:#00d4ff;letter-spacing:0.2em'>PARKINSON · AI</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("**📊 Dataset**")
        st.write(f"Registros: **{len(X):,}**")
        st.write(f"Features: **{X.shape[1]}**")
        st.markdown("---")

        st.markdown("**Distribución de clases**")
        unique, counts = np.unique(y, return_counts=True)
        for cls, cnt, col in zip(unique, counts, ['#10b981','#f59e0b','#ef4444']):
            pct = cnt/len(y)*100
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                f'<div style="width:10px;height:10px;border-radius:50%;background:{col}"></div>'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:0.8rem">'
                f'{ETIQUETAS[cls]}: <b style="color:{col}">{cnt}</b> '
                f'<span style="color:#64748b">({pct:.0f}%)</span></span></div>',
                unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Escala UPDRS**")
        for lbl, rng, col in [('Leve','0–20','#10b981'),('Moderado','21–50','#f59e0b'),('Severo','51+','#ef4444')]:
            st.markdown(
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.78rem;margin-bottom:4px">'
                f'<span style="color:{col};font-weight:600">{lbl}</span> · {rng}</div>',
                unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Accuracy modelos**")
        for name, acc, col in [('Red Neuronal', m_mlp['accuracy'],'#00d4ff'),('Reg. Logística', m_lr['accuracy'],'#7c3aed')]:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#64748b">{name}</span>'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:0.9rem;color:{col};font-weight:600">{acc*100:.1f}%</span></div>',
                unsafe_allow_html=True)

    # ── HERO ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-tag">🔬 Sistema de análisis clínico · IA</div>
        <h1>Clasificación de Severidad<br>de Parkinson</h1>
        <p>Análisis de biomarcadores acústicos de voz mediante redes neuronales
        y regresión logística para estimación del estadio UPDRS.</p>
    </div>""", unsafe_allow_html=True)

    # ── KPI CARDS ──────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title"><span class="dot"></span>Rendimiento del sistema</p>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col, lbl, val, sub in [
        (c1,"Accuracy · MLP",  f"{m_mlp['accuracy']*100:.1f}%", "red neuronal"),
        (c2,"F1 Macro · MLP",  f"{m_mlp['f1_macro']*100:.1f}%", "macro avg"),
        (c3,"Precisión · MLP", f"{m_mlp['precision']*100:.1f}%","macro avg"),
        (c4,"Accuracy · LR",   f"{m_lr['accuracy']*100:.1f}%",  "reg logística"),
        (c5,"F1 Macro · LR",   f"{m_lr['f1_macro']*100:.1f}%",  "macro avg"),
        (c6,"Recall · LR",     f"{m_lr['recall']*100:.1f}%",    "macro avg"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-sub">{sub}</div></div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MÉTRICAS DETALLADAS ────────────────────────────────────────────────────
    st.markdown('<p class="section-title"><span class="dot"></span>Análisis de métricas</p>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🧠 Red Neuronal", "📈 Regresión Logística"])

    for tab, key, title_cm in [
        (tab1,'mlp',   'Matriz de Confusión · Red Neuronal'),
        (tab2,'logreg','Matriz de Confusión · Reg. Logística'),
    ]:
        m = modelos[key]
        with tab:
            col_cm, col_rep, col_bar = st.columns([1.1, 1.2, 1.4])

            with col_cm:
                fig = plot_confusion_matrix(m['cm'], title_cm)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with col_rep:
                st.markdown("**Reporte por clase**")
                for cls_name in ['Leve','Moderado','Severo']:
                    cr = m['report'][cls_name]
                    col_c = COLOR_CLASE[cls_name]
                    st.markdown(
                        f'<div style="background:#1a2236;border:1px solid #1e2d45;'
                        f'border-left:3px solid {col_c};border-radius:8px;'
                        f'padding:0.7rem 1rem;margin-bottom:0.6rem">'
                        f'<div style="font-family:\'DM Mono\',monospace;font-size:0.8rem;'
                        f'color:{col_c};font-weight:600;margin-bottom:4px">{cls_name}</div>'
                        f'<div style="display:flex;gap:12px;flex-wrap:wrap">'
                        f'<span style="font-size:0.75rem;color:#64748b">P: <b style="color:#e2e8f0">{cr["precision"]:.2f}</b></span>'
                        f'<span style="font-size:0.75rem;color:#64748b">R: <b style="color:#e2e8f0">{cr["recall"]:.2f}</b></span>'
                        f'<span style="font-size:0.75rem;color:#64748b">F1: <b style="color:#e2e8f0">{cr["f1-score"]:.2f}</b></span>'
                        f'<span style="font-size:0.75rem;color:#64748b">N: <b style="color:#e2e8f0">{int(cr["support"])}</b></span>'
                        f'</div></div>',
                        unsafe_allow_html=True)
                ma = m['report']['macro avg']
                st.markdown(
                    f'<div style="background:#111827;border:1px solid #1e2d45;border-radius:8px;padding:0.6rem 1rem">'
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:#64748b">'
                    f'Macro avg · P={ma["precision"]:.2f} · R={ma["recall"]:.2f} · F1={ma["f1-score"]:.2f}</div></div>',
                    unsafe_allow_html=True)

            with col_bar:
                fig2 = plot_metrics_bar(m_mlp, m_lr)
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

    st.markdown("---")

    # ── PREDICCIÓN INDIVIDUAL ─────────────────────────────────────────────────
    st.markdown('<p class="section-title"><span class="dot"></span>Predicción individual</p>', unsafe_allow_html=True)

    col_sel, col_ej = st.columns([3, 1])
    with col_sel:
        modelo_seleccionado = st.radio(
            "Modelo:", ["Red Neuronal", "Regresion Logistica"],
            horizontal=True, label_visibility="collapsed"
        )
    with col_ej:
        if st.button("⚡ Cargar ejemplo", use_container_width=True):
            st.session_state.usar_ejemplo = not st.session_state.usar_ejemplo

    usar_ej = st.session_state.usar_ejemplo
    if usar_ej:
        st.markdown(
            '<div class="example-banner">✓ Valores de ejemplo cargados · paciente con Parkinson moderado</div>',
            unsafe_allow_html=True)

    st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#64748b;margin-bottom:1rem">16 biomarcadores acústicos de voz</p>', unsafe_allow_html=True)

    cols = st.columns(4)
    valores = []
    for i, (feature, (lo, hi)) in enumerate(zip(NOMBRES_FEATURES, RANGOS_FEATURES)):
        default_val = float(EJEMPLO_VALORES[i]) if usar_ej else float((lo + hi) / 2)
        with cols[i % 4]:
            st.markdown(
                f'<div class="feat-label">{i+1}. {feature}</div>'
                f'<div class="feat-range">[{lo:.6f} – {hi:.6f}]</div>',
                unsafe_allow_html=True)
            valor = st.number_input(
                f"feat_{i}", min_value=float(lo), max_value=float(hi),
                value=default_val, format="%.6f",
                label_visibility="collapsed",
                key=f"feat_{i}_{'ej' if usar_ej else 'norm'}"
            )
            valores.append(valor)

    st.markdown("<br>", unsafe_allow_html=True)
    _, col_c, _ = st.columns([1, 1.5, 1])
    with col_c:
        if st.button("🔬 Analizar muestra", type="primary", use_container_width=True):
            key = 'mlp' if modelo_seleccionado == "Red Neuronal" else 'logreg'
            resultado = predecir(modelos[key]['modelo'], modelos[key]['scaler'], valores)
            badge_cls = f"badge-{resultado.lower()}"
            icon = {'Leve':'🟢','Moderado':'🟡','Severo':'🔴'}[resultado]
            modelo_label = "Red Neuronal" if key == "mlp" else "Reg. Logística"
            st.markdown(
                f'<div style="text-align:center;margin-top:1.5rem">'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:#64748b;margin-bottom:0.5rem">Resultado · {modelo_label}</div>'
                f'<div class="result-badge {badge_cls}">{icon} {resultado}</div>'
                f'</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ── PREDICCIÓN POR LOTE ────────────────────────────────────────────────────
    st.markdown('<p class="section-title"><span class="dot"></span>Predicción por lote</p>', unsafe_allow_html=True)

    with st.expander("📋 Formato requerido del CSV"):
        st.markdown("""
        **17 columnas sin encabezado:**

        | # | Nombre | Descripción |
        |---|--------|-------------|
        | 1 | `total_UPDRS` | Puntaje total UPDRS |
        | 2–17 | 16 features | Jitter, Shimmer, NHR, HNR, RPDE, DFA, PPE… |

        - CSV separado por comas, **sin encabezado**
        - `total_UPDRS` genera la etiqueta real automáticamente
        """)

    col_up, col_info = st.columns([2, 1])
    with col_up:
        archivo = st.file_uploader("CSV", type=["csv"], key="batch_upload", label_visibility="collapsed")
    with col_info:
        st.markdown(
            '<div style="background:#1a2236;border:1px solid #1e2d45;border-radius:12px;'
            'padding:1rem;font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#64748b">'
            '17 columnas requeridas<br>'
            '<span style="color:#00d4ff">total_UPDRS</span> + 16 features acústicas</div>',
            unsafe_allow_html=True)

    if archivo is not None:
        try:
            df_nuevo = pd.read_csv(archivo, header=None, delimiter=',')
            if df_nuevo.shape[1] != 17:
                st.error(f"Se esperan 17 columnas. El archivo tiene {df_nuevo.shape[1]}.")
            else:
                datos = df_nuevo.values
                updrs_n = datos[:, 0]
                feats_n = datos[:, 1:]

                sev_nueva = np.zeros(len(updrs_n), dtype=int)
                sev_nueva[(updrs_n > 20) & (updrs_n <= 50)] = 1
                sev_nueva[updrs_n > 50] = 2

                with st.spinner('Generando predicciones…'):
                    key    = 'mlp' if modelo_seleccionado == "Red Neuronal" else 'logreg'
                    modelo = modelos[key]['modelo']
                    scaler = modelos[key]['scaler']
                    preds  = modelo.predict(scaler.transform(feats_n))
                    etiq_pred = [ETIQUETAS[p] for p in preds]
                    etiq_real = [ETIQUETAS[s] for s in sev_nueva]

                # Guardar: el archivo original tiene 22 columnas.
                # El CSV de lote tiene 17 (total_UPDRS + 16 features).
                # Rellenamos las columnas faltantes con 0 para mantener estructura.
                n_orig = 22
                n_actual = datos.shape[1]
                if n_actual < n_orig:
                    padding = np.zeros((len(datos), n_orig - n_actual))
                    datos_padded = np.hstack([datos, padding])
                else:
                    datos_padded = datos[:, :n_orig]
                with open(DATA_PATH, "a") as f:
                    np.savetxt(f, datos_padded, delimiter=",", fmt="%.6f")

                acc_lote = accuracy_score(sev_nueva, preds)
                correc   = sum(p == r for p, r in zip(preds, sev_nueva))

                # KPIs del lote
                ck1, ck2, ck3 = st.columns(3)
                for col_m, lbl, val, sub in [
                    (ck1,"Accuracy lote",  f"{acc_lote*100:.1f}%","vs etiqueta UPDRS"),
                    (ck2,"Correctas",       f"{correc}/{len(preds)}","predicciones"),
                    (ck3,"Registros",        str(len(preds)),"procesados"),
                ]:
                    with col_m:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
                            f'<div class="metric-value" style="font-size:1.5rem">{val}</div>'
                            f'<div class="metric-sub">{sub}</div></div>',
                            unsafe_allow_html=True)

                # ── Matriz de confusión del lote ──────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                col_cm_lote, col_rep_lote = st.columns([1, 1.4])

                with col_cm_lote:
                    clases_presentes = np.unique(np.concatenate([sev_nueva, preds]))
                    if len(clases_presentes) >= 2:
                        cm_lote = confusion_matrix(sev_nueva, preds, labels=[0,1,2])
                        fig_lote = plot_confusion_matrix(cm_lote, "Matriz de Confusión · Lote")
                        st.pyplot(fig_lote, use_container_width=True)
                        plt.close(fig_lote)
                    else:
                        st.info("Se necesitan al menos 2 clases distintas para mostrar la matriz.")

                with col_rep_lote:
                    st.markdown("**Métricas del lote por clase**")
                    labels_lote = sorted(clases_presentes.tolist())
                    nombres_lote = [ETIQUETAS[l] for l in labels_lote]
                    rep_lote = classification_report(
                        sev_nueva, preds, labels=labels_lote,
                        target_names=nombres_lote, output_dict=True, zero_division=0)
                    for cls_name in nombres_lote:
                        cr = rep_lote[cls_name]
                        col_c = COLOR_CLASE[cls_name]
                        st.markdown(
                            f'<div style="background:#1a2236;border:1px solid #1e2d45;'
                            f'border-left:3px solid {col_c};border-radius:8px;'
                            f'padding:0.7rem 1rem;margin-bottom:0.6rem">'
                            f'<div style="font-family:DM Mono,monospace;font-size:0.8rem;'
                            f'color:{col_c};font-weight:600;margin-bottom:4px">{cls_name}</div>'
                            f'<div style="display:flex;gap:12px;flex-wrap:wrap">'
                            f'<span style="font-size:0.75rem;color:#64748b">P: <b style="color:#e2e8f0">{cr["precision"]:.2f}</b></span> '
                            f'<span style="font-size:0.75rem;color:#64748b">R: <b style="color:#e2e8f0">{cr["recall"]:.2f}</b></span> '
                            f'<span style="font-size:0.75rem;color:#64748b">F1: <b style="color:#e2e8f0">{cr["f1-score"]:.2f}</b></span> '
                            f'<span style="font-size:0.75rem;color:#64748b">N: <b style="color:#e2e8f0">{int(cr["support"])}</b></span>'
                            f'</div></div>',
                            unsafe_allow_html=True)
                    ma = rep_lote.get("macro avg", {})
                    if ma:
                        f1_l  = f1_score(sev_nueva, preds, average="macro", zero_division=0)
                        pr_l  = precision_score(sev_nueva, preds, average="macro", zero_division=0)
                        re_l  = recall_score(sev_nueva, preds, average="macro", zero_division=0)
                        st.markdown(
                            f'<div style="background:#111827;border:1px solid #1e2d45;border-radius:8px;padding:0.6rem 1rem;margin-top:0.3rem">'
                            f'<div style="font-family:DM Mono,monospace;font-size:0.75rem;color:#64748b">'
                            f'Macro avg · P={pr_l:.2f} · R={re_l:.2f} · F1={f1_l:.2f}</div></div>',
                            unsafe_allow_html=True)

                st.markdown("<br>**Resultados por registro**")

                for idx in range(len(preds)):
                    real = etiq_real[idx]; pred = etiq_pred[idx]
                    ok   = pred == real
                    col_p = COLOR_CLASE[pred]
                    col_r = COLOR_CLASE[real]
                    icon  = "✓" if ok else "✗"
                    ic_col= "#10b981" if ok else "#ef4444"
                    st.markdown(
                        f'<div class="batch-row">'
                        f'<span style="font-family:\'DM Mono\',monospace;font-size:0.8rem;color:#64748b">#{idx+1}</span>'
                        f'<span style="font-family:\'DM Mono\',monospace;font-size:0.8rem;color:#64748b">UPDRS <b style="color:#e2e8f0">{updrs_n[idx]:.1f}</b></span>'
                        f'<span style="font-size:0.78rem;color:#64748b">Real: <b style="color:{col_r}">{real}</b></span>'
                        f'<span style="font-size:0.78rem;color:#64748b">Pred: <b style="color:{col_p}">{pred}</b></span>'
                        f'<span style="font-family:\'DM Mono\',monospace;font-size:0.85rem;color:{ic_col};font-weight:700">{icon}</span>'
                        f'</div>',
                        unsafe_allow_html=True)

                st.markdown("---")
                st.info(f"✅ {len(preds)} registros añadidos al dataset.")
                cargar_datos.clear()
                entrenar_modelos.clear()

                if st.button("🔄 Re-entrenar con nuevos datos"):
                    st.rerun()

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

if __name__ == "__main__":
    main()
