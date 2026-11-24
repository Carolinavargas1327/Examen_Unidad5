import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configuración general de la página
st.set_page_config(
    page_title="Examen Práctico U5 - Ciencia de Datos (CDI-2413)",
    page_icon="📊",
    layout="wide",
)

SEED = 42

st.title("📊 Examen Práctico Unidad 5: Aplicaciones de Ciencia de Datos")
st.caption("Ingeniería en Ciencia de Datos (CDI-2413) · Modelos Supervisados y No Supervisados")

tab_reg, tab_km = st.tabs(["📈 Ejercicio A: Regresión Lineal", "🎯 Ejercicio B: Clustering K-Means"])

# ==============================================================================
# PESTAÑA 1: REGRESIÓN LINEAL (CALIFORNIA HOUSING)
# ==============================================================================
with tab_reg:
    st.header("Ejercicio A: Predicción de Valor de Vivienda (LinearRegression)")

    @st.cache_data
    def cargar_housing():
        housing = fetch_california_housing(as_frame=True)
        return housing.frame.copy()

    df_housing = cargar_housing()

    col_ctrl, col_main = st.columns([1, 2.3])
    with col_ctrl:
        st.subheader("Parámetros")
        test_size = st.slider("Proporción de prueba (test_size)", 0.10, 0.40, 0.20, 0.05)
        filtrar_tope = st.checkbox("Filtrar valores topados (MedHouseVal >= 5.0)", value=False)
        st.info(f"Semilla fija: `random_state = {SEED}`")

    df_trabajo = df_housing.copy()
    if filtrar_tope:
        df_trabajo = df_trabajo[df_trabajo["MedHouseVal"] < 5.0]

    X = df_trabajo[["MedInc"]]
    y = df_trabajo["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED
    )

    modelo_lr = LinearRegression()
    modelo_lr.fit(X_train, y_train)
    y_pred = modelo_lr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    with col_main:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pendiente (MedInc)", f"{modelo_lr.coef_[0]:.4f}")
        m2.metric("MSE", f"{mse:.4f}")
        m3.metric("RMSE", f"{rmse:.4f}")
        m4.metric("R²", f"{r2:.4f}")

        fig_lr, ax_lr = plt.subplots(figsize=(8, 4.5))
        ax_lr.scatter(X_test["MedInc"], y_test, alpha=0.35, label="Datos reales (prueba)")

        orden = np.argsort(X_test["MedInc"].values)
        ax_lr.plot(
            X_test["MedInc"].values[orden],
            y_pred[orden],
            color="crimson",
            linewidth=2.5,
            label=f"Regresión lineal (y = {modelo_lr.intercept_:.2f} + {modelo_lr.coef_[0]:.2f}x)",
        )
        ax_lr.set_title("Valor Mediano de Vivienda vs. Ingreso Medio")
        ax_lr.set_xlabel("Ingreso medio del bloque (MedInc, decenas de miles USD)")
        ax_lr.set_ylabel("Valor mediano vivienda (MedHouseVal, cientos de miles USD)")
        ax_lr.legend()
        ax_lr.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig_lr)

    with st.expander("Ver inspección de datos y resumen estadístico"):
        st.write(f"Dimensiones utilizadas: {df_trabajo.shape} | Nulos: {int(df_trabajo[['MedInc', 'MedHouseVal']].isna().sum().sum())}")
        st.dataframe(df_trabajo[["MedInc", "MedHouseVal"]].describe().T, use_container_width=True)

# ==============================================================================
# PESTAÑA 2: CLUSTERING K-MEANS (SEGMENTACIÓN DE CLIENTES)
# ==============================================================================
with tab_km:
    st.header("Ejercicio B: Segmentación de Clientes (KMeans)")

    @st.cache_data
    def generar_clientes():
        centros_reales = [
            [25, 20],
            [25, 80],
            [55, 50],
            [85, 20],
            [85, 82],
        ]
        X_raw, _ = make_blobs(
            n_samples=200,
            centers=centros_reales,
            cluster_std=[6.5, 6.5, 7.5, 7.0, 6.5],
            random_state=SEED,
        )
        df = pd.DataFrame(X_raw, columns=["Ingreso_Anual_kUSD", "Puntaje_Gasto_1_100"])
        df["Ingreso_Anual_kUSD"] = df["Ingreso_Anual_kUSD"].clip(15, 137).round(1)
        df["Puntaje_Gasto_1_100"] = df["Puntaje_Gasto_1_100"].clip(1, 99).round(1)
        return df

    df_clientes = generar_clientes()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clientes)

    valores_k = list(range(2, 8))
    inercias = []
    silhouettes = []
    for k in valores_k:
        km_eval = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=SEED)
        labels_eval = km_eval.fit_predict(X_scaled)
        inercias.append(km_eval.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels_eval))

    df_eval = pd.DataFrame(
        {
            "k": valores_k,
            "Inercia": np.round(inercias, 2),
            "Silhouette Score": np.round(silhouettes, 4),
        }
    )

    col_km_ctrl, col_km_main = st.columns([1, 2.3])
    with col_km_ctrl:
        st.subheader("Configuración K-Means")
        k_optimo = st.slider("Número de clústeres (k)", 2, 7, 5)
        st.dataframe(df_eval, hide_index=True, use_container_width=True)

    kmeans_final = KMeans(n_clusters=k_optimo, init="k-means++", n_init=10, random_state=SEED)
    df_seg = df_clientes.copy()
    df_seg["Cluster"] = kmeans_final.fit_predict(X_scaled)

    centroides_orig = scaler.inverse_transform(kmeans_final.cluster_centers_)
    df_centroides = pd.DataFrame(
        centroides_orig, columns=["Centroide_Ingreso_kUSD", "Centroide_Gasto"]
    )
    df_centroides.insert(0, "Cluster", range(k_optimo))
    df_centroides["Tamano"] = df_seg["Cluster"].value_counts().sort_index().values

    with col_km_main:
        fig_km, ax_km = plt.subplots(figsize=(8, 4.5))
        for c in range(k_optimo):
            sub = df_seg[df_seg["Cluster"] == c]
            ax_km.scatter(
                sub["Ingreso_Anual_kUSD"],
                sub["Puntaje_Gasto_1_100"],
                s=55,
                alpha=0.75,
                label=f"Clúster {c} (n={len(sub)})",
            )
        ax_km.scatter(
            centroides_orig[:, 0],
            centroides_orig[:, 1],
            c="black",
            marker="X",
            s=220,
            label="Centroides",
        )
        ax_km.set_title(f"Segmentación de Clientes con K-Means (k = {k_optimo})")
        ax_km.set_xlabel("Ingreso anual (miles de USD)")
        ax_km.set_ylabel("Puntaje de gasto (1 - 100)")
        ax_km.legend()
        ax_km.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig_km)

        st.subheader("Perfil y centroides en escala original")
        st.dataframe(df_centroides.round(2), hide_index=True, use_container_width=True)
