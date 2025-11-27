import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Campañas - INTERIUS",
    page_icon="🎯",
    layout="wide"
)

# Título
st.title("🎯 SIMULADOR DE CAMPAÑAS - INTERIUS")
st.subheader("Predicción de Resultados")

# Sidebar - Inputs del usuario
st.sidebar.header("⚙️ Configuración de Campaña")

# Tipo de Campaña
tipo_campana = st.sidebar.selectbox(
    "Tipo de Campaña:",
    ["Awareness", "Tráfico", "Conversiones"]
)

# Plataforma info
st.sidebar.markdown("**Plataforma:**")
if tipo_campana == "Awareness":
    st.sidebar.markdown("📊 Se compararán Google y Meta")
elif tipo_campana == "Tráfico":
    st.sidebar.markdown("🔴 Solo Meta (sin datos de Google)")
else:  # Conversiones
    st.sidebar.markdown("📊 Se compararán Google y Meta")

# Tipo de presupuesto
tipo_presupuesto = st.sidebar.radio(
    "Tipo de Presupuesto:",
    ["Diario", "Total de Campaña"]
)

if tipo_presupuesto == "Diario":
    presupuesto = st.sidebar.slider(
        "Presupuesto Diario: $",
        min_value=10,
        max_value=10000,
        value=500,
        step=10
    )
else:
    presupuesto = st.sidebar.slider(
        "Presupuesto Total: $",
        min_value=100,
        max_value=100000,
        value=10000,
        step=100
    )

st.sidebar.markdown(f"**Actual:** ${presupuesto:,}")

# Contexto Temporal
st.sidebar.markdown("---")
st.sidebar.header("📅 Contexto Temporal")

mes = st.sidebar.selectbox(
    "Mes:",
    ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", 
     "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"],
    index=9  # Octubre por defecto
)

# Día de la semana (solo para Conversiones)
if tipo_campana == "Conversiones":
    dia_semana = st.sidebar.selectbox(
        "Día de la Semana:",
        ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"],
        index=0
    )

# Tipo de anuncio (varía según campaña)
st.sidebar.markdown("---")
st.sidebar.header("🎬 Configuración de Anuncios")

if tipo_campana == "Awareness":
    ad_type_google = st.sidebar.selectbox(
        "Tipo de Anuncio (Google):",
        ["Skippable in-stream ad", "Responsive video ad"]
    )
elif tipo_campana == "Conversiones":
    ad_type_google = st.sidebar.selectbox(
        "Tipo de Anuncio (Google):",
        ["Responsive search ad", "Responsive video ad", "Call-only ad", 
         "Local ad", "Demand Gen image ad", "Demand Gen video ad"]
    )

# Botón de predicción
st.sidebar.markdown("---")
if st.sidebar.button("🚀 Simular Campaña", type="primary"):
    
    # Convertir mes a número
    meses_dict = {
        "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
        "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
        "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
    }
    mes_num = meses_dict[mes]
    
    # Transformar presupuesto a log
    cost_log = np.log(presupuesto)
    
    # ========================================
    # CAMPAÑAS DE AWARENESS (IMPRESIONES)
    # ========================================
    if tipo_campana == "Awareness":
        # Cargar modelos de impresiones
        try:
            modelo_google_data = joblib.load('modelos/modelo_impresiones_google.pkl')
            modelo_meta_data = joblib.load('modelos/modelo_impresiones_meta.pkl')
            
            model_google = modelo_google_data['model']
            factor_correccion_google = modelo_google_data['factor_correccion']
            
            model_meta = modelo_meta_data['model']
            factor_correccion_meta = modelo_meta_data['factor_correccion']
            
        except Exception as e:
            st.error(f"Error al cargar los modelos: {e}")
            st.stop()
        
        # ===== PREDICCIÓN GOOGLE =====
        input_google = pd.DataFrame({
            'cost_log': [cost_log],
            'ad_type': [ad_type_google],
            'month': [mes_num]
        })
        
        pred_log_google = model_google.predict(input_google)[0]
        impresiones_google = factor_correccion_google * np.exp(pred_log_google) - 1
        
        # ===== PREDICCIÓN META =====
        input_meta = pd.DataFrame({
            'cost_log': [cost_log],
            'month': [mes_num],
            'categoria_campana': ['Awareness']
        })
        
        pred_log_meta = model_meta.predict(input_meta)[0]
        impresiones_meta = factor_correccion_meta * np.exp(pred_log_meta) - 1
        
        # ===== MOSTRAR RESULTADOS =====
        st.markdown("---")
        st.header("📊 Resultados de la Simulación - Impresiones")
        
        # Crear dos columnas
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔵 Google")
            st.metric(
                label="Impresiones Estimadas",
                value=f"{int(impresiones_google):,}"
            )
            cpm_google = (presupuesto / impresiones_google) * 1000
            st.metric(
                label="CPM Estimado",
                value=f"${cpm_google:.2f}"
            )
        
        with col2:
            st.markdown("### 🔴 Meta")
            st.metric(
                label="Impresiones Estimadas",
                value=f"{int(impresiones_meta):,}"
            )
            cpm_meta = (presupuesto / impresiones_meta) * 1000
            st.metric(
                label="CPM Estimado",
                value=f"${cpm_meta:.2f}"
            )
        
        # Comparación
        st.markdown("---")
        st.markdown("### 📈 Comparación")
        
        diferencia = impresiones_google - impresiones_meta
        porcentaje = (diferencia / impresiones_meta) * 100
        
        if diferencia > 0:
            st.success(f"✅ Google genera **{abs(int(diferencia)):,}** impresiones más que Meta ({abs(porcentaje):.1f}% más)")
        else:
            st.success(f"✅ Meta genera **{abs(int(diferencia)):,}** impresiones más que Google ({abs(porcentaje):.1f}% más)")
    
    # ========================================
    # CAMPAÑAS DE TRÁFICO (CLICKS)
    # ========================================
    elif tipo_campana == "Tráfico":
        # Cargar modelo de clicks
        try:
            modelo_clicks_data = joblib.load('modelos/modelo_clicks_meta.pkl')
            
            model_clicks = modelo_clicks_data['model']
            factor_correccion_clicks = modelo_clicks_data['factor_correccion']
            
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
            st.stop()
        
        # ===== PREDICCIÓN META =====
        input_meta_clicks = pd.DataFrame({
            'cost_log': [cost_log],
            'month': [mes_num]
        })
        
        pred_log_clicks = model_clicks.predict(input_meta_clicks)[0]
        clicks_meta = factor_correccion_clicks * np.exp(pred_log_clicks) - 1
        
        # ===== MOSTRAR RESULTADOS =====
        st.markdown("---")
        st.header("📊 Resultados de la Simulación - Clicks")
        
        # Información sobre Google
        st.info("ℹ️ No hay modelo disponible para Google en campañas de Tráfico (datos insuficientes)")
        
        # Resultados de Meta
        st.markdown("### 🔴 Meta")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Clicks Estimados",
                value=f"{int(clicks_meta):,}"
            )
        
        with col2:
            cpc_meta = presupuesto / clicks_meta
            st.metric(
                label="CPC Estimado",
                value=f"${cpc_meta:.2f}"
            )
    
    # ========================================
    # CAMPAÑAS DE CONVERSIONES
    # ========================================
    else:
        # Cargar modelos
        try:
            modelo_conv_google_data = joblib.load('modelos/modelo_conversiones_google.pkl')
            modelo_conv_meta_data = joblib.load('modelos/modelo_conversiones_meta.pkl')
            
            model_conv_google = modelo_conv_google_data['model']
            factor_conv_google = modelo_conv_google_data['factor_correccion']
            
            model_conv_meta = modelo_conv_meta_data['model']
            factor_conv_meta = modelo_conv_meta_data['factor_correccion']
            
        except Exception as e:
            st.error(f"Error al cargar los modelos: {e}")
            st.stop()
        
        # Convertir día de semana a número
        dias_dict = {
            "Lunes": 0, "Martes": 1, "Miércoles": 2, "Jueves": 3,
            "Viernes": 4, "Sábado": 5, "Domingo": 6
        }
        dia_num = dias_dict[dia_semana]
        
        # ===== PREDICCIÓN GOOGLE =====
        input_conv_google = pd.DataFrame({
            'cost_log': [cost_log],
            'ad_type': [ad_type_google],
            'day_of_week': [dia_num],
            'month': [mes_num]
        })
        
        pred_log_conv_google = model_conv_google.predict(input_conv_google)[0]
        conversiones_google = factor_conv_google * np.exp(pred_log_conv_google) - 1
        
        # ===== PREDICCIÓN META =====
        input_conv_meta = pd.DataFrame({
            'cost_log': [cost_log],
            'day_of_week': [dia_num],
            'month': [mes_num]
        })
        
        pred_log_conv_meta = model_conv_meta.predict(input_conv_meta)[0]
        conversiones_meta = factor_conv_meta * np.exp(pred_log_conv_meta) - 1
        
        # ===== MOSTRAR RESULTADOS =====
        st.markdown("---")
        st.header("📊 Resultados de la Simulación - Conversiones")
        
        # Crear dos columnas
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔵 Google")
            st.metric(
                label="Conversiones Estimadas",
                value=f"{int(conversiones_google):,}"
            )
            cpa_google = presupuesto / conversiones_google if conversiones_google > 0 else 0
            st.metric(
                label="CPA Estimado",
                value=f"${cpa_google:.2f}"
            )
        
        with col2:
            st.markdown("### 🔴 Meta")
            st.metric(
                label="Conversiones Estimadas",
                value=f"{int(conversiones_meta):,}"
            )
            cpa_meta = presupuesto / conversiones_meta if conversiones_meta > 0 else 0
            st.metric(
                label="CPA Estimado",
                value=f"${cpa_meta:.2f}"
            )
        
        # Comparación
        st.markdown("---")
        st.markdown("### 📈 Comparación")
        
        diferencia = conversiones_google - conversiones_meta
        porcentaje = (diferencia / conversiones_meta) * 100 if conversiones_meta > 0 else 0
        
        if diferencia > 0:
            st.success(f"✅ Google genera **{abs(int(diferencia)):,}** conversiones más que Meta ({abs(porcentaje):.1f}% más)")
        else:
            st.success(f"✅ Meta genera **{abs(int(diferencia)):,}** conversiones más que Google ({abs(porcentaje):.1f}% más)")

else:
    # Mensaje inicial
    st.info("👈 Configura los parámetros de tu campaña en el panel izquierdo y haz clic en 'Simular Campaña'")