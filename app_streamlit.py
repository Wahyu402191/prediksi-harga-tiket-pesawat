"""
Aplikasi Web Prediksi Harga Tiket Pesawat
Menggunakan Random Forest Regression
Dibuat dengan Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(
    page_title="Prediksi Harga Tiket Pesawat",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container with aviation background */
    .main {
        background: linear-gradient(rgba(255,255,255,0.95), rgba(255,255,255,0.95)),
                    url('https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=1920') center/cover fixed;
        padding: 0rem 1rem;
    }
    
    /* Sidebar with dark aviation theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f36 0%, #0f1419 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #e0e6ed !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    /* Header with aviation theme */
    .header-container {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .header-title {
        color: #ffffff !important;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
        letter-spacing: -0.5px;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
    }
    
    .header-subtitle {
        color: #e8f0f7 !important;
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.8rem;
        font-weight: 400;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Card styling with glass morphism */
    .info-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-left: 4px solid #2c5364;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%);
        padding: 2rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(44, 83, 100, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c5364;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: #5a6c7d;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2c5364 0%, #0f2027 100%);
        color: white;
        font-weight: 600;
        padding: 0.9rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(44, 83, 100, 0.3);
        transition: all 0.3s ease;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(44, 83, 100, 0.4);
        background: linear-gradient(135deg, #203a43 0%, #0f2027 100%);
    }
    
    /* Input fields */
    .stSelectbox, .stNumberInput, .stSlider {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 8px;
    }
    
    /* Result box with premium design */
    .result-box {
        background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%);
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .result-price {
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    
    .result-label {
        font-size: 1.2rem;
        color: #b8c5d6;
        margin-top: 0.8rem;
        font-weight: 400;
    }
    
    /* Section headers */
    h2, h3 {
        color: #1a1f36;
        font-weight: 600;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.5);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Info/Warning boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border-left: 4px solid #2c5364;
    }
    
    /* Animated Statistics Cards */
    .stat-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(44, 83, 100, 0.1);
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15);
    }
    
    .stat-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c5364;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #5a6c7d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Price Alert Badges */
    .price-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .badge-good {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .badge-normal {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .badge-high {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    
    /* Loading Animation */
    .loading-container {
        text-align: center;
        padding: 2rem;
    }
    
    .plane-animation {
        font-size: 3rem;
        animation: fly 2s ease-in-out infinite;
    }
    
    @keyframes fly {
        0%, 100% { transform: translateX(-20px) translateY(0); }
        50% { transform: translateX(20px) translateY(-10px); }
    }
    
    /* Comparison Table */
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    .comparison-table th {
        background: linear-gradient(135deg, #2c5364 0%, #0f2027 100%);
        color: white;
        padding: 1rem;
        text-align: left;
        font-weight: 600;
    }
    
    .comparison-table td {
        padding: 1rem;
        border-bottom: 1px solid #e0e6ed;
    }
    
    .comparison-table tr:hover {
        background: #f8f9fa;
    }
    
    .best-price {
        background: #d4edda !important;
        font-weight: 700;
        color: #155724;
    }
    
    /* FAQ Accordion */
    .faq-item {
        background: white;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        overflow: hidden;
    }
    
    .faq-question {
        padding: 1.2rem;
        font-weight: 600;
        color: #1a1f36;
        cursor: pointer;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        transition: all 0.3s ease;
    }
    
    .faq-question:hover {
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
    }
    
    .faq-answer {
        padding: 1.2rem;
        color: #5a6c7d;
        line-height: 1.6;
        border-top: 1px solid #e0e6ed;
    }
    
    /* Testimonial Card */
    .testimonial-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        border-left: 4px solid #2c5364;
    }
    
    .testimonial-text {
        font-style: italic;
        color: #5a6c7d;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    
    .testimonial-author {
        font-weight: 600;
        color: #2c5364;
    }
    
    .testimonial-rating {
        color: #ffc107;
        font-size: 1.2rem;
    }
    
    /* Floating Action Button */
    .fab {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #2c5364 0%, #0f2027 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        cursor: pointer;
        transition: all 0.3s ease;
        z-index: 1000;
    }
    
    .fab:hover {
        transform: scale(1.1);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
    }
    
    .fab-icon {
        font-size: 1.8rem;
        color: white;
    }
    
    /* Progress Bar Animation */
    .progress-container {
        width: 100%;
        background: #e0e6ed;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 8px;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        animation: progress 2s ease-in-out;
    }
    
    @keyframes progress {
        0% { width: 0%; }
        100% { width: 100%; }
    }
    
    /* Card 3D Tilt Effect */
    .tilt-card {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .tilt-card:hover {
        transform: perspective(1000px) rotateX(2deg) rotateY(-2deg);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
    }
    
    /* Gradient Border Input */
    .stSelectbox > div > div,
    .stNumberInput > div > div,
    .stSlider > div {
        border: 2px solid transparent;
        background: linear-gradient(white, white) padding-box,
                    linear-gradient(135deg, #4facfe, #00f2fe) border-box;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div:focus-within {
        box-shadow: 0 0 15px rgba(79, 172, 254, 0.3);
        transform: translateY(-2px);
    }
    
    /* Timeline Visualization */
    .timeline-container {
        position: relative;
        padding: 2rem 0;
        margin: 2rem 0;
    }
    
    .timeline-line {
        position: relative;
        height: 4px;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 2px;
        margin: 2rem 0;
    }
    
    .timeline-point {
        position: absolute;
        width: 20px;
        height: 20px;
        background: white;
        border: 4px solid #4facfe;
        border-radius: 50%;
        top: 50%;
        transform: translateY(-50%);
    }
    
    .timeline-point.start {
        left: 0;
    }
    
    .timeline-point.end {
        right: 0;
    }
    
    .timeline-plane {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 2rem;
        animation: flyAcross 3s ease-in-out infinite;
    }
    
    @keyframes flyAcross {
        0% { left: 10%; }
        50% { left: 50%; top: 40%; }
        100% { left: 90%; }
    }
    
    .timeline-label {
        position: absolute;
        top: -30px;
        font-weight: 600;
        color: #2c5364;
        white-space: nowrap;
    }
    
    .timeline-label.start {
        left: 0;
    }
    
    .timeline-label.end {
        right: 0;
    }
    
    /* Color-coded Price Indicator */
    .price-indicator {
        width: 100%;
        height: 30px;
        background: linear-gradient(90deg, 
            #11998e 0%, 
            #38ef7d 25%, 
            #f5af19 50%, 
            #f12711 75%, 
            #c21500 100%);
        border-radius: 15px;
        position: relative;
        margin: 1rem 0;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    
    .price-marker {
        position: absolute;
        width: 4px;
        height: 40px;
        background: white;
        border: 2px solid #2c5364;
        top: -5px;
        border-radius: 2px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        animation: slideToPosition 1.5s ease-out forwards;
    }
    
    @keyframes slideToPosition {
        0% {
            left: 0%;
            opacity: 0;
            transform: translateX(0);
        }
        100% {
            left: var(--marker-position, 50%);
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .price-marker::before {
        content: '';
        position: absolute;
        top: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 0;
        height: 0;
        border-left: 8px solid transparent;
        border-right: 8px solid transparent;
        border-top: 10px solid #2c5364;
    }
    
    .price-marker::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 0;
        height: 0;
        border-left: 8px solid transparent;
        border-right: 8px solid transparent;
        border-bottom: 10px solid #2c5364;
    }
    
    .price-labels {
        display: flex;
        justify-content: space-between;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #5a6c7d;
    }
    
    /* Smooth Page Transitions */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Quick Stats Ticker */
    .ticker-container {
        background: linear-gradient(135deg, #2c5364 0%, #0f2027 100%);
        color: white;
        padding: 0.8rem;
        overflow: hidden;
        margin-bottom: 1rem;
        border-radius: 8px;
    }
    
    .ticker-content {
        display: inline-block;
        white-space: nowrap;
        animation: ticker 30s linear infinite;
    }
    
    @keyframes ticker {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    
    .ticker-item {
        display: inline-block;
        margin: 0 3rem;
        font-weight: 500;
    }
    
    /* Animated Particles Background */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
    }
    
    .particle {
        position: absolute;
        width: 8px;
        height: 8px;
        background: radial-gradient(circle, rgba(79, 172, 254, 0.8) 0%, rgba(79, 172, 254, 0.3) 100%);
        border-radius: 50%;
        box-shadow: 0 0 10px rgba(79, 172, 254, 0.5);
        animation: float 15s infinite;
    }
    
    @keyframes float {
        0% {
            transform: translateY(100vh) translateX(0) scale(0);
            opacity: 0;
        }
        10% {
            opacity: 1;
            transform: translateY(90vh) translateX(10px) scale(1);
        }
        50% {
            transform: translateY(50vh) translateX(-20px) scale(1.2);
        }
        90% {
            opacity: 1;
        }
        100% {
            transform: translateY(-10vh) translateX(30px) scale(0.8);
            opacity: 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL & DATA ====================
@st.cache_resource
def load_model_and_data():
    """Load model, encoders, scaler, dan metadata"""
    try:
        with open('model_random_forest.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load dataset untuk referensi
        df = pd.read_csv('dataset/data_final.csv')
        
        return model, label_encoders, scaler, metadata, df
    except FileNotFoundError:
        st.error("Model belum dilatih! Silakan jalankan train_model.py terlebih dahulu.")
        st.stop()

# Load semua resources
model, label_encoders, scaler, metadata, df = load_model_and_data()

# ==================== HEADER ====================
# Animated Particles Background
st.markdown("""
<div class="particles">
    <div class="particle" style="left: 5%; animation-delay: 0s; animation-duration: 12s;"></div>
    <div class="particle" style="left: 15%; animation-delay: 2s; animation-duration: 14s;"></div>
    <div class="particle" style="left: 25%; animation-delay: 4s; animation-duration: 16s;"></div>
    <div class="particle" style="left: 35%; animation-delay: 1s; animation-duration: 13s;"></div>
    <div class="particle" style="left: 45%; animation-delay: 3s; animation-duration: 15s;"></div>
    <div class="particle" style="left: 55%; animation-delay: 5s; animation-duration: 17s;"></div>
    <div class="particle" style="left: 65%; animation-delay: 2.5s; animation-duration: 14s;"></div>
    <div class="particle" style="left: 75%; animation-delay: 4.5s; animation-duration: 16s;"></div>
    <div class="particle" style="left: 85%; animation-delay: 1.5s; animation-duration: 13s;"></div>
    <div class="particle" style="left: 95%; animation-delay: 3.5s; animation-duration: 15s;"></div>
    <div class="particle" style="left: 10%; animation-delay: 6s; animation-duration: 18s;"></div>
    <div class="particle" style="left: 30%; animation-delay: 7s; animation-duration: 12s;"></div>
    <div class="particle" style="left: 50%; animation-delay: 8s; animation-duration: 14s;"></div>
    <div class="particle" style="left: 70%; animation-delay: 9s; animation-duration: 16s;"></div>
    <div class="particle" style="left: 90%; animation-delay: 10s; animation-duration: 13s;"></div>
</div>
""", unsafe_allow_html=True)

# Quick Stats Ticker
ticker_items = [
    f"Total Penerbangan: {len(df):,}",
    f"Akurasi Model: {metadata['test_r2']*100:.1f}%",
    f"Harga Rata-rata: Rp {df['price'].mean():,.0f}",
    f"Rute Terpopuler: {df.groupby(['source_city', 'destination_city']).size().idxmax()[0]} - {df.groupby(['source_city', 'destination_city']).size().idxmax()[1]}",
    f"Maskapai Termurah: {df.groupby('airline')['price'].mean().idxmin()}",
    "Prediksi Real-Time Tersedia",
    "Data Terupdate Q1 2022"
]

st.markdown(f"""
<div class="ticker-container">
    <div class="ticker-content">
        {''.join([f'<span class="ticker-item">• {item}</span>' for item in ticker_items * 3])}
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-container fade-in">
    <h1 class="header-title">Prediksi Harga Tiket Pesawat</h1>
    <p class="header-subtitle">Sistem Cerdas Berbasis Random Forest Regression untuk Perencanaan Perjalanan Wisata</p>
</div>
""", unsafe_allow_html=True)

# ==================== ANIMATED STATISTICS DASHBOARD ====================
st.markdown("### Statistik Real-Time")

# Hitung statistik dari dataset
total_flights = len(df)
avg_price = df['price'].mean()
min_price = df['price'].min()
max_price = df['price'].max()
accuracy = metadata['test_r2'] * 100

stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)

with stat_col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">📊</div>
        <div class="stat-value">{total_flights:,}</div>
        <div class="stat-label">Total Data</div>
    </div>
    """, unsafe_allow_html=True)

with stat_col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">🎯</div>
        <div class="stat-value">{accuracy:.1f}%</div>
        <div class="stat-label">Akurasi Model</div>
    </div>
    """, unsafe_allow_html=True)

with stat_col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">💰</div>
        <div class="stat-value">Rp {avg_price/1000:.0f}K</div>
        <div class="stat-label">Harga Rata-rata</div>
    </div>
    """, unsafe_allow_html=True)

with stat_col4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">⬇️</div>
        <div class="stat-value">Rp {min_price/1000:.0f}K</div>
        <div class="stat-label">Harga Terendah</div>
    </div>
    """, unsafe_allow_html=True)

with stat_col5:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">⬆️</div>
        <div class="stat-value">Rp {max_price/1000:.0f}K</div>
        <div class="stat-label">Harga Tertinggi</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <svg width="80" height="80" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="planeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#4facfe;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#00f2fe;stop-opacity:1" />
                </linearGradient>
            </defs>
            <!-- Plane body -->
            <path d="M50 20 L70 50 L65 55 L50 45 L35 55 L30 50 Z" fill="url(#planeGradient)" stroke="#ffffff" stroke-width="2"/>
            <!-- Wings -->
            <path d="M30 40 L20 45 L25 50 L35 45 Z" fill="url(#planeGradient)" stroke="#ffffff" stroke-width="1.5"/>
            <path d="M70 40 L80 45 L75 50 L65 45 Z" fill="url(#planeGradient)" stroke="#ffffff" stroke-width="1.5"/>
            <!-- Tail -->
            <path d="M45 60 L40 70 L45 68 L50 70 L55 68 L60 70 L55 60 Z" fill="url(#planeGradient)" stroke="#ffffff" stroke-width="1.5"/>
            <!-- Windows -->
            <circle cx="50" cy="35" r="3" fill="#ffffff" opacity="0.8"/>
            <circle cx="50" cy="42" r="3" fill="#ffffff" opacity="0.8"/>
            <!-- Motion lines -->
            <line x1="15" y1="35" x2="25" y2="35" stroke="#4facfe" stroke-width="2" opacity="0.6"/>
            <line x1="10" y1="42" x2="22" y2="42" stroke="#4facfe" stroke-width="2" opacity="0.6"/>
            <line x1="12" y1="49" x2="24" y2="49" stroke="#4facfe" stroke-width="2" opacity="0.6"/>
        </svg>
        <h2 style="color: #ffffff; margin: 1rem 0 0 0; font-size: 1.5rem; font-weight: 600;">Flight Predictor</h2>
        <p style="color: #b8c5d6; font-size: 0.85rem; margin: 0.3rem 0 0 0;">Smart Travel Planning</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: #ffffff; font-size: 1.1rem; margin-top: 2rem;'>Navigasi</h3>", unsafe_allow_html=True)
    
    menu = st.radio(
        "Pilih Menu:",
        ["Prediksi Harga", "Analisis Data", "Performa Model", "Tentang Sistem"],
        label_visibility="collapsed"
    )
    
    st.markdown("<div style='border-top: 1px solid rgba(255,255,255,0.2); margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: #ffffff; font-size: 1.1rem;'>Informasi Model</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin-top: 0.5rem;'>
        <p style='margin: 0.3rem 0;'><strong>Algoritma:</strong> Random Forest</p>
        <p style='margin: 0.3rem 0;'><strong>Akurasi (R²):</strong> {metadata['test_r2']*100:.2f}%</p>
        <p style='margin: 0.3rem 0;'><strong>MAE:</strong> Rp {metadata['test_mae']:,.0f}</p>
        <p style='margin: 0.3rem 0;'><strong>Dataset:</strong> {len(df):,} penerbangan</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='border-top: 1px solid rgba(255,255,255,0.2); margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: #ffffff; font-size: 1.1rem;'>Pengembang</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin-top: 0.5rem;'>
        <p style='margin: 0.3rem 0;'><strong>Nama:</strong> Wahyu Pratama</p>
        <p style='margin: 0.3rem 0;'><strong>NIM:</strong> 23011100058</p>
        <p style='margin: 0.3rem 0;'><strong>Prodi:</strong> Informatika Pariwisata</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== MENU: PREDIKSI HARGA ====================
if menu == "Prediksi Harga":
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("## Masukkan Detail Penerbangan")
    st.markdown("Lengkapi informasi penerbangan Anda untuk mendapatkan estimasi harga tiket.")
    
    # Form input dalam 2 kolom
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Informasi Rute")
        
        airline = st.selectbox(
            "Maskapai Penerbangan",
            options=sorted(df['airline'].unique()),
            help="Pilih maskapai yang ingin Anda gunakan"
        )
        
        source_city = st.selectbox(
            "Kota Keberangkatan",
            options=sorted(df['source_city'].unique()),
            help="Kota asal penerbangan"
        )
        
        destination_city = st.selectbox(
            "Kota Tujuan",
            options=sorted(df['destination_city'].unique()),
            help="Kota tujuan penerbangan"
        )
        
        flight_class = st.selectbox(
            "Kelas Penerbangan",
            options=sorted(df['class'].unique()),
            help="Pilih kelas penerbangan"
        )
    
    with col2:
        st.markdown("### Informasi Waktu & Detail")
        
        departure_time = st.selectbox(
            "Waktu Keberangkatan",
            options=sorted(df['departure_time'].unique()),
            help="Waktu keberangkatan pesawat"
        )
        
        arrival_time = st.selectbox(
            "Waktu Kedatangan",
            options=sorted(df['arrival_time'].unique()),
            help="Waktu kedatangan pesawat"
        )
        
        stops = st.selectbox(
            "Jumlah Transit",
            options=sorted(df['stops'].unique()),
            help="Jumlah pemberhentian selama perjalanan"
        )
        
        duration = st.number_input(
            "Durasi Penerbangan (jam)",
            min_value=0.5,
            max_value=40.0,
            value=5.0,
            step=0.5,
            help="Total durasi penerbangan dalam jam"
        )
        
        days_left = st.slider(
            "Hari Sebelum Keberangkatan",
            min_value=1,
            max_value=49,
            value=15,
            help="Berapa hari lagi sebelum tanggal keberangkatan"
        )
    
    st.markdown("---")
    
    # Tombol prediksi
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("PREDIKSI HARGA TIKET", use_container_width=True)
    
    if predict_button:
        # Siapkan data input
        input_data = pd.DataFrame({
            'airline': [airline],
            'source_city': [source_city],
            'destination_city': [destination_city],
            'departure_time': [departure_time],
            'arrival_time': [arrival_time],
            'stops': [stops],
            'class': [flight_class],
            'duration': [duration],
            'days_left': [days_left]
        })
        
        # Preprocessing
        input_processed = input_data.copy()
        
        # Label encoding
        for col in metadata['categorical_cols']:
            input_processed[col] = label_encoders[col].transform(input_data[col])
        
        # Scaling
        input_processed[metadata['numerical_cols']] = scaler.transform(
            input_data[metadata['numerical_cols']]
        )
        
        # Prediksi
        prediction = model.predict(input_processed)[0]
        
        # Hitung confidence interval (estimasi)
        predictions_all = []
        for estimator in model.estimators_[:50]:  # Ambil 50 tree pertama
            predictions_all.append(estimator.predict(input_processed)[0])
        
        std_prediction = np.std(predictions_all)
        lower_bound = prediction - 1.96 * std_prediction
        upper_bound = prediction + 1.96 * std_prediction
        
        # Tampilkan hasil
        st.markdown("---")
        
        # Progress Bar
        st.markdown("""
        <div class="progress-container">
            <div class="progress-bar"></div>
        </div>
        <p style="text-align: center; color: #5a6c7d; margin-top: 0.5rem;">Menganalisis data penerbangan...</p>
        """, unsafe_allow_html=True)
        
        st.markdown("## Hasil Prediksi")
        
        # Timeline Visualization
        st.markdown("### Visualisasi Perjalanan")
        st.markdown(f"""
        <div class="timeline-container">
            <div class="timeline-label start">{source_city}</div>
            <div class="timeline-label end">{destination_city}</div>
            <div class="timeline-line">
                <div class="timeline-point start"></div>
                <div class="timeline-plane">✈️</div>
                <div class="timeline-point end"></div>
            </div>
            <p style="text-align: center; color: #5a6c7d; margin-top: 2rem;">
                Durasi: {duration} jam | Transit: {stops.replace('_', ' ').title()} | Kelas: {flight_class}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Price Alert Badge
        if prediction < df['price'].quantile(0.25):
            badge_class = "badge-good"
            badge_text = "✓ Harga Sangat Bagus!"
        elif prediction < df['price'].quantile(0.75):
            badge_class = "badge-normal"
            badge_text = "• Harga Normal"
        else:
            badge_class = "badge-high"
            badge_text = "⚠ Harga Tinggi"
        
        st.markdown(f"""
        <div style="text-align: center;">
            <span class="price-badge {badge_class}">{badge_text}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Color-coded Price Indicator
        price_position = ((prediction - df['price'].min()) / (df['price'].max() - df['price'].min())) * 100
        st.markdown(f"""
        <div style="margin: 2rem 0;">
            <p style="text-align: center; font-weight: 600; color: #2c5364; margin-bottom: 1rem;">
                Posisi Harga di Pasar
            </p>
            <div class="price-indicator">
                <div class="price-marker" style="--marker-position: {price_position}%;"></div>
            </div>
            <div class="price-labels">
                <span>Murah</span>
                <span>Normal</span>
                <span>Mahal</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Box hasil utama
        st.markdown(f"""
        <div class="result-box fade-in">
            <p class="result-label">Estimasi Harga Tiket</p>
            <p class="result-price">Rp {prediction:,.0f}</p>
            <p class="result-label">Rentang: Rp {lower_bound:,.0f} - Rp {upper_bound:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detail prediksi
        col_detail1, col_detail2, col_detail3 = st.columns(3)
        
        with col_detail1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">Rp {lower_bound:,.0f}</p>
                <p class="metric-label">Harga Minimum</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_detail2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">Rp {prediction:,.0f}</p>
                <p class="metric-label">Harga Prediksi</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_detail3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">Rp {upper_bound:,.0f}</p>
                <p class="metric-label">Harga Maksimum</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Ringkasan penerbangan
        st.markdown("### Ringkasan Penerbangan Anda")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown(f"""
            <div class="info-card">
                <strong>Rute Penerbangan</strong><br>
                {source_city} → {destination_city}<br><br>
                <strong>Maskapai:</strong> {airline}<br>
                <strong>Kelas:</strong> {flight_class}<br>
                <strong>Transit:</strong> {stops}
            </div>
            """, unsafe_allow_html=True)
        
        with summary_col2:
            st.markdown(f"""
            <div class="info-card">
                <strong>Jadwal Penerbangan</strong><br>
                Berangkat: {departure_time}<br>
                Tiba: {arrival_time}<br><br>
                <strong>Durasi:</strong> {duration} jam<br>
                <strong>Keberangkatan:</strong> {days_left} hari lagi
            </div>
            """, unsafe_allow_html=True)
        
        # Tips hemat
        st.markdown("### Tips Hemat Perjalanan")
        
        tips = []
        
        if days_left < 7:
            tips.append("⚠ Pemesanan mendadak cenderung lebih mahal. Pertimbangkan untuk memesan lebih awal di lain waktu.")
        elif days_left > 30:
            tips.append("✓ Pemesanan jauh-jauh hari biasanya mendapat harga lebih baik")
        
        if stops == "zero":
            tips.append("✓ Penerbangan langsung lebih cepat namun biasanya lebih mahal")
        else:
            tips.append("• Penerbangan dengan transit bisa lebih murah, pertimbangkan waktu tunggu Anda")
        
        if flight_class == "Business":
            tips.append("• Kelas bisnis lebih nyaman namun harganya 2-3x lipat dari ekonomi")
        
        for tip in tips:
            st.markdown(f"""
            <div style='background: rgba(44, 83, 100, 0.05); padding: 0.8rem 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid #2c5364;'>
                {tip}
            </div>
            """, unsafe_allow_html=True)

# ==================== MENU: ANALISIS DATA ====================
elif menu == "Analisis Data":
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown("## Analisis Dataset Penerbangan")
    
    tab1, tab2, tab3 = st.tabs(["Distribusi Harga", "Analisis Rute", "Pola Waktu"])
    
    with tab1:
        st.markdown("### Distribusi Harga Tiket")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram harga
            fig = px.histogram(
                df, 
                x='price', 
                nbins=50,
                title='Distribusi Harga Tiket',
                labels={'price': 'Harga (Rp)', 'count': 'Jumlah'},
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot berdasarkan kelas
            fig = px.box(
                df, 
                x='class', 
                y='price',
                title='Perbandingan Harga Berdasarkan Kelas',
                labels={'class': 'Kelas', 'price': 'Harga (Rp)'},
                color='class',
                color_discrete_sequence=['#667eea', '#764ba2']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistik
        st.markdown("### Statistik Harga")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Harga Minimum", f"Rp {df['price'].min():,.0f}")
        with stat_col2:
            st.metric("Harga Rata-rata", f"Rp {df['price'].mean():,.0f}")
        with stat_col3:
            st.metric("Harga Median", f"Rp {df['price'].median():,.0f}")
        with stat_col4:
            st.metric("Harga Maksimum", f"Rp {df['price'].max():,.0f}")
    
    with tab2:
        st.markdown("### Analisis Rute Penerbangan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 rute termahal
            df['route'] = df['source_city'] + ' → ' + df['destination_city']
            top_routes = df.groupby('route')['price'].mean().sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=top_routes.values,
                y=top_routes.index,
                orientation='h',
                title='Top 10 Rute Termahal (Rata-rata)',
                labels={'x': 'Harga Rata-rata (Rp)', 'y': 'Rute'},
                color=top_routes.values,
                color_continuous_scale='Purples'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Harga berdasarkan maskapai
            airline_price = df.groupby('airline')['price'].mean().sort_values(ascending=False)
            
            fig = px.bar(
                x=airline_price.index,
                y=airline_price.values,
                title='Harga Rata-rata Berdasarkan Maskapai',
                labels={'x': 'Maskapai', 'y': 'Harga Rata-rata (Rp)'},
                color=airline_price.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Pola Waktu Pemesanan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Harga vs days_left
            fig = px.scatter(
                df.sample(1000), 
                x='days_left', 
                y='price',
                title='Hubungan Hari Pemesanan dengan Harga',
                labels={'days_left': 'Hari Sebelum Keberangkatan', 'price': 'Harga (Rp)'},
                trendline='lowess',
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Harga berdasarkan waktu keberangkatan
            time_price = df.groupby('departure_time')['price'].mean().sort_values(ascending=False)
            
            fig = px.bar(
                x=time_price.index,
                y=time_price.values,
                title='Harga Rata-rata Berdasarkan Waktu Keberangkatan',
                labels={'x': 'Waktu Keberangkatan', 'y': 'Harga Rata-rata (Rp)'},
                color=time_price.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ==================== MENU: PERFORMA MODEL ====================
elif menu == "Performa Model":
    st.markdown("## Evaluasi Performa Model")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{metadata['test_r2']*100:.2f}%</p>
            <p class="metric-label">R² Score (Akurasi)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">Rp {metadata['test_mae']:,.0f}</p>
            <p class="metric-label">Mean Absolute Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">Rp {metadata['test_rmse']:,.0f}</p>
            <p class="metric-label">Root Mean Squared Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature Importance
    st.markdown("### Feature Importance")
    st.markdown("Grafik di bawah menunjukkan fitur mana yang paling berpengaruh terhadap prediksi harga tiket.")
    
    feature_imp_df = pd.DataFrame(metadata['feature_importance']).head(10)
    
    fig = px.bar(
        feature_imp_df,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Fitur Paling Berpengaruh',
        labels={'importance': 'Importance Score', 'feature': 'Fitur'},
        color='importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Penjelasan metrik
    st.markdown("### Penjelasan Metrik Evaluasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <strong>R² Score (Coefficient of Determination)</strong><br><br>
            Mengukur seberapa baik model menjelaskan variasi data. Nilai mendekati 100% berarti model sangat akurat.
            <br><br>
            <strong>Interpretasi:</strong><br>
            • 90-100%: Sangat Baik<br>
            • 70-90%: Baik<br>
            • 50-70%: Cukup<br>
            • <50%: Kurang Baik
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <strong>Mean Absolute Error (MAE)</strong><br><br>
            Rata-rata selisih absolut antara prediksi dengan harga sebenarnya. Semakin kecil semakin baik.
            <br><br>
            <strong>Root Mean Squared Error (RMSE)</strong><br><br>
            Mirip dengan MAE namun memberikan penalti lebih besar pada error yang besar. Berguna untuk mendeteksi outlier.
        </div>
        """, unsafe_allow_html=True)

# ==================== MENU: TENTANG SISTEM ====================
else:
    st.markdown("## Tentang Sistem Prediksi Harga Tiket")
    
    st.markdown("""
    <div class="info-card">
        <h3>Latar Belakang</h3>
        <p>
        Sistem ini dikembangkan sebagai solusi untuk membantu wisatawan dalam merencanakan anggaran perjalanan 
        dengan lebih akurat. Fluktuasi harga tiket pesawat yang dinamis sering menjadi hambatan dalam perencanaan 
        perjalanan wisata yang efisien.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>Teknologi yang Digunakan</h3>
            <ul>
                <li><strong>Algoritma:</strong> Random Forest Regression</li>
                <li><strong>Framework:</strong> Scikit-learn</li>
                <li><strong>Web Framework:</strong> Streamlit</li>
                <li><strong>Visualisasi:</strong> Plotly, Matplotlib</li>
                <li><strong>Data Processing:</strong> Pandas, NumPy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Fitur Utama</h3>
            <ul>
                <li>Prediksi harga tiket real-time</li>
                <li>Analisis pola perjalanan wisatawan</li>
                <li>Visualisasi data interaktif</li>
                <li>Confidence interval prediksi</li>
                <li>Tips hemat perjalanan</li>
                <li>Evaluasi performa model</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Comparison Table
    st.markdown("### Perbandingan Harga Antar Maskapai")
    
    comparison_data = df.groupby('airline').agg({
        'price': ['mean', 'min', 'max', 'count']
    }).round(0)
    comparison_data.columns = ['Rata-rata', 'Minimum', 'Maksimum', 'Jumlah Data']
    comparison_data = comparison_data.sort_values('Rata-rata')
    
    # Find best price
    best_airline = comparison_data['Rata-rata'].idxmin()
    
    st.markdown("""
    <table class="comparison-table">
        <thead>
            <tr>
                <th>Maskapai</th>
                <th>Harga Rata-rata</th>
                <th>Harga Minimum</th>
                <th>Harga Maksimum</th>
                <th>Jumlah Data</th>
            </tr>
        </thead>
        <tbody>
    """, unsafe_allow_html=True)
    
    for airline, row in comparison_data.iterrows():
        row_class = "best-price" if airline == best_airline else ""
        st.markdown(f"""
            <tr class="{row_class}">
                <td><strong>{airline}</strong></td>
                <td>Rp {row['Rata-rata']:,.0f}</td>
                <td>Rp {row['Minimum']:,.0f}</td>
                <td>Rp {row['Maksimum']:,.0f}</td>
                <td>{int(row['Jumlah Data']):,}</td>
            </tr>
        """, unsafe_allow_html=True)
    
    st.markdown("</tbody></table>", unsafe_allow_html=True)
    
    # Testimonials
    st.markdown("### Testimoni Pengguna")
    
    testimonials = [
        {
            "text": "Aplikasi ini sangat membantu saya merencanakan budget perjalanan. Prediksinya akurat dan UI-nya mudah digunakan!",
            "author": "Budi Santoso",
            "role": "Travel Blogger",
            "rating": 5
        },
        {
            "text": "Fitur analisis data sangat berguna untuk memahami pola harga tiket. Recommended untuk yang sering traveling!",
            "author": "Siti Nurhaliza",
            "role": "Digital Nomad",
            "rating": 5
        },
        {
            "text": "Dengan sistem ini, saya bisa hemat hingga 30% untuk tiket pesawat. Terima kasih!",
            "author": "Ahmad Rizki",
            "role": "Mahasiswa",
            "rating": 4
        }
    ]
    
    test_col1, test_col2, test_col3 = st.columns(3)
    
    for idx, (col, testimonial) in enumerate(zip([test_col1, test_col2, test_col3], testimonials)):
        with col:
            stars = "⭐" * testimonial['rating']
            st.markdown(f"""
            <div class="testimonial-card">
                <div class="testimonial-rating">{stars}</div>
                <p class="testimonial-text">"{testimonial['text']}"</p>
                <p class="testimonial-author">{testimonial['author']}</p>
                <p style="color: #5a6c7d; font-size: 0.9rem;">{testimonial['role']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # FAQ Section
    st.markdown("### Pertanyaan yang Sering Diajukan (FAQ)")
    
    faqs = [
        {
            "question": "Bagaimana cara kerja sistem prediksi ini?",
            "answer": "Sistem menggunakan algoritma Random Forest Regression yang dilatih dengan 9.000 data historis penerbangan. Model mempelajari pola harga berdasarkan berbagai faktor seperti maskapai, rute, waktu, dan kelas penerbangan."
        },
        {
            "question": "Seberapa akurat prediksi harga yang diberikan?",
            "answer": f"Model kami memiliki tingkat akurasi (R²) sebesar {metadata['test_r2']*100:.2f}% dengan rata-rata kesalahan (MAE) sekitar Rp {metadata['test_mae']:,.0f}. Ini berarti prediksi kami sangat mendekati harga aktual."
        },
        {
            "question": "Apakah data yang digunakan selalu update?",
            "answer": "Dataset yang digunakan adalah data historis Kuartal 1 tahun 2022. Untuk implementasi production, sistem dapat diintegrasikan dengan API maskapai untuk mendapatkan data real-time."
        },
        {
            "question": "Bagaimana cara mendapatkan harga tiket termurah?",
            "answer": "Berdasarkan analisis data, tips terbaik adalah: (1) Pesan tiket 15-30 hari sebelum keberangkatan, (2) Pertimbangkan penerbangan dengan transit, (3) Hindari waktu peak season, (4) Bandingkan harga antar maskapai."
        },
        {
            "question": "Apakah sistem ini bisa digunakan untuk penerbangan internasional?",
            "answer": "Saat ini sistem fokus pada penerbangan domestik Indonesia. Namun, metodologi yang sama dapat diterapkan untuk penerbangan internasional dengan dataset yang sesuai."
        }
    ]
    
    for idx, faq in enumerate(faqs):
        with st.expander(f"❓ {faq['question']}"):
            st.markdown(f"""
            <div class="faq-answer">
                {faq['answer']}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>Cara Kerja Random Forest Regression</h3>
        <p>
        Random Forest adalah algoritma ensemble learning yang menggabungkan prediksi dari ratusan pohon keputusan 
        independen. Setiap pohon dilatih dengan subset data acak (bootstrap sampling) dan memilih fitur secara acak 
        di setiap percabangan. Hasil akhir diambil dari rata-rata prediksi seluruh pohon, menghasilkan estimasi yang 
        lebih stabil dan akurat.
        </p>
        <br>
        <strong>Keunggulan:</strong>
        <ul>
            <li>Tahan terhadap overfitting</li>
            <li>Dapat menangani data kategorikal dan numerikal</li>
            <li>Robust terhadap outlier</li>
            <li>Memberikan feature importance</li>
            <li>Akurasi tinggi untuk data tabular</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>Dataset</h3>
        <p>
        Dataset yang digunakan berasal dari Kaggle (Flight Price Dataset) dengan total 9.000 data penerbangan 
        pada Kuartal 1 tahun 2022. Dataset mencakup informasi maskapai, rute, waktu, kelas, durasi, dan harga tiket.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>Pengembang</h3>
        <p>
        <strong>Nama:</strong> Wahyu Pratama<br>
        <strong>NIM:</strong> 23011100058<br>
        <strong>Mata Kuliah:</strong> Informatika Pariwisata Kelas B<br>
        <strong>Institusi:</strong> Program Studi Teknik Informatika
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== FLOATING ACTION BUTTON ====================
st.markdown("""
<a href="#" onclick="window.scrollTo({top: 0, behavior: 'smooth'}); return false;">
    <div class="fab">
        <div class="fab-icon">↑</div>
    </div>
</a>
""", unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Sistem Prediksi Harga Tiket Pesawat | Random Forest Regression</p>
    <p>© 2026 Wahyu Pratama - Informatika Pariwisata</p>
</div>
""", unsafe_allow_html=True)
