"""
Script untuk melatih model Random Forest Regression
dan menyimpan model beserta preprocessing objects
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set style untuk visualisasi
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("TRAINING MODEL RANDOM FOREST REGRESSION")
print("Prediksi Harga Tiket Pesawat untuk Pariwisata")
print("=" * 70)

# 1. Load Dataset
print("\n[1/7] Memuat dataset...")
df = pd.read_csv('dataset/data_final.csv')
print(f"Dataset berhasil dimuat: {len(df)} baris, {len(df.columns)} kolom")
print(f"Kolom: {', '.join(df.columns.tolist())}")

# 2. Exploratory Data Analysis
print("\n[2/7] Analisis data...")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nDistribusi kelas:")
print(df['class'].value_counts())
print(f"\nStatistik harga:")
print(f"  Min: Rp {df['price'].min():,.0f}")
print(f"  Max: Rp {df['price'].max():,.0f}")
print(f"  Mean: Rp {df['price'].mean():,.0f}")
print(f"  Median: Rp {df['price'].median():,.0f}")

# 3. Feature Engineering & Preprocessing
print("\n[3/7] Preprocessing data...")

# Pilih fitur yang relevan
features = ['airline', 'source_city', 'destination_city', 'departure_time', 
            'arrival_time', 'stops', 'class', 'duration', 'days_left']
target = 'price'

X = df[features].copy()
y = df[target].copy()

# Identifikasi kolom kategorikal dan numerikal
categorical_cols = ['airline', 'source_city', 'destination_city', 
                    'departure_time', 'arrival_time', 'stops', 'class']
numerical_cols = ['duration', 'days_left']

# Label Encoding untuk kolom kategorikal
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"  Encoded {col}: {len(le.classes_)} unique values")

# Standardization untuk kolom numerikal
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
print(f"  Scaled numerical features: {', '.join(numerical_cols)}")

# 4. Split Data
print("\n[4/7] Membagi data training dan testing...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Training set: {len(X_train)} samples")
print(f"  Testing set: {len(X_test)} samples")

# 5. Training Model
print("\n[5/7] Melatih model Random Forest Regression...")
print("  Hyperparameters:")
print("    - n_estimators: 200")
print("    - max_depth: 20")
print("    - min_samples_split: 5")
print("    - min_samples_leaf: 2")
print("    - random_state: 42")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)
print("  Model training selesai!")

# 6. Evaluasi Model
print("\n[6/7] Evaluasi performa model...")

# Prediksi
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrik Training
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

# Metrik Testing
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("\n  TRAINING SET:")
print(f"    MAE  : Rp {train_mae:,.2f}")
print(f"    RMSE : Rp {train_rmse:,.2f}")
print(f"    R²   : {train_r2:.4f} ({train_r2*100:.2f}%)")

print("\n  TESTING SET:")
print(f"    MAE  : Rp {test_mae:,.2f}")
print(f"    RMSE : Rp {test_rmse:,.2f}")
print(f"    R²   : {test_r2:.4f} ({test_r2*100:.2f}%)")

# Feature Importance
print("\n  FEATURE IMPORTANCE (Top 5):")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(5).iterrows():
    print(f"    {row['feature']:20s}: {row['importance']:.4f}")

# 7. Simpan Model dan Preprocessing Objects
print("\n[7/7] Menyimpan model dan preprocessing objects...")

# Simpan model
with open('model_random_forest.pkl', 'wb') as f:
    pickle.dump(model, f)
print("  ✓ Model disimpan: model_random_forest.pkl")

# Simpan label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("  ✓ Label encoders disimpan: label_encoders.pkl")

# Simpan scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("  ✓ Scaler disimpan: scaler.pkl")

# Simpan feature names dan metadata
metadata = {
    'features': features,
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'target': target,
    'train_r2': train_r2,
    'test_r2': test_r2,
    'test_mae': test_mae,
    'test_rmse': test_rmse,
    'feature_importance': feature_importance.to_dict('records')
}

with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("  ✓ Metadata disimpan: model_metadata.pkl")

# 8. Visualisasi Hasil
print("\n[8/7] Membuat visualisasi...")

# Plot 1: Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Training set
axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=10)
axes[0].plot([y_train.min(), y_train.max()], 
             [y_train.min(), y_train.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Price (Rp)', fontsize=12)
axes[0].set_ylabel('Predicted Price (Rp)', fontsize=12)
axes[0].set_title(f'Training Set\nR² = {train_r2:.4f}', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Testing set
axes[1].scatter(y_test, y_test_pred, alpha=0.5, s=10, color='green')
axes[1].plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Price (Rp)', fontsize=12)
axes[1].set_ylabel('Predicted Price (Rp)', fontsize=12)
axes[1].set_title(f'Testing Set\nR² = {test_r2:.4f}', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('image/model_evaluation.png', dpi=300, bbox_inches='tight')
print("  ✓ Visualisasi disimpan: image/model_evaluation.png")

# Plot 2: Feature Importance
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(top_features['feature'], top_features['importance'], color='steelblue')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 10 Feature Importance - Random Forest', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('image/feature_importance.png', dpi=300, bbox_inches='tight')
print("  ✓ Visualisasi disimpan: image/feature_importance.png")

print("\n" + "=" * 70)
print("TRAINING SELESAI!")
print("=" * 70)
print("\nFile yang dihasilkan:")
print("  1. model_random_forest.pkl")
print("  2. label_encoders.pkl")
print("  3. scaler.pkl")
print("  4. model_metadata.pkl")
print("  5. image/model_evaluation.png")
print("  6. image/feature_importance.png")
print("\nSilakan jalankan: streamlit run app_streamlit.py")
print("=" * 70)
