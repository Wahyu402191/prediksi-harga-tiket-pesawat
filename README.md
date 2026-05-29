# Aplikasi Prediksi Harga Tiket Pesawat

Sistem prediksi harga tiket pesawat menggunakan Random Forest Regression dengan antarmuka web Streamlit.

## Cara Menjalankan

### Pertama Kali:
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app_streamlit.py
```

### Setelah Keluar (Jalankan Lagi):
```bash
streamlit run app_streamlit.py
```

**Atau double-click:** `JALANKAN_APLIKASI.bat`

## File Penting

- `train_model.py` - Script training model
- `app_streamlit.py` - Aplikasi web
- `requirements.txt` - Daftar library
- `dataset/data_final.csv` - Dataset (9,000 data penerbangan)

## Performa Model

- **Akurasi (R²):** 95.69%
- **MAE:** Rp 1,687
- **Algoritma:** Random Forest Regression

## Pengembang

**Wahyu Pratama** - 23011100058 - Informatika Pariwisata
