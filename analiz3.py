import pandas as pd

# CSV dosyasını oku
veriler = pd.read_csv('train.csv')

# Residence_type'e göre analiz
residence_verileri = veriler.groupby('Residence_type')['stroke'].mean() * 100

# avg_glucose_level'e göre analiz
glucose_verileri = veriler.groupby('avg_glucose_level')['stroke'].mean() * 100

# bmi'ye göre analiz
bmi_verileri = veriler.groupby('bmi')['stroke'].mean() * 100

# smoking_status'e göre analiz
smoking_verileri = veriler.groupby('smoking_status')['stroke'].mean() * 100

# Sonuçları görüntüle
print("Residence_type'e göre stroke yüzdeleri:")
print(residence_verileri)
print("\navg_glucose_level'e göre stroke yüzdeleri:")
print(glucose_verileri)
print("\nbmi'ye göre stroke yüzdeleri:")
print(bmi_verileri)
print("\nsmoking_status'e göre stroke yüzdeleri:")
print(smoking_verileri)