import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model, load_model
from sklearn.preprocessing import MinMaxScaler

# CSV dosyasını okuma
data = pd.read_csv('test.csv')

# **************** veri temizleme başlar ************ kategori olan verilari numeric verilere döüştürülülür
# One-Hot Encoding
one_hot_encoded = pd.get_dummies(data["gender"])
# # Sonuçları veri kümesine ekleme
data = pd.concat([data, one_hot_encoded], axis=1)
data = data.drop("gender", axis=1)

one_hot_encoded = pd.get_dummies(data['ever_married'])
# # Sonuçları veri kümesine ekleme
data = pd.concat([data, one_hot_encoded], axis=1)
data = data.drop('ever_married', axis=1)

one_hot_encoded = pd.get_dummies(data['Residence_type'])
# # Sonuçları veri kümesine ekleme
data = pd.concat([data, one_hot_encoded], axis=1)
data = data.drop('Residence_type', axis=1)
# data['Residence_type'] = 0.5

one_hot_encoded = pd.get_dummies(data['smoking_status'])
# # Sonuçları veri kümesine ekleme
data = pd.concat([data, one_hot_encoded], axis=1)
data = data.drop('smoking_status', axis=1)

one_hot_encoded = pd.get_dummies(data['work_type'])
# # Sonuçları veri kümesine ekleme
data = pd.concat([data, one_hot_encoded], axis=1)
data = data.drop('work_type', axis=1)



data_update = pd.DataFrame()

# Sürekli olarak DataFrame'leri birleştirme
for i in range(25507 -15304):  # n, eklemek istediğiniz DataFrame sayısı
    # Yeni DataFrame'i oluşturma veya verileri okuma
    new_df = data[data['id'] == i + 15304] 
    # DataFrame'leri birleştirme
    data_update = pd.concat([data_update, new_df])
id = data["id"]
data_update = data_update.drop('id', axis=1)
# Input ve output verilerinin ayarlanması

X = data_update.values
print(X)

# Verilerin ölçeklendirilmesi ana_kod ile üst kısım aynıdır eğer ana kodun devamına yüklenecekse aşşası yüklenmeli

model5 = load_model('stroke.99u003')
predictions5 = model5.predict(X)



# buradan sonrası ana_kod ile farklıdır
import csv
import numpy as np

with open("dosya3.csv", mode="w", newline="") as dosya:
    yazici = csv.writer(dosya)
    yazici.writerow(["id", "stroke"])  # Başlıkları yazdır

    # Her iki dizinin elemanlarını eşleştirerek yazdır
    for eleman1, eleman2 in zip(id, predictions5):
        yazici.writerow([eleman1, "{:.17f}".format(eleman2[0])])