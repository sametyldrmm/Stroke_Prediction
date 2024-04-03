import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model, load_model
from sklearn.preprocessing import MinMaxScaler

# CSV dosyasını okuma
data = pd.read_csv('train.csv')

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

stroke_output = data["stroke"]    
data = data.drop('stroke', axis=1)


print(data)


data_update = pd.DataFrame()

# Sürekli olarak DataFrame'leri birleştirme
for i in range(15304):  # n, eklemek istediğiniz DataFrame sayısı
    # Yeni DataFrame'i oluşturma veya verileri okuma
    new_df = data[data['id'] == i] 
    # DataFrame'leri birleştirme
    data_update = pd.concat([data_update, new_df])

data_update = data_update.drop('id', axis=1)
print(data_update)
# Input ve output verilerinin ayarlanması

X = data_update.values
y = stroke_output.values






model5 = load_model('stroke.99u003')
predictions5 = model5.predict(X)

# buradan sonrası ana_kod tan farklıdır


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Gerçek değerler
y_true = y
# Tahmin değerler
y_pred = predictions5

# Ortalama Mutlak Hata (Mean Absolute Error - MAE)

mae = mean_absolute_error(y, predictions5)
print("MAE:", mae)

# Ortalama Kare Hata (Mean Squared Error - MSE)
mse = mean_squared_error(y_true, y_pred)
print("MSE:", mse)

# Kök Ortalama Kare Hata (Root Mean Squared Error - RMSE)
rmse = mean_squared_error(y_true, y_pred, squared=False)
print("RMSE:", rmse)

# R-Kare skoru (Coefficient of Determination)
r2 = r2_score(y_true, y_pred)
print("R2 Score:", r2)

def count_close_numbers(arr1, arr2):
    count = 0
    for i in range(len(arr1)):
        if arr2[i] <= 0.01:
            count += 1
        if arr2[i] >= 0.99 :
          count += 1
    return count


print(count_close_numbers(y,predictions5))

