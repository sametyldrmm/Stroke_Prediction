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
print(one_hot_encoded)
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

# sayısal verileri normalize etmenin yararlı olduğu bir çok farklı yerde okuyup öğrensekte bunun pek bir geçerliliği yok
#  sebebi ise normalize etmek bir tek verilerle daha hızlı işlem yapmak içindir bizde böyle bir gereksinim yok

print(data)

data_update = pd.DataFrame()

# Sürekli olarak DataFrame'leri birleştirme
for i in range(15304):  # n, eklemek istediğiniz DataFrame sayısı
    # Yeni DataFrame'i oluşturma veya verileri okuma
    new_df = data[data['id'] == i] 
    # DataFrame'leri birleştirme
    data_update = pd.concat([data_update, new_df])

data_update = data_update.drop('id', axis=1)
# Input ve output verilerinin ayarlanması

X = data_update.values 
y = stroke_output.values


# Modelin oluşturulması

model = Sequential()

# Yoğun katman
model.add(Dense(64, input_dim=21, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
# Çıktı katmanı
model.add(Dense(1, activation='sigmoid'))

# Modelin derlenmesi
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# model = tf.keras.Model(inputs, outputs)
model = load_model("stroke.99u003")

model.summary()

# Modelin eğitilmesi

model.fit(X, y, epochs=5000, batch_size=512)

# Modelin kaydedilmesi
save_model(model, "stroke.99u004")
