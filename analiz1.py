import pandas as pd
from scipy.stats import pearsonr

# CSV dosyasını oku
veriler = pd.read_csv('train.csv')

# avg_glucose_level ve stroke arasındaki korelasyonu hesapla
korelasyon, p_value = pearsonr(veriler['avg_glucose_level'], veriler['stroke'])
# Sonucu görüntüle
print("avg_glucose_level ve stroke arasındaki korelasyon: {:.2f}".format(korelasyon))

korelasyon2, p_value2 = pearsonr(veriler['bmi'], veriler['stroke'])
# Sonucu görüntüle
print("bmi ve stroke arasındaki korelasyon: {:.2f}".format(korelasyon2))

korelasyon, p_value = pearsonr(veriler['age'], veriler['stroke'])
# Sonucu görüntüle
print("age ve stroke arasındaki korelasyon: {:.2f}".format(korelasyon))

