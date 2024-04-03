import pandas as pd

# CSV dosyasını oku
veriler = pd.read_csv('train.csv')

# Evli olanlara göre analiz
evli_veriler = veriler[veriler['ever_married'] == 'Yes']
evli_stroke_count = evli_veriler['stroke'].sum()
evli_stroke_percentage = (evli_stroke_count / len(evli_veriler)) * 100

# Evli olmayanlara göre analiz
evli_olmayan_veriler = veriler[(veriler['ever_married'] == 'No') & (veriler['age'] >= 16)]
evli_olmayan_stroke_count = evli_olmayan_veriler['stroke'].sum()
evli_olmayan_stroke_percentage = (evli_olmayan_stroke_count / len(evli_olmayan_veriler)) * 100

# ERKEK olanlara göre analiz
Male_data = veriler[veriler['ever_married'] == 'Yes']
Male_data_count = Male_data['stroke'].sum()
Male_data_percentage = (Male_data_count / len(Male_data)) * 100

# KADIN olanlara göre analiz
Female_data = veriler[veriler['ever_married'] == 'No']
Female_data_count = Female_data['stroke'].sum()
Female_data_percentage = (Female_data_count / len(Female_data)) * 100

# İşlere göre analiz
is_verileri = veriler.groupby('work_type')['stroke'].mean() * 100

# Sonuçları görüntüle
print("Evli olanların stroke yüzdesi: {:.2f}%".format(evli_stroke_percentage))
print("Evli olmayanların stroke yüzdesi: {:.2f}%".format(evli_olmayan_stroke_percentage))
print("Erkeklere göre stroke yüzdesi: {:.2f}%".format(Male_data_percentage))
print("Kadınlara göre stroke yüzdesi: {:.2f}%".format(Female_data_percentage))
print("İşlere göre stroke yüzdeleri:")
print(is_verileri)
