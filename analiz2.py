import pandas as pd
from scipy.stats import ttest_ind

# CSV dosyasını oku
veriler = pd.read_csv('train.csv')

# Rural ve Urban gruplarını ayır
rural_veriler = veriler[veriler['Residence_type'] == 'Rural']
urban_veriler = veriler[veriler['Residence_type'] == 'Urban']

# Rural ve Urban gruplarındaki stroke yüzdelerini al
rural_stroke_yuzdeleri = rural_veriler['stroke']
urban_stroke_yuzdeleri = urban_veriler['stroke']

# T-testi uygula
t_stat, p_value = ttest_ind(rural_stroke_yuzdeleri, urban_stroke_yuzdeleri)

# Sonuçları görüntüle
print(p_value)
if p_value < 0.05:
    print("Rural ve Urban bölgeleri arasında istatistiksel olarak anlamlı bir fark var.")
else:
    print("Rural ve Urban bölgeleri arasında istatistiksel olarak anlamlı bir fark yok.")


male_veriler = veriler[veriler['gender'] == 'Male']
Female_veriler = veriler[veriler['gender'] == 'Female']

# male ve Female gruplarındaki stroke yüzdelerini al
male_stroke_yuzdeleri = male_veriler['stroke']
Female_stroke_yuzdeleri = Female_veriler['stroke']

# T-testi uygula
t_stat, p_value = ttest_ind(male_stroke_yuzdeleri, Female_stroke_yuzdeleri)

# Sonuçları görüntüle
print(p_value)
if p_value < 0.05:
    print("male ve Female bölgeleri arasında istatistiksel olarak anlamlı bir fark var.")
else:
    print("male ve Female bölgeleri arasında istatistiksel olarak anlamlı bir fark yok.")


Yes_veriler = veriler[veriler['ever_married'] == 'Yes']
No_veriler = veriler[veriler['ever_married'] == 'No']

# Yes ve No gruplarındaki stroke yüzdelerini al
Yes_stroke_yuzdeleri = Yes_veriler['stroke']
No_stroke_yuzdeleri = No_veriler['stroke']

# T-testi uygula
t_stat, p_value = ttest_ind(Yes_stroke_yuzdeleri, No_stroke_yuzdeleri)

# Sonuçları görüntüle
print(p_value)
if p_value < 0.05:
    print("Yes ve No bölgeleri arasında istatistiksel olarak anlamlı bir fark var.")
else:
    print("Yes ve No bölgeleri arasında istatistiksel olarak anlamlı bir fark yok.")
