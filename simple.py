from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from faker import Faker
import random

# Inicjalizacja obiektu Faker
fake = Faker()

df = pd.read_csv("sample_data.csv")

# 1. Analiza statystyczna
print("Średnia wieku:", df["Age"].mean())
print("Minimalny wiek:", df["Age"].min())
print("Maksymalny wiek:", df["Age"].max())

# 2. Wizualizacja danych (przy użyciu Matplotlib)
plt.hist(df["Age"], bins=10, edgecolor='k')
plt.xlabel("Wiek")
plt.ylabel("Liczba osób")
plt.title("Histogram wieku")
plt.show()

# 3. Filtrowanie danych
young_people = df[df["Age"] < 30]
print("Osoby poniżej 30 roku życia:")
print(young_people)

# 4. Analiza grupowa
age_groups = df.groupby("Age").size()
print("Liczba osób w różnych grupach wiekowych:")
print(age_groups)
