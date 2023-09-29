from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from faker import Faker
import random

# Inicjalizacja obiektu Faker
fake = Faker()

# Tworzenie listy miast
# 10 unikalnych miast, można dostosować ilość
cities = [fake.city() for _ in range(10)]

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

# 5. Wyszukiwanie danych
name_to_find = "John Doe"
john_doe = df[df["Name"] == name_to_find]
print(f"Dane dla {name_to_find}:")
print(john_doe)

# 6. Przetwarzanie danych (zmiana formatu numeru telefonu)
df["Phone Number"] = df["Phone Number"].str.replace("-", "")
print("Dane po przetworzeniu numeru telefonu:")
print(df)

# 7. Modelowanie predykcyjne (przy użyciu regresji liniowej)

X = df[["Age"]]
# Przykładowy cel predykcyjny - długość numeru telefonu
y = df["Phone Number"].str.len()

model = LinearRegression()
model.fit(X, y)

# 8. Klastrowanie (przy użyciu KMeans)

X = df[["Age"]]
kmeans = KMeans(n_clusters=3)
df["Cluster"] = kmeans.fit_predict(X)
print("Wyniki klastrowania:")
print(df[["Age", "Cluster"]])


# 9. Eksploracja danych (np. liczba unikalnych miast)
unique_cities = df["City"].nunique()
print("Liczba unikalnych miast:", unique_cities)
