import pandas as pd
from faker import Faker
import random

# Inicjalizacja obiektu Faker
fake = Faker()

# Tworzenie listy miast
cities = [fake.city() for _ in range(10)]  # 10 unikalnych miast, można dostosować ilość

# Tworzenie przykładowych danych z powtarzającymi się miastami
data = []
for _ in range(100):
    name = fake.name()
    age = random.randint(18, 65)
    city = random.choice(cities)  # Losowe wybieranie miasta z listy
    email = fake.email()
    phone_number = fake.phone_number()
    
    data.append([name, age, city, email, phone_number])

# Tworzenie DataFrame
df = pd.DataFrame(data, columns=["Name", "Age", "City", "Email", "Phone Number"])

# 1. Analiza statystyczna
print("Średnia wieku:", df["Age"].mean())
print("Minimalny wiek:", df["Age"].min())
print("Maksymalny wiek:", df["Age"].max())

# 2. Wizualizacja danych (przy użyciu Matplotlib)
import matplotlib.pyplot as plt

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
from sklearn.linear_model import LinearRegression

X = df[["Age"]]
y = df["Phone Number"].str.len()  # Przykładowy cel predykcyjny - długość numeru telefonu

model = LinearRegression()
model.fit(X, y)

# 8. Klastrowanie (przy użyciu KMeans)
from sklearn.cluster import KMeans

X = df[["Age"]]
kmeans = KMeans(n_clusters=3)
df["Cluster"] = kmeans.fit_predict(X)
print("Wyniki klastrowania:")
print(df[["Age", "Cluster"]])

# # 9. Analiza korelacji
# correlation_matrix = df.corr()
# print("Macierz korelacji:")
# print(correlation_matrix)

# 10. Eksploracja danych (np. liczba unikalnych miast)
unique_cities = df["City"].nunique()
print("Liczba unikalnych miast:", unique_cities)

# Wykres słupkowy z statystyką miast
city_counts = df["City"].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(city_counts.index, city_counts.values)
plt.xlabel("Miasto")
plt.ylabel("Liczba Użytkowników")
plt.title("Statystyka Miast")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Wykresy pudełkowe wieku w poszczególnych miastach
plt.subplot(1, 2, 2)
plt.boxplot([df[df["City"] == city]["Age"] for city in cities], labels=cities)
plt.xlabel("Miasto")
plt.ylabel("Wiek")
plt.title("Rozkład Wieku w Poszczególnych Miastach")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()