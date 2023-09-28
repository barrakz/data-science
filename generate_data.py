import pandas as pd
from faker import Faker
import random

# Inicjalizacja obiektu Faker
fake = Faker()

# Tworzenie przykładowych danych
data = []
for _ in range(100):
    name = fake.name()
    age = random.randint(18, 65)
    city = fake.city()
    email = fake.email()
    phone_number = fake.phone_number()
    
    data.append([name, age, city, email, phone_number])

# Tworzenie DataFrame
df = pd.DataFrame(data, columns=["Name", "Age", "City", "Email", "Phone Number"])

# Zapisywanie danych do pliku CSV
df.to_csv("sample_data.csv", index=False)

print(f"Przykładowe dane zostały zapisane do pliku sample_data.csv.")
