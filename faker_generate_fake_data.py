from faker import Faker

fake = Faker()

print(fake.name())
print(fake.text())
print(fake.address())
print(fake.catch_phrase())
print(fake.company())
print(fake.job())


for _ in range(10):
  print(fake.name())


