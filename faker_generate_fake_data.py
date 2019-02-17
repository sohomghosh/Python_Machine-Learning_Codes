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

  
# Generate fake texts
import faker
from faker import Faker
fake = Faker('en_US')
for _ in range(100):
    print(fake.text())
    
