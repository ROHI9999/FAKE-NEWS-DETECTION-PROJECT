import csv
from faker import Faker
import random

fake = Faker()

csvheader = ['AUTHOR', 'TITLE', 'PUBLISHED AT', 'LABEL']
ourdata = []

for i in range(20): # Generating 20 fake news articles
    author = fake.name()
    title = fake.sentence(nb_words=6)
    published_at = fake.date_time_this_year()
    label = random.choice(['fake']) # Adding a random label to each article
    listing = [author, title, published_at, label]
    ourdata.append(listing)
    ourdata.append([])

with open('fake.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csvheader)
    writer.writerows(ourdata)
