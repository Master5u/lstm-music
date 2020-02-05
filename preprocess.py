import pandas as pd
data = pd.read_csv("IBM.csv")
close_price = []
normalization_price = []

print(data.head(5))

# Extract all close price into a list
for row in data['Close']:
    close_price.append(row)

print(close_price)
print(len(close_price))
print(data.shape)

# Find the max and min in close price
print(max(close_price))
print(min(close_price))

# Normalize the price into= piano roll
def pricenormalization (close_price, normalization_price, a, b):
    max_price = max(close_price)
    min_price = min(close_price)

    for price in close_price:
        result = a + (b - a)/(max_price - min_price)*(price - min_price)
        normalization_price.append(result)
    return;

pricenormalization(close_price,normalization_price,0,88)

print(normalization_price)