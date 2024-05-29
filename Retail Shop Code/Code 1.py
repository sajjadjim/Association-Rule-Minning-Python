import pandas as pd
from itertools import combinations, chain
from collections import defaultdict
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv('retail_dataset.csv')

# Generate list of transactions, removing NaN values
transactions = df.stack().groupby(level=0).apply(list).tolist()

# Function to generate itemsets and their frequencies
def get_itemsets(transactions):
    itemsets = defaultdict(int)
    for transaction in transactions:
        for size in range(1, len(transaction) + 1):
            for combo in combinations(transaction, size):
                itemsets[frozenset(combo)] += 1
    return itemsets

# Function to calculate support
def calculate_support(itemsets, transactions):
    support = {item: freq / len(transactions) for item, freq in itemsets.items()}
    return support

# Function to generate rules and calculate confidence and lift
def generate_rules(itemsets, support):
    rules = []
    for itemset in itemsets:
        if len(itemset) > 1:
            for consequence in itemset:
                antecedent = itemset - frozenset([consequence])
                if antecedent in itemsets:
                    confidence = itemsets[itemset] / itemsets[antecedent]
                    lift = confidence / support[frozenset([consequence])]
                    rules.append((antecedent, frozenset([consequence]), support[itemset], confidence, lift))
    return rules

# Generate itemsets and their frequencies
itemsets = get_itemsets(transactions)

# Calculate support for each itemset
support = calculate_support(itemsets, transactions)

# Generate rules and calculate confidence and lift
rules = generate_rules(itemsets, support)

# Convert rules to a DataFrame
rules_df = pd.DataFrame(rules, columns=['Antecedent', 'Consequence', 'Support', 'Confidence', 'Lift'])

# Print association rules
print("Association Rules:")
print(rules_df)

# Plot histogram of products
product_counts = defaultdict(int)
for transaction in transactions:
    for item in transaction:
        product_counts[item] += 1

import numpy as np

# Get a list of unique colors
colors = plt.cm.viridis(np.linspace(0, 1, len(product_counts)))

# Plot histogram of products with each bar having a different color
for i, (product, count) in enumerate(product_counts.items()):
    plt.bar(product, count, color=colors[i])

plt.xlabel('Products')
plt.ylabel('Frequency')
plt.title('Histogram of Products')
plt.xticks(rotation=90)
plt.show()


# Print support and highest/lowest confidence rates
print("\nSupport:")
for item, supp in support.items():
    print(f"{item}: {supp:.2f}")

# Identify highest and lowest confidence rates
rules_sorted = sorted(rules, key=lambda x: x[3], reverse=True)
highest_confidence = rules_sorted[0] if rules_sorted else None
lowest_confidence = rules_sorted[-1] if rules_sorted else None

if highest_confidence:
    print("\nHighest Confidence Rule:")
    print(f"{set(highest_confidence[0])} -> {highest_confidence[1]}: {highest_confidence[3]:.2f}")

if lowest_confidence:
    print("\nLowest Confidence Rule:")
    print(f"{set(lowest_confidence[0])} -> {lowest_confidence[1]}: {lowest_confidence[3]:.2f}")