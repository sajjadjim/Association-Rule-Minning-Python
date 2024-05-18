import pandas as pd
from itertools import combinations, chain
from collections import defaultdict
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv('retail_dataset.csv')

# Generate list of transactions
transactions = df.apply(lambda x: x.dropna().tolist(), axis=1).tolist()

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

# Function to generate rules and calculate confidence
def generate_rules(itemsets, support):
    rules = []
    for itemset in itemsets:
        if len(itemset) > 1:
            for consequence in itemset:
                antecedent = itemset - frozenset([consequence])
                if antecedent in itemsets:
                    confidence = itemsets[itemset] / itemsets[antecedent]
                    rules.append((antecedent, frozenset([consequence]), confidence))
    return rules

# Generate itemsets and their frequencies
itemsets = get_itemsets(transactions)

# Calculate support for each itemset
support = calculate_support(itemsets, transactions)

# Generate rules and calculate confidence
rules = generate_rules(itemsets, support)

# Identify highest and lowest confidence rates
rules_sorted = sorted(rules, key=lambda x: x[2], reverse=True)
highest_confidence = rules_sorted[0]
lowest_confidence = rules_sorted[-1]

# Print support, confidence rates, and highest/lowest confidence rates
print("Support:")
for item, supp in support.items():
    if len(item) == 1:  # Only print support for single items
        print(f"{set(item)}: {supp:.2f}")

"""print("\nAssociation Rules and Confidence Rates:")
for antecedent, consequence, confidence in rules:
    print(f"{set(antecedent)} -> {set(consequence)}: {confidence:.2f}")"""

print("\nHighest Confidence Rule:")
print(f"{set(highest_confidence[0])} -> {set(highest_confidence[1])}: {highest_confidence[2]:.2f}")

print("\nLowest Confidence Rule:")
print(f"{set(lowest_confidence[0])} -> {set(lowest_confidence[1])}: {lowest_confidence[2]:.2f}")

# Generate histogram of products
product_counts = defaultdict(int)
for transaction in transactions:
    for item in transaction:
        product_counts[item] += 1

plt.bar(product_counts.keys(), product_counts.values())
plt.xlabel('Products')
plt.ylabel('Frequency')
plt.title('Histogram of Products')
plt.xticks(rotation=90)
plt.show()
