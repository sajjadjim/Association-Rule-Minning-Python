import pandas as pd 
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv('retail_dataset.csv')

# Generate list of transactions, removing NaN values
transactions = df.stack().groupby(level=0).apply(list).tolist()

# Function to calculate support
def calculate_support(transactions):
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in set(transaction):
            item_counts[item] += 1
    total_transactions = len(transactions)
    support = {item: count / total_transactions for item, count in item_counts.items()}
    return support

# Function to generate itemsets and their frequencies
def get_itemsets(transactions):
    itemsets = defaultdict(int)
    for transaction in transactions:
        for size in range(2, len(transaction) + 1):
            for combo in combinations(transaction, size):
                itemsets[combo] += 1
    return itemsets

# Function to generate rules and calculate confidence
def generate_rules(itemsets, transactions):
    total_transactions = len(transactions)
    rules = []
    for itemset, freq in itemsets.items():
        for i in range(len(itemset)):
            antecedent = itemset[:i] + itemset[i+1:]
            consequent = itemset[i]
            antecedent_freq = sum(1 for transaction in transactions if set(antecedent).issubset(transaction))
            confidence = freq / antecedent_freq
            rules.append((antecedent, consequent, confidence))
    return rules

# Calculate support for single items
support = calculate_support(transactions)

# Generate itemsets and their frequencies
itemsets = get_itemsets(transactions)

# Generate rules and calculate confidence
rules = generate_rules(itemsets, transactions)

# Identify highest and lowest confidence rates
rules_sorted = sorted(rules, key=lambda x: x[2], reverse=True)
highest_confidence = rules_sorted[0] if rules_sorted else None
lowest_confidence = rules_sorted[-1] if rules_sorted else None

# Print support, confidence rates, and highest/lowest confidence rates
print("Support:")
for item, supp in support.items():
    print(f"{item}: {supp:.2f}")

"""print("\nAssociation Rules and Confidence Rates:")
for antecedent, consequent, confidence in rules:
    print(f"{set(antecedent)} -> {consequent}: {confidence:.2f}")"""

if highest_confidence:
    print("\nHighest Confidence Rule:")
    print(f"{set(highest_confidence[0])} -> {highest_confidence[1]}: {highest_confidence[2]:.2f}")

if lowest_confidence:
    print("\nLowest Confidence Rule:")
    print(f"{set(lowest_confidence[0])} -> {lowest_confidence[1]}: {lowest_confidence[2]:.2f}")

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

