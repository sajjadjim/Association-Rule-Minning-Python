import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Read data from CSV file
df = pd.read_csv('retail_dataset.csv')

# Preprocess the data
transactions = []
for _, row in df.iterrows():
    transactions.append([item for item in row if pd.notna(item)])

# Convert transactions into a suitable format for TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Find frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.15, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Filter rules based on confidence
confident_rules = rules[rules['confidence'] >= 0.5]

# Print confident rules
print("Confident association rules:")
print(confident_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Display rules with highest and lowest confidence rates
highest_confidence_rule = confident_rules.nlargest(1, 'confidence')
lowest_confidence_rule = confident_rules.nsmallest(1, 'confidence')

print("\nRule with the highest confidence:")
print(highest_confidence_rule[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

print("\nRule with the lowest confidence:")
print(lowest_confidence_rule[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Print frequent itemsets
print("\nFrequent itemsets:")
print(frequent_itemsets)

# Plot histogram of products
product_counts = defaultdict(int)
for transaction in transactions:
    for item in transaction:
        product_counts[item] += 1

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
