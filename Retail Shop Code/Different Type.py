import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Function to load CSV file
def load_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

# Function to preprocess the data (assuming the CSV file has transactional data)
def preprocess_data(df):
    # Ensure all data is boolean (True/False)
    return df.apply(lambda x: x.apply(lambda y: True if y else False))

# Function to perform Apriori algorithm and generate rules
def run_apriori(df, min_support=0.1, min_confidence=0.5, min_lift=1.0):
    # Generate frequent itemsets
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    # Generate the association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # Filter rules by lift
    rules = rules[rules['lift'] >= min_lift]

    return frequent_itemsets, rules

# Main function
def main():
    # File path to the CSV file
    file_path = input("Enter the path to the CSV file: ")

    # Load the data
    data = load_csv(file_path)
    if data is None:
        return

    # Preprocess the data
    data = preprocess_data(data)

    # Parameters for Apriori
    min_support = float(input("Enter minimum support (e.g., 0.1 for 10%): "))
    min_confidence = float(input("Enter minimum confidence (e.g., 0.5 for 50%): "))
    min_lift = float(input("Enter minimum lift (e.g., 1.0): "))

    # Run Apriori algorithm
    frequent_itemsets, rules = run_apriori(data, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift)

    # Create a DataFrame for the rules
    rules_table = pd.DataFrame({
        'Rule': [f"{', '.join(list(rule['antecedents']))} -> {', '.join(list(rule['consequents']))}" for idx, rule in rules.iterrows()],
        'Support': rules['support'],
        'Confidence': rules['confidence'],
        'Lift': rules['lift']
    })

    # Display the results
    print("\nFrequent Itemsets:")
    print(frequent_itemsets)

    print("\nAssociation Rules:")
    print(rules_table)

if __name__ == "__main__":
    main()
