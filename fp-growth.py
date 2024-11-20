import pandas as pd
import pyfpgrowth


def data_transform(data):
    data = data.values.tolist()
    transactions = [[x for x in sublist if str(x) != 'nan'] for sublist in data]
    return transactions

if __name__ == '__main__':
    data = pd.read_csv("datasets/store_data.csv")
    transactions = data_transform(data)
    freq_patterns = pyfpgrowth.find_frequent_patterns(transactions=transactions, support_threshold=200)
    rules = pyfpgrowth.generate_association_rules(patterns=freq_patterns, confidence_threshold=0.2)

    print(freq_patterns)
    print(rules)

