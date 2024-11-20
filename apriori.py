import pandas as pd
from apyori import apriori

def transactional_format(db):
    records = []
    for i, row in db.iterrows():
        record = [f"id({row['id']})", f"outlook({row['outlook']})", f"temperature({row['temperature']})",
                  f"humidity({row['humidity']})", f"windy({row['windy']})", f"play({row['play']})"]
        records.append(record)
    return records

def perform_apriori(transactions, minimum_support, minimum_confidence):
    association_rules = apriori(transactions, min_support=minimum_support, min_confidence=minimum_confidence)
    association_rules = list(association_rules)
    return association_rules


def format_apriori_output(association_rules):
    for item in association_rules:
        pair = item[0]
        items = [x for x in pair]
        print("frequent item sets: " + str(items))
        print("Support: " + str(item[1]))
        if (len(pair) > 1):
            for rule in item[2]:
                print("Rule: " + str(rule[0]) + " -> " + str(rule[1]))
                print("Confidence: " + str(rule[2]))
                print("Lift: " + str(rule[3]))
                print("=======================")

if __name__ == '__main__':
    data = pd.read_csv("datasets/weather_nominal.csv")
    transactional_data = transactional_format(data)
    association_rules = perform_apriori(transactional_data, 0.1, 0.7)
    format_apriori_output(association_rules)

