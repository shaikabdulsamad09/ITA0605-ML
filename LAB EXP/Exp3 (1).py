import pandas as pd
import numpy as np

data = {
    'Outlook' : ['Sunny','Sunny','Overcast','Rain','Rain','Rain',
                 'Overcast','Sunny','Sunny','Rain','Sunny',
                 'Overcast','Overcast','Rain'],

    'Temperature' : ['Hot','Hot','Hot','Mild','Cool','Cool',
                     'Cool','Mild','Cool','Mild','Mild',
                     'Mild','Hot','Mild'],

    'Humidity' : ['High','High','High','High','Normal','Normal',
                  'Normal','High','Normal','Normal','Normal',
                  'High','Normal','High'],

    'Wind' : ['Weak','Strong','Weak','Weak','Weak','Strong',
              'Strong','Weak','Weak','Weak','Strong',
              'Strong','Weak','Strong'],

    'Play' : ['No','No','Yes','Yes','Yes','No',
              'Yes','No','Yes','Yes','Yes',
              'Yes','Yes','No']
}

df = pd.DataFrame(data)

print("Training Dataset\n")
print(df)

def entropy(target):

    classes, counts = np.unique(target, return_counts=True)

    ent = 0

    for count in counts:

        p = count / np.sum(counts)

        ent -= p * np.log2(p)

    return ent

def information_gain(data, attribute, target):

    total_entropy = entropy(data[target])

    values, counts = np.unique(data[attribute], return_counts=True)

    weighted_entropy = 0

    for i in range(len(values)):

        subset = data[data[attribute] == values[i]]

        weighted_entropy += (counts[i] / np.sum(counts)) * entropy(subset[target])

    return total_entropy - weighted_entropy

def ID3(data, original_data, features, target="Play", parent_class=None):

    # All examples belong to one class
    if len(np.unique(data[target])) == 1:

        return np.unique(data[target])[0]

    # Empty dataset
    elif len(data) == 0:

        return np.unique(original_data[target])[
            np.argmax(np.unique(original_data[target], return_counts=True)[1])
        ]

    # No features left
    elif len(features) == 0:

        return parent_class

    else:

        parent_class = np.unique(data[target])[
            np.argmax(np.unique(data[target], return_counts=True)[1])
        ]

        gains = [information_gain(data, feature, target) for feature in features]

        best_feature = features[np.argmax(gains)]

        tree = {best_feature: {}}

        remaining_features = [f for f in features if f != best_feature]

        for value in np.unique(data[best_feature]):

            subset = data[data[best_feature] == value]

            subtree = ID3(
                subset,
                original_data,
                remaining_features,
                target,
                parent_class
            )

            tree[best_feature][value] = subtree

        return tree

features = list(df.columns[:-1])

tree = ID3(df, df, features)

print("\nDecision Tree\n")
print(tree)

def predict(query, tree):

    for key in list(query.keys()):

        if key in tree.keys():

            try:

                result = tree[key][query[key]]

            except:

                return "Unknown"

            if isinstance(result, dict):

                return predict(query, result)

            else:

                return result

sample = {
    "Outlook":"Sunny",
    "Temperature":"Cool",
    "Humidity":"High",
    "Wind":"Strong"
}

print("\nNew Sample")
print(sample)

prediction = predict(sample, tree)

print("\nPrediction =", prediction)
