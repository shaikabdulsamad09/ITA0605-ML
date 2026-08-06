import pandas as pd

def find_s(training_data):
    hypothesis = None

    # Iterate through each training example
    for i, row in training_data.iterrows():
        features = list(row[:-1])
        target = row.iloc[-1]   

        if target == 'Yes':  # Consider only positive examples
            if hypothesis is None:
                # Initialize hypothesis with the first positive example
                hypothesis = list(features)
            else:
                # Generalize the hypothesis based on the current positive example
                for j in range(len(hypothesis)):
                    if hypothesis[j] != features[j]:
                        hypothesis[j] = '?'

    return hypothesis
# Sample training data
data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

df = pd.DataFrame(data, columns=['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'EnjoySport'])
print(df)

# Run the FIND-S algorithm
most_specific_hypothesis = find_s(df)

print(f"\nMost Specific Hypothesis: {most_specific_hypothesis}")
