import numpy as np

concepts = np.array([
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change"]
])

target = np.array(["Yes", "Yes", "No", "Yes"])

print("Training Data:\n")
for i in range(len(concepts)):
    print(concepts[i], " -> ", target[i])

def candidate_elimination(concepts, target):

    # Initialize Specific Hypothesis
    S = concepts[0].copy()

    # Initialize General Hypothesis
    G = [["?" for i in range(len(S))] for j in range(len(S))]

    print("\nInitial Specific Hypothesis:")
    print(S)

    print("\nInitial General Hypothesis:")
    for row in G:
        print(row)

    # Process each training example
    for i, h in enumerate(concepts):

        print("\n--------------------------------")
        print("Training Example", i + 1)
        print(h, "->", target[i])

        if target[i] == "Yes":

            for x in range(len(S)):
                if h[x] != S[x]:
                    S[x] = "?"
                    G[x][x] = "?"

        else:

            for x in range(len(S)):
                if h[x] != S[x]:
                    G[x][x] = S[x]
                else:
                    G[x][x] = "?"

        print("\nSpecific Hypothesis:")
        print(S)

        print("\nGeneral Hypothesis:")
        for row in G:
            print(row)

    # Remove fully-general rows
    G_final = [row for row in G if row != ["?"] * len(S)]

    return S, G_final

S_final, G_final = candidate_elimination(concepts, target)

print("\n===================================")
print("Final Specific Hypothesis:")
print(S_final)

print("\nFinal General Hypothesis:")
for row in G_final:
    print(row)
