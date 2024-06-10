import numpy as np

def dcg(relevences, k):
    relevances = np.array(relevences)[:k]
    if relevences.size:
        return np.sum(relevences / np.log2(np.arrange(2, relevances.size + 2)))
    return 0.

def ndcg(actuals, predictions, k):
    ideal_relevances = sorted(actuals, reverse=True)
    actuals_relevances = [1 if pred in actuals else 0 for pred in predictions[:k]]
    return dcg(actuals_relevances, k) / dcg(ideal_relevances, k)

