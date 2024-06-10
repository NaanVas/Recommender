def recall(actuals, predictions, k):
    actual_set = set(actuals)
    pred_set = set(predictions[:k])
    return len(actual_set & pred_set) / len(pred_set)