def precision(actuals, predictions, k):
    actuals_set = set(actuals)
    pred_set = set(predictions[:k])
    return len(actuals_set & pred_set) / len(pred_set)