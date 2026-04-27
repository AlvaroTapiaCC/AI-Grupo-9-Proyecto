def analyze_training(history, threshold=0.05):
    train_acc = history["train_acc"][-1]
    test_acc = history["test_acc"][-1]

    train_loss = history["train_loss"][-1]
    test_loss = history["test_loss"][-1]

    gap = train_acc - test_acc

    if train_acc < 0.7 and test_acc < 0.7:
        return "Absolute --> Underfitting"

    if gap > threshold and test_loss > train_loss:
        return "Absolute --> Overfitting"

    return "Absolute --> Good fit"


def compare_with_best(curr_metrics, best_metrics, curr_history, best_history, threshold=0.05):
    curr_train = curr_history["train_acc"][-1]
    curr_test = curr_history["test_acc"][-1]
    best_train = best_history["train_acc"][-1]
    best_test = best_history["test_acc"][-1]

    curr_gap = curr_train - curr_test
    best_gap = best_train - best_test

    messages = []

    if curr_metrics["accuracy"] < best_metrics["accuracy"]:
        messages.append("Relative to best model --> Worse performance")

    if curr_gap > best_gap + threshold:
        messages.append("Relative to best model --> More overfitting")

    if curr_train < best_train - threshold:
        messages.append("Relative to best model --> More underfitting")

    if not messages:
        return "Relative to best model --> Similar or better"

    return " | ".join(messages)