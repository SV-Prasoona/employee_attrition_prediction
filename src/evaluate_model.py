from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score


def evaluate_model(y_test, y_pred):

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("\nClassification Report:\n", classification_report(y_test, y_pred))