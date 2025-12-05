from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

def compute_metrics(loss, y_true, y_pred, y_prob, num_classes):
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except Exception:
        auc = float('nan')

    return {
        'loss': float(loss),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall':  float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1-score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'auc': auc
    }, confusion_matrix(y_true, y_pred)
