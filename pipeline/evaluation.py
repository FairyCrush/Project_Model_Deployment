import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)


class ModelEvaluator:
    def __init__(self, label_encoder=None):
        self.label_encoder = label_encoder

    def compute_metrics(self, y_true, y_pred):
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro')),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro')),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        }

    def get_label_names(self):
        if self.label_encoder is not None:
            return list(self.label_encoder.classes_)
        return None

    def classification_report(self, y_true, y_pred):
        target_names = self.get_label_names()
        return classification_report(y_true, y_pred, target_names=target_names)

    def confusion_matrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    def plot_confusion_matrix(self, y_true, y_pred, title='Confusion Matrix', save_path=None):
        cm = self.confusion_matrix(y_true, y_pred)
        labels = self.get_label_names()
        fig, ax = plt.subplots(figsize=(7, 5))
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(ax=ax, colorbar=True)
        ax.set_title(title)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
        return fig

    def evaluate_and_log(self, y_true, y_pred, run_label=None):
        metrics = self.compute_metrics(y_true, y_pred)
        report = self.classification_report(y_true, y_pred)
        prefix = f'[{run_label}] ' if run_label else ''
        print(f'{prefix}Metrics:')
        for k, v in metrics.items():
            print(f'  {k}: {v:.4f}')
        print(f'{prefix}Classification report:')
        print(report)
        return metrics, report
