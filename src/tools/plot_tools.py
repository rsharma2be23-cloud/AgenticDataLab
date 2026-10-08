from sklearn.metrics import accuracy_score, mean_squared_error

class MetricValidatorTool:
    def validate_classification(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        return {"accuracy": acc}

    def validate_regression(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {"mse": mse}
