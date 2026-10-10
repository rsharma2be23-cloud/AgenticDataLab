"""Optional MLflow adapter. Tracking never changes model selection or training."""
import os


class MLflowTracker:
    def __init__(self, tracking_uri=None, enabled=None):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or ""
        self.enabled = (bool(self.tracking_uri) if enabled is None else bool(enabled))

    def log(self, experiment, artifact_path=None, workflow_id=None):
        if not self.enabled:
            return {"status": "disabled"}
        try:
            import mlflow
            import mlflow.sklearn
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            name = os.getenv("MLFLOW_EXPERIMENT_NAME", "AgenticDataLab")
            mlflow.set_experiment(name)
            with mlflow.start_run() as run:
                mlflow.set_tags({"workflow_id": str(workflow_id or ""), "experiment_id": str(experiment.get("experiment_id", "")),
                                 "model_type": str(experiment.get("model", "")), "task_type": str(experiment.get("task_type", ""))})
                params = experiment.get("hyperparameters", {})
                for key, value in params.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        mlflow.log_param(str(key)[:250], str(value)[:500])
                metrics = dict(experiment.get("test_metrics", {}))
                metrics.update({"cv_mean": experiment.get("cv_results", {}).get("mean"),
                                "cv_std": experiment.get("cv_results", {}).get("std")})
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and value is not None:
                        mlflow.log_metric(str(key)[:250], float(value))
                dataset = experiment.get("dataset", {})
                mlflow.set_tags({"dataset_fingerprint": str(dataset.get("fingerprint", "")),
                                 "dataset_rows": str(dataset.get("rows", "")),
                                 "dataset_columns": str(dataset.get("columns", ""))})
                if artifact_path and os.path.isfile(artifact_path):
                    mlflow.log_artifact(artifact_path, artifact_path="model_artifacts")
                return {"status": "logged", "tracking_uri": mlflow.get_tracking_uri(),
                        "experiment_id": run.info.experiment_id, "run_id": run.info.run_id}
        except Exception as exc:
            return {"status": "unavailable", "error": str(exc)[:500]}
