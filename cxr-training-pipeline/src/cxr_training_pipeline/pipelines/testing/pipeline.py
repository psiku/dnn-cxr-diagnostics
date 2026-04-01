from kedro.pipeline import Node, Pipeline, node

from .nodes import (
    evaluate_default_thresholds_node,
    evaluate_tuned_thresholds_node,
    compare_metrics_node,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=evaluate_default_thresholds_node,
                inputs=[
                    "test_targets",
                    "test_pred_proba",
                    "params:data_processing.pathology_list",
                    "params:threshold_tuning.default_threshold",
                ],
                outputs=["test_metrics_default", "per_class_default"],
                name="evaluate_default_thresholds_node",
            ),
            node(
                func=evaluate_tuned_thresholds_node,
                inputs=[
                    "test_targets",
                    "test_pred_proba",
                    "best_thresholds",
                    "params:data_processing.pathology_list",
                ],
                outputs=["test_metrics_tuned", "per_class_tuned"],
                name="evaluate_tuned_thresholds_node",
            ),
            node(
                func=compare_metrics_node,
                inputs=[
                    "test_metrics_default",
                    "test_metrics_tuned",
                    "per_class_default",
                    "per_class_tuned",
                ],
                outputs=[
                    "test_metrics_comparison",
                    "test_per_class_comparison",
                ],
                name="compare_metrics_node",
            ),
        ]
    )