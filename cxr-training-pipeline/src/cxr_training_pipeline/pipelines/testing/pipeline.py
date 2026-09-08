from kedro.pipeline import Pipeline, node

from .nodes import (
    evaluate_default_thresholds_node,
    evaluate_localization_iou_node,
    evaluate_tuned_thresholds_node,
    evaluate_weighted_localization_iou_node,
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
                outputs=["test_metrics_default", "test_per_class_before_tuning"],
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
                outputs=["test_metrics_tuned", "test_per_class_after_tuning"],
                name="evaluate_tuned_thresholds_node",
            ),
            node(
                func=evaluate_localization_iou_node,
                inputs=[
                    "cxr8_bbox",
                    "tensor_test_dataset",
                    "best_checkpoint_path",
                    "params:training.classifier",
                    "params:training.lit_module",
                    "params:data_processing.pathology_list",
                    "params:localization",
                    "params:training.datamodule.kwargs",
                    "params:training.transforms",
                ],
                outputs=[
                    "localization_iou_per_image",
                    "localization_iou_per_class",
                    "localization_iou_metrics",
                ],
                name="evaluate_localization_iou_node",
            ),
            node(
                func=evaluate_weighted_localization_iou_node,
                inputs=[
                    "cxr8_bbox",
                    "tensor_test_dataset",
                    "best_checkpoint_path",
                    "params:training.classifier",
                    "params:training.lit_module",
                    "params:localization",
                    "params:training.datamodule.kwargs",
                    "params:training.transforms",
                ],
                outputs=[
                    "localization_weighted_iou_per_image",
                    "localization_weighted_iou_metrics",
                ],
                name="evaluate_weighted_localization_iou_node",
            ),
        ]
    )
