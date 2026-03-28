from kedro.pipeline import Pipeline, node
from .nodes import tune_thresholds


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=tune_thresholds,
                inputs=[
                    "val_targets",
                    "val_pred_proba",
                    "params:data_processing.pathology_list",
                    "params:threshold_tuning",
                ],
                outputs=["best_thresholds", "threshold_search_results"],
                name="tune_thresholds_node",
            )
        ]
    )
