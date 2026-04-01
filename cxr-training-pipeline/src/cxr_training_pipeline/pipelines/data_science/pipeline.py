from kedro.pipeline import Node, Pipeline

from .nodes import (
    make_train_val_indices,
    build_transforms_node,
    prepare_datamodule_config,
    build_datamodule_node,
    build_classifier_node,
    build_lightning_module_node,
    train_model_node,
    predict_validation_node,
    predict_test_node,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=make_train_val_indices,
                inputs=["xray_train_val", "params:training.split"],
                outputs=["train_idx", "val_idx"],
                name="make_train_val_indices_node",
            ),
            Node(
                func=build_transforms_node,
                inputs=["params:training.transforms"],
                outputs=["train_tfms", "eval_tfms"],
                name="build_transforms_node",
            ),
            Node(
                func=prepare_datamodule_config,
                inputs=[
                    "xray_train_val",
                    "xray_test",
                    "train_idx",
                    "val_idx",
                    "train_tfms",
                    "eval_tfms",
                    "params:training.datamodule",
                ],
                outputs="datamodule_config",
                name="prepare_datamodule_config_node",
            ),
            Node(
                func=build_datamodule_node,
                inputs=["params:training.datamodule.name", "datamodule_config"],
                outputs="datamodule",
                name="build_datamodule_node",
            ),
            Node(
                func=build_classifier_node,
                inputs=["params:training.classifier"],
                outputs="model",
                name="build_classifier_node",
            ),
            Node(
                func=build_lightning_module_node,
                inputs=["model", "params:training.lit_module"],
                outputs="lit_model",
                name="build_lightning_module_node",
            ),
            Node(
                func=train_model_node,
                inputs=[
                    "datamodule",
                    "lit_model",
                    "params:training.trainer",
                ],
                outputs="best_checkpoint_path",
                name="train_model_node",
            ),
            Node(
                func=predict_validation_node,
                inputs=[
                    "datamodule",
                    "lit_model",
                    "best_checkpoint_path",
                    "params:training.trainer",
                ],
                outputs=["val_pred_proba", "val_targets"],
                name="predict_validation_node",
            ),
            Node(
                func=predict_test_node,
                inputs=[
                    "datamodule",
                    "lit_model",
                    "best_checkpoint_path",
                    "params:training.trainer",
                ],
                outputs=["test_pred_proba", "test_targets"],
                name="predict_test_node",
            ),
        ]
    )
