from kedro.pipeline import Node, Pipeline

from .nodes import create_train_val_test_dfs, precompute_labels, precompute_images_to_npy


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=create_train_val_test_dfs,
                inputs=[
                    "cxr8_metadata",
                    "params:data_processing.columns_to_keep",
                    "cxr8_train_val_list",
                    "cxr8_test_list",
                    "params:data_processing.pathology_list",
                    "params:data_processing.ohe_column",
                ],
                outputs=["xray_train_val", "xray_test"],
                name="create_train_val_test_dfs_node",
            ),
            Node(
                func=precompute_images_to_npy,
                inputs=[
                    "xray_train_val",
                    "xray_test",
                    "params:data_processing_precompute.images_dir",
                    "params:data_processing_precompute.train_val_output_path",
                    "params:data_processing_precompute.test_output_path",
                    "params:data_processing_transforms",
                    "params:data_processing_precompute.image_size",
                    "params:data_processing_precompute.image_col",

                ],
                outputs=None,
                name="precompute_images_to_npy_node",
            ),
            Node(
                func=precompute_labels,
                inputs=[
                    "xray_train_val",
                    "xray_test",
                    "params:data_processing.pathology_list",
                ],
                outputs=["train_val_labels_npy", "test_labels_npy"],
                name="precompute_labels_to_npy_node",
            ),
        ]
    )
