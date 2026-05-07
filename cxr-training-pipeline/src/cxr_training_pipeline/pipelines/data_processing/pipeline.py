from kedro.pipeline import Node, Pipeline

from .nodes import (
    create_train_val_test_dfs,
    build_tensor_dataset,
)


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
                func=build_tensor_dataset,
                inputs=[
                    "xray_train_val",
                    "cxr8_segmentation_masks",
                    "params:data_processing_precompute.images_dir",
                    "params:data_processing_precompute.out_dir_full",
                    "params:data_processing_precompute.out_dir_crop",
                    "params:data_processing_precompute.out_dir_mask",
                    "params:data_processing_precompute.image_size",
                    "params:data_processing_precompute.image_type",
                    "params:data_processing_precompute.compute_tensors",
                ],
                outputs="tensor_train_val_dataset",
                name="build_tensor_dataset_train_node",
            ),
            Node(
                func=build_tensor_dataset,
                inputs=[
                    "xray_test",
                    "cxr8_segmentation_masks",
                    "params:data_processing_precompute.images_dir",
                    "params:data_processing_precompute.out_dir_full",
                    "params:data_processing_precompute.out_dir_crop",
                    "params:data_processing_precompute.out_dir_mask",
                    "params:data_processing_precompute.image_size",
                    "params:data_processing_precompute.image_type",
                    "params:data_processing_precompute.compute_tensors",
                ],
                outputs="tensor_test_dataset",
                name="build_tensor_dataset_test_node",
            ),
        ]
    )
