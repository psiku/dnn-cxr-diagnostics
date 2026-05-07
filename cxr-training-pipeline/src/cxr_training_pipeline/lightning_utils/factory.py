from cxr_training_pipeline.lightning_utils.data_module import ImageOnlyDataModule, TensorDataModule, BaseDataModule
from cxr_training_pipeline.lightning_utils.classifier_module import ClassifierModule, BaseClassifier
from cxr_training_pipeline.models.paper_based.classifier import CXRClassifier
from cxr_training_pipeline.models.classifier import ChestXRayClassifier


class DataModuleFactory:
    """Factory class for creating data modules. Created for scalability and to avoid circular imports."""

    _modules = {
        "image_only": ImageOnlyDataModule,
        "tensor": TensorDataModule,
    }

    @classmethod
    def create(cls, module_name: str, **kwargs) -> BaseDataModule:
        module_class = cls._modules.get(module_name)
        if not module_class:
            raise ValueError(f"Data module '{module_name}' not found in factory.")
        return module_class(**kwargs)


class ClassifierFactory:
    """Factory class for creating classifiers. Created for scalability and to avoid circular imports."""

    _classifiers = {
        "cxr_classifier": CXRClassifier,
        "chest_xray_classifier": ChestXRayClassifier,
    }

    @classmethod
    def create(cls, classifier_name: str, **kwargs):
        classifier_class = cls._classifiers.get(classifier_name)
        if not classifier_class:
            raise ValueError(f"Classifier '{classifier_name}' not found in factory.")
        return classifier_class(**kwargs)


class LightningModuleFactory:
    """Factory class for creating PyTorch Lightning wrappers around classifiers."""

    _modules = {
        "default_classifier": ClassifierModule,
    }

    @classmethod
    def create(cls, module_name: str, model, **kwargs) -> BaseClassifier:
        module_class = cls._modules.get(module_name)
        if not module_class:
            raise ValueError(f"Lightning module '{module_name}' not found in factory.")
        return module_class(model=model, **kwargs)
