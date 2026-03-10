import lightning as L
from src.cxr_training_pipeline.models.paper_based.w_cel import BatchBalancedBCEWithLogitsLoss
from torch import nn
import torch


class LitPaperResNet50ChestXray(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr_backbone: float = 1e-5,
        lr_head: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.model = model
        self.lr_backbone = lr_backbone
        self.lr_head = lr_head
        self.weight_decay = weight_decay

        self.criterion = BatchBalancedBCEWithLogitsLoss()

    def forward(self, image, retain_transition_grad: bool = False):
        return self.model(image, retain_transition_grad=retain_transition_grad)

    def _shared_step(self, batch, stage: str):
        image, target = batch   # bez metadata
        out = self(image)
        logits = out["logits"]

        loss = self.criterion(logits, target)

        self.log(f"{stage}_loss", loss, prog_bar=(stage != "train"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        head_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and not n.startswith("backbone.")
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.lr_backbone},
                {"params": head_params, "lr": self.lr_head},
            ],
            weight_decay=self.weight_decay,
        )
        return optimizer
