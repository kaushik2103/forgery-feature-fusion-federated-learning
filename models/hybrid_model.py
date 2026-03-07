import torch
import torch.nn as nn
from torchvision import models
import timm


class HybridResNetXception(nn.Module):
    """
    Hybrid Model for Unified Forgery Detection

    Branch 1 : ResNet  (texture / spoof artifacts)
    Branch 2 : Xception (deepfake semantic artifacts)

    Fusion : Feature Concatenation + Fully Connected

    Output :
        0 → Real
        1 → Fake
    """

    def __init__(
        self,
        resnet_type="resnet50",
        pretrained=True,
        fusion_dim=512,
        dropout=0.2,
        num_classes=2
    ):
        super(HybridResNetXception, self).__init__()

        # --------------------------------------------------
        # ResNet Backbone
        # --------------------------------------------------
        if resnet_type == "resnet18":
            backbone = models.resnet18(pretrained=pretrained)
            resnet_out = 512

        elif resnet_type == "resnet34":
            backbone = models.resnet34(pretrained=pretrained)
            resnet_out = 512

        elif resnet_type == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
            resnet_out = 2048

        else:
            raise ValueError("Unsupported ResNet type")

        # Remove classification layer
        self.resnet = nn.Sequential(*list(backbone.children())[:-1])
        self.resnet_out = int(resnet_out)

        # --------------------------------------------------
        # Xception Backbone (from timm)
        # --------------------------------------------------
        self.xception = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=0,        # remove classifier
            global_pool="avg"
        )

        # Ensure integer type
        self.xception_out = int(self.xception.num_features)

        # --------------------------------------------------
        # Feature Fusion Layer
        # --------------------------------------------------
        fusion_input = self.resnet_out + self.xception_out

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # --------------------------------------------------
        # Final Classifier
        # --------------------------------------------------
        self.classifier = nn.Linear(fusion_dim, num_classes)

    # ------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------
    def forward(self, x):

        # ResNet branch
        r_feat = self.resnet(x)            # (B, C, 1, 1)
        r_feat = torch.flatten(r_feat, 1)  # (B, C)

        # Xception branch
        x_feat = self.xception(x)          # (B, C)

        # Feature fusion
        fused = torch.cat((r_feat, x_feat), dim=1)

        fused = self.fusion(fused)

        out = self.classifier(fused)

        return out

    # ------------------------------------------------------
    # Feature Extraction (used in analysis / FL similarity)
    # ------------------------------------------------------
    def extract_features(self, x):

        r_feat = self.resnet(x)
        r_feat = torch.flatten(r_feat, 1)

        x_feat = self.xception(x)

        fused = torch.cat((r_feat, x_feat), dim=1)

        fused = self.fusion(fused)

        return fused


# ----------------------------------------------------------
# Freeze Utilities (useful for transfer learning)
# ----------------------------------------------------------

def freeze_resnet(model):
    for param in model.resnet.parameters():
        param.requires_grad = False


def freeze_xception(model):
    for param in model.xception.parameters():
        param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


# ----------------------------------------------------------
# Build Model (used by other scripts)
# ----------------------------------------------------------

def build_model(resnet_type="resnet50"):

    model = HybridResNetXception(
        resnet_type=resnet_type
    )

    return model


# ----------------------------------------------------------
# Test Model
# ----------------------------------------------------------

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model("resnet50").to(device)

    dummy = torch.randn(4, 3, 224, 224).to(device)

    output = model(dummy)

    print("Input shape :", dummy.shape)
    print("Output shape:", output.shape)  # Expected: [4, 2]