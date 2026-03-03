import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm

class HybridResNetXception(nn.Module):
    """
    Hybrid Model for Unified Forgery Detection

    Branch 1: ResNet (low-level texture learning)
    Branch 2: Xception (deepfake semantic learning)
    Fusion: Feature Concatenation + FC
    Output: 2 classes (Real=0, Fake=1)
    """

    def __init__(
        self,
        resnet_type="resnet50",
        pretrained=True,
        fusion_dim=512,
        dropout=0.5,
        num_classes=2
    ):
        super(HybridResNetXception, self).__init__()

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

        # Remove FC layer
        self.resnet = nn.Sequential(*list(backbone.children())[:-1])
        self.resnet_out = resnet_out

        self.xception = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=0,     # removes classifier
            global_pool="avg"  # ensures feature output
        )

        self.xception_out = self.xception.num_features

        self.fusion = nn.Sequential(
            nn.Linear(self.resnet_out + self.xception_out, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(fusion_dim, num_classes)


    def forward(self, x):


        r_feat = self.resnet(x)              # (B, C, 1, 1)
        r_feat = torch.flatten(r_feat, 1)    # (B, C)


        x_feat = self.xception(x)            # (B, C)

        fused = torch.cat((r_feat, x_feat), dim=1)

        fused = self.fusion(fused)

        out = self.classifier(fused)

        return out


    def extract_features(self, x):

        r_feat = self.resnet(x)
        r_feat = torch.flatten(r_feat, 1)

        x_feat = self.xception(x)

        fused = torch.cat((r_feat, x_feat), dim=1)
        fused = self.fusion(fused)

        return fused


def freeze_resnet(model):
    for param in model.resnet.parameters():
        param.requires_grad = False


def freeze_xception(model):
    for param in model.xception.parameters():
        param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

def build_model(resnet_type="resnet50"):
    model = HybridResNetXception(resnet_type=resnet_type)
    return model

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model("resnet50").to(device)

    dummy = torch.randn(4, 3, 224, 224).to(device)

    output = model(dummy)

    print("Input shape :", dummy.shape)
    print("Output shape:", output.shape)  # Expected: [4,2]