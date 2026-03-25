import torch
import torch.nn as nn
from torchvision import models
import timm
import torch.nn.functional as F


class HybridResNetXception(nn.Module):

    def __init__(
        self,
        resnet_type="resnet50",
        pretrained=True,
        fusion_dim=512,
        dropout=0.3,
        num_classes=2
    ):
        super(HybridResNetXception, self).__init__()


        if resnet_type == "resnet18":
            backbone = models.resnet18(weights="DEFAULT" if pretrained else None)
            resnet_out = 512

        elif resnet_type == "resnet34":
            backbone = models.resnet34(weights="DEFAULT" if pretrained else None)
            resnet_out = 512

        elif resnet_type == "resnet50":
            backbone = models.resnet50(weights="DEFAULT" if pretrained else None)
            resnet_out = 2048

        else:
            raise ValueError("Unsupported ResNet type")

        self.resnet = nn.Sequential(*list(backbone.children())[:-1])
        self.resnet_out = resnet_out


        self.xception = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )

        self.xception_out = self.xception.num_features


        self.norm_resnet = nn.LayerNorm(self.resnet_out)
        self.norm_xception = nn.LayerNorm(self.xception_out)


        fusion_input = self.resnet_out + self.xception_out

        self.attention = nn.Sequential(
            nn.Linear(fusion_input, fusion_input),
            nn.ReLU(),
            nn.Linear(fusion_input, fusion_input),
            nn.Sigmoid()
        )


        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )


        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )


    def forward(self, x):

        r_feat = self.resnet(x)
        r_feat = torch.flatten(r_feat, 1)
        r_feat = self.norm_resnet(r_feat)

        x_feat = self.xception(x)
        x_feat = self.norm_xception(x_feat)

        fused = torch.cat((r_feat, x_feat), dim=1)

        attn = self.attention(fused)
        fused = fused * attn

        fused = self.fusion(fused)

        out = self.classifier(fused)

        return out

    def extract_features(self, x):

        r_feat = self.resnet(x)
        r_feat = torch.flatten(r_feat, 1)
        r_feat = self.norm_resnet(r_feat)

        x_feat = self.xception(x)
        x_feat = self.norm_xception(x_feat)

        fused = torch.cat((r_feat, x_feat), dim=1)

        attn = self.attention(fused)
        fused = fused * attn

        fused = self.fusion(fused)

        fused = F.normalize(fused, dim=1)

        return fused


def freeze_backbone(model):
    for param in model.resnet.parameters():
        param.requires_grad = False
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
    print("Output shape:", output.shape)