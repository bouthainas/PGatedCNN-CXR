import cv2
import math
import numpy as np
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
import torchmetrics.functional as metrics


# -------------------------
# Attention & Gated blocks
# -------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(cat))
        return x * attn


class GatedCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.gate_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xn = self.norm(x)
        gate = self.gate_conv1(xn)
        gate = self.sigmoid(gate)
        main = self.conv(x)
        main = self.bn(main)
        main = self.act(main)
        return main * gate + main


# -------------------------
# Utility helpers
# -------------------------
def ensure_4d_feature(tensor):
    """Convert possible 2D/3D feature outputs from timm into (B,C,H,W)."""
    if tensor.dim() == 4:
        return tensor
    if tensor.dim() == 3:
        # common pattern: (B, N, C) or (B, C, N)
        B, N, C = tensor.shape
        side = int(math.sqrt(N))
        if side * side == N:
            # assume (B, N, C) -> (B, C, side, side)
            t = tensor.permute(0, 2, 1).contiguous()
            return t.view(B, C, side, side)
    if tensor.dim() == 2:
        return tensor.unsqueeze(-1).unsqueeze(-1)
    return tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


# -------------------------
# Model with augmentations
# -------------------------
class Model(pl.LightningModule):
    def __init__(
        self,
        backbone='mambaout_femto',
        seg_model=None,
        dataset='RALO',
        embed_channels=288,
        fc_hidden=128,
        lr=1e-5,
        trainingsize=100,
        share_backbone=True,
        pretrained_backbone=True,
    ):
        """
        seg_model: pre-trained segmentation nn.Module (UNet++). It should accept a single-image
                   tensor of shape (1,1,H,W) or (1,3,H,W) and output a single-channel mask logits.
        """
        super().__init__()
        self.save_hyperparameters()

        self.seg_model = seg_model
        self.share_backbone = share_backbone
        self.lr = lr
        self.trainingsize = trainingsize
        self.embed_channels = embed_channels
        self.fc_hidden = fc_hidden

        # Loss 
        self.loss_func = nn.L1Loss()

        # Encoders
        self.encoder1 = timm.create_model(backbone, pretrained=pretrained_backbone, num_classes=0)
        self.encoder2 = timm.create_model(backbone, pretrained=pretrained_backbone, num_classes=0)
        self.encoder3 = timm.create_model(backbone, pretrained=pretrained_backbone, num_classes=0)
        self.encoder4 = timm.create_model(backbone, pretrained=pretrained_backbone, num_classes=0)

        # Adapter conv (lazy init)
        self.adapter = None

        # Attention & gated blocks
        self.ca = ChannelAttention(self.embed_channels)
        self.gated_block = GatedCNNBlock(self.embed_channels, self.embed_channels)
        self.spatial_att = SpatialAttention(kernel_size=7)

        # Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.embed_channels, self.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_hidden, 1),
        )

        # metrics storage (optional)
        self.tr_loss, self.tr_mae, self.vl_loss, self.vl_mae, self.ts_loss, self.ts_mae = [], [], [], [], [], []

    # ------------------------------------
    # Image splitting (4 quadrants)
    # ------------------------------------
    def split(self, x):
        B, C, H, W = x.shape
        return (
            x[:, :, :H//2, :W//2],  # top-left
            x[:, :, :H//2, W//2:],  # top-right
            x[:, :, H//2:, :W//2],  # bottom-left
            x[:, :, H//2:, W//2:]   # bottom-right
        )


    # -------------------------
    # Segmentation helpers
    # -------------------------
    def segment_lungs(self, image_tensor):
        """
        Run seg_model on a single image tensor.
        image_tensor: torch.Tensor (C,H,W) or (1,C,H,W), values expected normalized for seg_model.
        returns: left_mask, right_mask as boolean np arrays (H,W).
        """
        if self.seg_model is None:
            raise ValueError("seg_model is None; provide a pretrained segmentation model at init.")

        self.seg_model.eval()
        device = next(self.seg_model.parameters()).device

        # Prepare input: single image, single batch
        with torch.no_grad():
            if image_tensor.dim() == 3:
                inp = image_tensor.unsqueeze(0).to(device)  # (1,C,H,W)
            else:
                inp = image_tensor.to(device)
            # If seg_model expects single channel, convert if necessary
            # Assume seg_model outputs single-channel logits
            pred = self.seg_model(inp)
            # If output has channels >1, assume first channel is lung mask
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            pred = pred.squeeze(0).squeeze(0)  # (H,W)
            prob = torch.sigmoid(pred).cpu().numpy()

        h, w = prob.shape
        mid = w // 2
        left_mask = (prob[:, :mid] > 0.5).astype(np.uint8)
        right_mask = (prob[:, mid:] > 0.5).astype(np.uint8)
        left_mask = np.pad(left_mask, ((0, 0), (0, mid)), constant_values=0)
        right_mask = np.pad(right_mask, ((0, 0), (mid, 0)), constant_values=0)
        return left_mask, right_mask

    @staticmethod
    def flip_region_np(image_np, mask_np):
        """image_np: HxW or HxWxC (C channels). mask_np: HxW binary mask (0/1)."""
        # Accept either grayscale or multi-channel
        if image_np.ndim == 2:
            img_masked = (image_np * mask_np).astype(image_np.dtype)
            flipped = cv2.flip(img_masked, 1)
            return flipped
        elif image_np.ndim == 3:
            masked = (image_np * mask_np[..., None]).astype(image_np.dtype)
            flipped = cv2.flip(masked, 1)
            return flipped
        else:
            raise ValueError("Unexpected image_np ndim: %d" % image_np.ndim)

    # -------------------------
    # Augmentations (tensor in, tensor out)
    # -------------------------
    def self_slr_image(self, image_tensor, S_L, S_R):
        """
        image_tensor: torch.Tensor (C,H,W) float (0..1 or normalized).
        Returns list of (aug_image_tensor, score)
        """
        # convert to numpy in CPU for cv2 operations
        device = image_tensor.device
        dtype = image_tensor.dtype

        # Bring image to HxW or HxWxC in numpy (use the original pixels, not normalized to specific mean/std)
        img_np = image_tensor.detach().cpu().numpy()  # (C,H,W)
        img_np = np.transpose(img_np, (1, 2, 0)) if img_np.shape[0] > 1 else img_np.squeeze(0)

        # Get masks
        left_mask, right_mask = self.segment_lungs(image_tensor)

        # Flip left and right regions
        L_flip = self.flip_region_np(img_np, left_mask)
        R_flip = self.flip_region_np(img_np, right_mask)

        aug_results = []

        # (i) Left lung replaced by flipped left
        A1 = img_np.copy()
        A1[left_mask == 1] = L_flip[left_mask == 1]
        S_A1 = 2.0 * float(S_L)
        aug_results.append((A1, S_A1))

        # (ii) Right lung replaced by flipped right
        A2 = img_np.copy()
        A2[right_mask == 1] = R_flip[right_mask == 1]
        S_A2 = 2.0 * float(S_R)
        aug_results.append((A2, S_A2))

        # (iii) Both lungs replaced by their flips
        A3 = img_np.copy()
        both_mask = np.logical_or(left_mask, right_mask)
        # build combined flipped (if C>1 sum channels; but we will replace per pixel)
        combined_flip = A1 * 0
        combined_flip[left_mask == 1] = L_flip[left_mask == 1]
        combined_flip[right_mask == 1] = R_flip[right_mask == 1]
        A3[both_mask == 1] = combined_flip[both_mask == 1]
        S_A3 = float(S_L) + float(S_R)
        aug_results.append((A3, S_A3))

        # convert back to tensors (C,H,W) and to original device/dtype
        out_tensors = []
        for aug_img_np, score in aug_results:
            if aug_img_np.ndim == 2:
                aug_img_np = np.expand_dims(aug_img_np, axis=-1)
            aug_t = torch.from_numpy(np.transpose(aug_img_np, (2, 0, 1))).to(device=device, dtype=dtype)
            out_tensors.append((aug_t, torch.tensor(score, device=device, dtype=dtype)))
        return out_tensors

    def cross_slr_image(self, A_tensor, B_tensor, S_LA, S_RA, S_LB, S_RB):
        """
        A_tensor, B_tensor: torch.Tensor (C,H,W)
        returns list of (aug_image_tensor, score)
        """
        device = A_tensor.device
        dtype = A_tensor.dtype

        A_np = A_tensor.detach().cpu().numpy()
        B_np = B_tensor.detach().cpu().numpy()
        if A_np.shape[0] > 1:
            A_np = np.transpose(A_np, (1, 2, 0))
            B_np = np.transpose(B_np, (1, 2, 0))
        else:
            A_np = A_np.squeeze(0)
            B_np = B_np.squeeze(0)

        L_B, R_B = self.segment_lungs(B_tensor)
        L_B_flip = self.flip_region_np(B_np, L_B)
        R_B_flip = self.flip_region_np(B_np, R_B)

        results = []

        # (i) Flipped left lung
        C1 = A_np.copy()
        C1[L_B == 1] = L_B_flip[L_B == 1]
        S_C1 = float(S_LA) + float(S_LB)
        results.append((C1, S_C1))

        # (ii) Flipped right lung
        C2 = A_np.copy()
        C2[R_B == 1] = R_B_flip[R_B == 1]
        S_C2 = float(S_RA) + float(S_RB)
        results.append((C2, S_C2))

        # (iii) Left lung (non-flipped)
        C3 = A_np.copy()
        C3[L_B == 1] = B_np[L_B == 1]
        S_C3 = float(S_LB) + float(S_RA)
        results.append((C3, S_C3))

        # (iv) Right lung (non-flipped)
        C4 = A_np.copy()
        C4[R_B == 1] = B_np[R_B == 1]
        S_C4 = float(S_LA) + float(S_RB)
        results.append((C4, S_C4))

        # (v) Both flipped lungs
        C5 = A_np.copy()
        both_mask = np.logical_or(L_B, R_B)
        combined_flip = C5 * 0
        combined_flip[L_B == 1] = L_B_flip[L_B == 1]
        combined_flip[R_B == 1] = R_B_flip[R_B == 1]
        C5[both_mask == 1] = combined_flip[both_mask == 1]
        S_C5 = float(S_LB) + float(S_RB)
        results.append((C5, S_C5))

        # convert to tensors
        out = []
        for img_np, sc in results:
            if img_np.ndim == 2:
                img_np = np.expand_dims(img_np, axis=-1)
            t = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).to(device=device, dtype=dtype)
            out.append((t, torch.tensor(sc, device=device, dtype=dtype)))
        return out

    # Batch wrappers
    def self_slr_batch(self, images, left_scores=None, right_scores=None):
        """
        images: tensor (B,C,H,W)
        left_scores/right_scores: tensors (B,) or None
        returns: (aug_images_tensor, aug_scores_tensor)
        """
        device = images.device
        dtype = images.dtype
        B = images.shape[0]

        aug_imgs = []
        aug_scores = []

        for i in range(B):
            img = images[i]
            if left_scores is not None and right_scores is not None:
                S_L = left_scores[i].item()
                S_R = right_scores[i].item()
            else:
                # fallback: split total label equally if user didn't give left/right
                S_total = 0.0
                # try to read total label from attribute provided by caller; if not available assume zeros
                # The training_step will pass left_scores/right_scores if available.
                S_L = S_R = 0.0

            out = self.self_slr_image(img, S_L, S_R)
            for timg, tscore in out:
                aug_imgs.append(timg)
                aug_scores.append(tscore)

        if len(aug_imgs) == 0:
            return torch.empty(0, device=device, dtype=images.dtype), torch.empty(0, device=device, dtype=images.dtype)

        aug_imgs_t = torch.stack(aug_imgs, dim=0)
        aug_scores_t = torch.stack(aug_scores, dim=0).view(-1, 1)
        return aug_imgs_t, aug_scores_t

    def cross_slr_batch(self, images, left_scores=None, right_scores=None, pairs=None):
        """
        Create cross augmentations by sampling pair partners in the batch.
        If pairs is provided, it's an iterable of (i,j) indices to use; otherwise we sample random partners.
        """
        device = images.device
        dtype = images.dtype
        B = images.shape[0]

        aug_imgs = []
        aug_scores = []

        if pairs is None:
            # sample up to B partners (avoid i==j)
            pairs = []
            for i in range(B):
                # choose a random j different from i
                j = random.choice([k for k in range(B) if k != i])
                pairs.append((i, j))

        for (i, j) in pairs:
            A = images[i]
            Bimg = images[j]
            if left_scores is not None and right_scores is not None:
                S_LA = left_scores[i].item(); S_RA = right_scores[i].item()
                S_LB = left_scores[j].item(); S_RB = right_scores[j].item()
            else:
                S_LA = S_RA = S_LB = S_RB = 0.0

            out = self.cross_slr_image(A, Bimg, S_LA, S_RA, S_LB, S_RB)
            for timg, tscore in out:
                aug_imgs.append(timg)
                aug_scores.append(tscore)

        if len(aug_imgs) == 0:
            return torch.empty(0, device=device, dtype=images.dtype), torch.empty(0, device=device, dtype=images.dtype)

        aug_imgs_t = torch.stack(aug_imgs, dim=0)
        aug_scores_t = torch.stack(aug_scores, dim=0).view(-1, 1)
        return aug_imgs_t, aug_scores_t

    # -------------------------
    # Forward (feature extraction & head)
    # -------------------------
    def encode_quadrant(self, model, x):
        """Run encoder model and ensure (B,C,Hf,Wf), plus adapt channels if needed."""
        feat = None
        if hasattr(model, 'forward_features'):
            feat = model.forward_features(x)
        else:
            feat = model.forward(x)
        feat = ensure_4d_feature(feat)
        # lazy adapter
        if self.adapter is None and feat.shape[1] != self.embed_channels:
            self.adapter = nn.Conv2d(feat.shape[1], self.embed_channels, kernel_size=1).to(feat.device)
        if self.adapter is not None:
            feat = self.adapter(feat)
        return feat

    def forward(self, images):
        """
        images: (B, C, H, W)
        """
        B = images.size(0)
        # split quadrants
        i1, i2, i3, i4 = self.split(images)

        f1 = self.encode_quadrant(self.encoder1, i1)
        f2 = self.encode_quadrant(self.encoder2, i2)
        f3 = self.encode_quadrant(self.encoder3, i3)
        f4 = self.encode_quadrant(self.encoder4, i4)

        # apply channel attention
        z1 = self.ca(f1)
        z2 = self.ca(f2)
        z3 = self.ca(f3)
        z4 = self.ca(f4)

        # ensure spatial dims equal
        ref_h, ref_w = z1.shape[2], z1.shape[3]

        def resize_to_ref(t):
            if t.shape[2] != ref_h or t.shape[3] != ref_w:
                return F.interpolate(t, size=(ref_h, ref_w), mode='bilinear', align_corners=False)
            return t

        z2 = resize_to_ref(z2)
        z3 = resize_to_ref(z3)
        z4 = resize_to_ref(z4)

        top = torch.cat([z1, z2], dim=3)
        bottom = torch.cat([z3, z4], dim=3)
        combined = torch.cat([top, bottom], dim=2)

        # gated block + spatial attention
        gated = self.gated_block(combined)
        spatial = self.spatial_att(gated)

        pooled = self.global_pool(spatial).view(spatial.shape[0], -1)
        out = self.fc(pooled)
        return out

    # -------------------------
    # Training / validation
    # -------------------------
    def training_step(self, batch, batch_idx):
        """
        Expect batch to be either:
         - (imgs, total_labels) or
         - (imgs, total_labels, left_labels, right_labels)
        imgs: (B,C,H,W), labels: (B,1) or (B,)
        """
        if len(batch) == 2:
            imgs, labels = batch
            left_labels = None
            right_labels = None
        elif len(batch) >= 4:
            imgs, labels, left_labels, right_labels = batch[:4]
        else:
            raise ValueError("Unexpected batch format. Provide (imgs, labels) or (imgs, labels, Llabel, Rlabel).")

        device = imgs.device
        dtype = imgs.dtype

        # Convert total labels to shape (B,1)
        labels = labels.view(-1, 1).to(device=device, dtype=dtype)

        # Generate Self-SLR augmentations (if seg_model provided)
        if self.seg_model is not None:
            aug_imgs_self, aug_scores_self = self.self_slr_batch(imgs, left_scores=left_labels, right_scores=right_labels)
            aug_imgs_cross, aug_scores_cross = self.cross_slr_batch(imgs, left_scores=left_labels, right_scores=right_labels)
        else:
            aug_imgs_self = torch.empty(0, device=device, dtype=imgs.dtype)
            aug_scores_self = torch.empty(0, device=device, dtype=imgs.dtype)
            aug_imgs_cross = torch.empty(0, device=device, dtype=imgs.dtype)
            aug_scores_cross = torch.empty(0, device=device, dtype=imgs.dtype)

        # If there are no augmentations, just use the original batch
        parts_imgs = [imgs]
        parts_labels = [labels]

        if aug_imgs_self.numel() > 0:
            parts_imgs.append(aug_imgs_self)
            parts_labels.append(aug_scores_self)
        if aug_imgs_cross.numel() > 0:
            parts_imgs.append(aug_imgs_cross)
            parts_labels.append(aug_scores_cross)

        imgs_all = torch.cat(parts_imgs, dim=0)
        labels_all = torch.cat(parts_labels, dim=0)

        preds = self.forward(imgs_all)  # (N,1)
        loss = self.loss_func(preds, labels_all)

        mae = metrics.mean_absolute_error(preds.view(-1), labels_all.view(-1))
        pc = metrics.pearson_corrcoef(preds.view(-1), labels_all.view(-1))

        self.log('Loss/Train', loss, prog_bar=True, on_epoch=True)
        self.log('MAE/Train', mae, prog_bar=True, on_epoch=True)
        self.log('PC/Train', pc, prog_bar=True, on_epoch=True)

        return {'loss': loss, 'mae': mae, 'pc': pc}

    def validation_step(self, batch, batch_idx):
        images, labels = batch[:2]
        device = images.device
        dtype = images.dtype
        labels = labels.view(-1, 1).to(device=device, dtype=dtype)

        output = self.forward(images)
        loss = self.loss_func(output, labels)
        mae = metrics.mean_absolute_error(output.view(-1), labels.view(-1))
        pc = metrics.pearson_corrcoef(output.view(-1), labels.view(-1))
        self.log('Loss/Val', loss, prog_bar=True, on_epoch=True)
        self.log('MAE/Val', mae, prog_bar=True, on_epoch=True)
        self.log('PC/Val', pc, prog_bar=True, on_epoch=True)
        return {'loss': loss, 'mae': mae, 'pc': pc}

    def test_step(self, batch, batch_idx):
        images, labels = batch[:2]
        device = images.device
        dtype = images.dtype
        labels = labels.view(-1, 1).to(device=device, dtype=dtype)

        output = self.forward(images)
        loss = self.loss_func(output, labels)
        mae = metrics.mean_absolute_error(output.view(-1), labels.view(-1))
        pc = metrics.pearson_corrcoef(output.view(-1), labels.view(-1))
        self.log('Loss/Test', loss, prog_bar=True, on_epoch=True)
        self.log('MAE/Test', mae, prog_bar=True, on_epoch=True)
        self.log('PC/Test', pc, prog_bar=True, on_epoch=True)
        return {'loss': loss, 'mae': mae, 'pc': pc}
    	
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.5, 0.99))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.trainingsize, T_mult=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": 'Loss/Val', "interval": 'epoch'}
