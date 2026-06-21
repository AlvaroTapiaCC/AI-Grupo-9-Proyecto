from torch import cuda

# ── Environment ───────────────────────────────────────────────────────────────
device = "cuda" if cuda.is_available() else "cpu"
level  = "easy"          # "easy" | "medium" | "hard"

# ── Run mode ──────────────────────────────────────────────────────────────────
model     = "classifier"   # "classifier" | "detector" | "pipeline"
encode    = False        # precompute features before training
train_new = False         # True → train from scratch | False → load best checkpoint

# ── Training ──────────────────────────────────────────────────────────────────
batch_size = 64
epochs     = 50
lr         = 1e-3

# ── DINOv2 backbone ───────────────────────────────────────────────────────────
dinov2_model    = "dinov2_vitb14"   # vits14 | vitb14 | vitl14 | vitg14
freeze_backbone = True
image_size      = (224, 224)        # input resolution (multiple of patch size 14)

# ── Detector fine-tuning ──────────────────────────────────────────────────────
finetune_dino   = True   # True → unfreeze last DINOv2 blocks during detector training
unfreeze_blocks = 4      # number of transformer blocks to unfreeze (from the end)
backbone_lr     = 1e-5   # LR for unfrozen blocks (much lower than heads)

# ── Detector ──────────────────────────────────────────────────────────────────
max_detections    = 10   # max products per image (verified from dataset)
num_count_classes = 11   # CrossEntropy over 0..10 products
