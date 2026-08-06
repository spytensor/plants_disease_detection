import torch


class DefaultConfigs(object):
    # 1. path parameters
    train_data = "./data/train/"
    test_data = "./data/test/"
    val_data = "no"
    model_name = "resnet50"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"

    # 2. device parameters
    # 用哪些 GPU（逗号分隔），设为 "" 则强制 CPU
    gpus = "0"
    # 运行设备，自动探测，无 GPU 时回退到 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    # 是否启用混合精度（AMP），仅在 CUDA 上生效
    use_amp = True

    # 3. numeric parameters
    epochs = 40
    batch_size = 8
    img_height = 650
    img_width = 650
    num_classes = 59
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4
    # 是否使用 FocalLoss（否则用 CrossEntropyLoss）
    use_focal_loss = False


config = DefaultConfigs()
