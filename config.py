class DefaultConfigs(object):
    #1.string parameters
    train_data = "./data/train/"
    test_data = "./data/test/"
    val_data = "no"
    model_name = "resnet50"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "1"

    #2.numeric parameters
    epochs = 40
    batch_size = 8
    img_height = 650
    img_weight = 650
    num_classes = 59
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4

config = DefaultConfigs()
