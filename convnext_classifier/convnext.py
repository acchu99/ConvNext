# IMPORTS
import timm
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# custom imports
from convnext.methods import get_device, save_model_state
from convnext.engine import train_one_epoch, evaluate_roc
from convnext.dataset import CustomImageDataset
from convnext.model import ConvNeXtModelOfficial
from convnext.session import Session


# MAIN FUNCTION
def run_convnext(cur_model, epochs):
    ses = Session(cur_model=cur_model, epochs=epochs)

    # SESSION ID (UID)
    UID = f"{ses.get_sessid()}"
    
    print(f"MODEL: Convnext-{cur_model}")

    # CONFIGS (subject to change)
    CUR_MODEL, BATCH_SIZE, NUM_WORKERS, EPOCHS = ses.get_config()

    # CONSTANTS (will not change)
    CLASS_MAPPING, MODELS, NUM_LABELS = ses.get_consts()

    # MACHINE CONSTANTS
    DEVICE, DEVICE_IDS = get_device()
    
    # 1. Read train and test dataframe
    df = pd.read_csv("path_class.csv")
    df_train = pd.read_csv("datasets/chestxrays14/train_val_list.txt", sep=" ", header=None, names=["filename"])
    df_test = pd.read_csv("datasets/chestxrays14/test_list.txt", sep=" ", header=None, names=["filename"])
    df_train = df_train.merge(df, on="filename", how="left")
    df_test = df_test.merge(df, on="filename", how="left")

    # 2.1. Create model
    model=ConvNeXtModelOfficial(
        n_labels=15, 
        model_name=MODELS[CUR_MODEL]
    )

    # 2.2. Freeze hidden layers
    for param in model.convnext_model.parameters():
        param.requires_grad = False
    for param in model.convnext_model.head.parameters():
        param.requires_grad = True
    for param in model.sigmoid.parameters():
        param.requires_grad = True

    # 2.3. Use the transformer provided with the model for data augmentation
    data_config = timm.data.resolve_model_data_config(model.convnext_model)
    transforms_train = timm.data.create_transform(**data_config, is_training=True)
    transforms_test = timm.data.create_transform(**data_config, is_training=False)

    # 3. Initialize loss function and optimizer and enable data parallelism across the GPU(s)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = torch.nn.DataParallel(model, device_ids=DEVICE_IDS)
    model = model.to(DEVICE)

    # 4. Prepare the dataset for training
    train_dataloader = torch.utils.data.DataLoader(
        dataset=CustomImageDataset(df_train, transform=transforms_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=CustomImageDataset(df_test, transform=transforms_test),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # 5.1. Training the model
    best_test_loss = [float('inf'), None, None]
    
    for epoch in tqdm(range(EPOCHS), ncols = 100, desc ="MODEL TRAINING PROGRESS"):
        # 5.1.1 Train model
        metrics = train_one_epoch(model, train_dataloader, test_dataloader, NUM_LABELS, DEVICE, loss_fn, optimizer)
        metrics['epoch'] = epoch+1

        # 5.1.2 Check if the current epoch's validation loss is the lowest and update the variable best_test_loss
        if metrics['test']['loss'] < best_test_loss[0]:
            best_test_loss[0], best_test_loss[1], best_test_loss[2] = metrics['test']['loss'], metrics, epoch+1

        # 5.1.3 Save model state along with the loss and auc scores
        save_model_state(model, metrics, f"outputs/{CUR_MODEL}{UID}_epoch_{epoch+1}.pth")

    
    # 5.2. Save best epoch score
    print(f"Best validation loss: {best_test_loss[0]} (epoch: {best_test_loss[2]})")

    # 6. Evaluate model and get the fpr & tpr and plot the ROC
    fpr, tpr = evaluate_roc(model, test_dataloader, DEVICE)
    plt.figure(figsize=(8, 6))
    for fp, tp in zip(fpr, tpr):
        plt.plot(fp, tp)
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title(MODELS[CUR_MODEL])
    plt.legend(CLASS_MAPPING.values())

    plt.savefig(f"outputs/{CUR_MODEL}_{UID}.jpg")
    
    return UID, best_test_loss[1]
    

if __name__=="__main__":
    MODEL = "small"
    EPOCHS = 10
    
    uid, best_metrics = run_convnext(MODEL, EPOCHS)
