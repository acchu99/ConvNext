import torch
from torchmetrics.classification import MultilabelAUROC, ROC
from tqdm import tqdm


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_step(model, train_dataloader, num_labels, device, loss_fn, optimizer):
    auroc_micro = MultilabelAUROC(num_labels=num_labels, average="micro", thresholds=None)
    auroc_macro = MultilabelAUROC(num_labels=num_labels, average="macro", thresholds=None)

    # Train Step
    model.train()
    total_loss = 0
    
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        
        model.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        current_loss = loss.item()
        total_loss += current_loss
    
    loss = total_loss/len(train_dataloader)
    micro_auc = auroc_micro(outputs, labels.type(torch.int)).item()
    macro_auc = auroc_macro(outputs, labels.type(torch.int)).item()

    return round(loss, 4), round(micro_auc, 4), round(macro_auc, 4)

def test_step(model, test_dataloader, num_labels, device, loss_fn):
    auroc_micro = MultilabelAUROC(num_labels=num_labels, average="micro", thresholds=None)
    auroc_macro = MultilabelAUROC(num_labels=num_labels, average="macro", thresholds=None)

    # Test Step
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_dataloader:
            torch.cuda.empty_cache()
            y_true += labels.cpu().numpy().tolist()
            y_pred += model(images.to(device)).cpu().numpy().tolist()

    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    loss = loss_fn(y_pred, y_true).item()
    micro_auc = auroc_micro(y_pred, y_true.type(torch.int)).item()
    macro_auc = auroc_macro(y_pred, y_true.type(torch.int)).item()

    return round(loss, 4), round(micro_auc, 4), round(macro_auc, 4)

def evaluate_roc(model, test_dataloader, device):
    model.eval()
    
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader):
            torch.cuda.empty_cache()
            y_true += labels.cpu().numpy().tolist()
            y_pred += model(images.to(device)).cpu().numpy().tolist()

    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    
    roc = ROC(task='multilabel', num_labels=15)
    fpr, tpr, _ = roc(y_pred, y_true.type(torch.int))
    
    return [item.numpy() for item in fpr], [item.numpy() for item in tpr]

def train_one_epoch(model, train_dataloader, test_dataloader, num_labels, device, loss_fn, optimizer):
    train_loss, train_micro_auc, train_macro_auc = train_step(model, train_dataloader, num_labels, device, loss_fn, optimizer)
    test_loss, test_micro_auc, test_macro_auc = test_step(model, test_dataloader, num_labels, device, loss_fn)

    # print(f"TRAIN [ loss={train_loss} ; micro_auc={train_micro_auc} ; macro_auc={train_macro_auc} ]", end=", ")
    # print(f"TEST [ loss={test_loss} ; micro_auc={test_micro_auc} ; macro_auc={test_macro_auc} ]")

    return {
        "train": {
            "loss": train_loss,
            "micro_auc": train_micro_auc,
            "macro_auc": train_macro_auc
        },
        "test": {
            "loss": test_loss,
            "micro_auc": test_micro_auc,
            "macro_auc": test_macro_auc
        }
    }