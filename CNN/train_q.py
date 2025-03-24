import torch
import torch.nn as nn
import copy
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from model import CNNModel
from model_quantized import CNNModel_LSQ
from torchsummary import summary
from dataset import train_loader, val_loader
from utils import get_lr, loss_epoch
import matplotlib.pyplot as plt
import torch.quantization

cnn_model = CNNModel()

# define computation hardware approach (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = cnn_model.to(device)

model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")  # Default QAT config
torch.quantization.prepare_qat(model, inplace=True)  # Prepare model for QAT

loss_func = nn.NLLLoss(reduction="sum")
opt = optim.Adam(cnn_model.parameters(), lr=3e-4)

def train_val(model, params,verbose=False):
    
    # Get the parameters
    epochs=params["epochs"]
    loss_func=params["f_loss"]
    opt=params["optimiser"]
    train_dl=params["train"]
    val_dl=params["val"]
    lr_scheduler=params["lr_change"]
    weight_path=params["weight_path"]
    
    loss_history={"train": [],"val": []} # history of loss values in each epoch
    metric_history={"train": [],"val": []} # histroy of metric values in each epoch
    best_model_wts = copy.deepcopy(model.state_dict()) # a deep copy of weights for the best performing model
    best_loss=float('inf') # initialize best loss to a large value
    
    ''' Train Model n_epochs '''
    
    for epoch in tqdm(range(epochs)):
        
        ''' Get the Learning Rate '''
        current_lr=get_lr(opt)
        if(verbose):
            print('Epoch {}/{}, current lr={}'.format(epoch, epochs - 1, current_lr))
        
        '''
        
        Train Model Process
        
        '''
        
        model.train()
        train_loss, train_metric = loss_epoch(model,loss_func,train_dl,opt, device=device)

        # collect losses
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        
        '''
        
        Evaluate Model Process
        
        '''
        
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model,loss_func,val_dl,device=device)
        
        # store best model
        if(val_loss < best_loss):
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
            # store weights into a local file
            torch.save(model.state_dict(), weight_path)
            if(verbose):
                print("Copied best model weights!")
        
        # collect loss and metric for validation dataset
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        # learning rate schedule
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            if(verbose):
                print("Loading best model weights!")
            model.load_state_dict(best_model_wts) 

        if(verbose):
            print(f"train loss: {train_loss:.6f}, dev loss: {val_loss:.6f}, accuracy: {100*val_metric:.2f}")
            print("-"*10) 

    # load best model weights
    model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history

params_train = {
    "train": train_loader, "val": val_loader,
    "epochs": 50,
    "optimiser": opt,
    "lr_change": ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=0),
    "f_loss": loss_func,
    "weight_path": "weights_best_qat.pt",
}

model, loss_hist, metric_hist = train_val(model, params_train, verbose=True)

cnn_model.eval()
cnn_model = torch.quantization.convert(cnn_model, inplace=True)

# Save the quantized model
torch.save(cnn_model.state_dict(), "quantized_model.pt")

print("Quantization-aware training complete. Model saved as 'quantized_model.pt'.")

