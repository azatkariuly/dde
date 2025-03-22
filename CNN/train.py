import torch
import torch.nn as nn
import copy
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
# from model import CNNModel
from model_quantized import CNNModel_Q
from torchsummary import summary
from dataset import train_loader, val_loader
from utils import get_lr, loss_epoch
import matplotlib.pyplot as plt

def save_quantized_model(model, path="quantized_model.pt"):
    quantized_state_dict = {}
    
    for name, param in model.state_dict().items():
        if "step_size" in name:
            # Keep step size in float32 to avoid issues during reloading
            quantized_state_dict[name] = param.float()
        else:
            # Convert weights to int8 representation
            quantized_state_dict[name] = param.to(torch.int8)
    
    torch.save(quantized_state_dict, path)
    print(f"Quantized model saved at {path}")

params_model = {
        "shape_in": (3, 480, 480), 
        "initial_filters": 8,    
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 2,
        "nbits": 8,
}

cnn_model = CNNModel_Q(params_model)

# define computation hardware approach (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = cnn_model.to(device)

summary(cnn_model, input_size=(3, 480, 480),device=device.type)

loss_func = nn.NLLLoss(reduction="sum")

opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)

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
        train_loss, train_metric = loss_epoch(model,loss_func,train_dl,opt)

        # collect losses
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        
        '''
        
        Evaluate Model Process
        
        '''
        
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model,loss_func,val_dl)
        
        # store best model
        if(val_loss < best_loss):
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
            # store weights into a local file
            # torch.save(model.state_dict(), weight_path)
            save_quantized_model(model, "quantized_model.pt")
            if(verbose):
                print("Copied best model weights!")
        
        # collect loss and metric for validation dataset
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        # # learning rate schedule
        # lr_scheduler.step(val_loss)
        # if current_lr != get_lr(opt):
        #     if(verbose):
        #         print("Loading best model weights!")
        #     # model.load_state_dict(best_model_wts) 
        #     # save_quantized_model(model, "quantized_model.pt")

        if(verbose):
            print(f"train loss: {train_loss:.6f}, dev loss: {val_loss:.6f}, accuracy: {100*val_metric:.2f}")
            print("-"*10) 

    # # load best model weights
    # model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history

params_train={
 "train": train_loader,"val": val_loader,
 "epochs": 10,
 "optimiser": optim.Adam(cnn_model.parameters(),lr=3e-4),
 "lr_change": ReduceLROnPlateau(opt,
                                mode='min',
                                factor=0.5,
                                patience=20,
                                verbose=0),
 "f_loss": nn.NLLLoss(reduction="sum"),
 "weight_path": "weights_best_480.pt",
}

''' Actual Train / Evaluation of CNN Model '''
# train and validate the model

cnn_model,loss_hist,metric_hist=train_val(cnn_model,params_train, verbose=True)

import seaborn as sns

# Set the seaborn style
sns.set(style='whitegrid')

# Assuming `params_train`, `loss_hist`, and `metric_hist` are already defined
epochs = params_train["epochs"]

# Create a 1x2 subplot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plotting the loss history
sns.lineplot(x=[*range(1, epochs + 1)], y=loss_hist["train"], ax=ax[0], label='Train Loss')
sns.lineplot(x=[*range(1, epochs + 1)], y=loss_hist["val"], ax=ax[0], label='Validation Loss')
ax[0].set_title('Loss History')  # Title for the first subplot
ax[0].legend()  # Show legend for loss plot

# Plotting the metric history
sns.lineplot(x=[*range(1, epochs + 1)], y=metric_hist["train"], ax=ax[1], label='Train Metric')
sns.lineplot(x=[*range(1, epochs + 1)], y=metric_hist["val"], ax=ax[1], label='Validation Metric')
ax[1].set_title('Metric History')  # Title for the second subplot
ax[1].legend()  # Show legend for metric plot

# Set a common title for the entire figure
fig.suptitle('Convergence History', fontsize=16)

# Show the plot
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Adjust the top space to fit the main title
plt.show()
