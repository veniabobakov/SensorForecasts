import torch
import torch.optim as optim
import numpy as np
import os
import time
import gc
import yaml
import pickle
from contextlib import redirect_stdout
from utils.utils import *
from utils.data_utils import get_dataloaders, get_dataset_dims
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.LSTNet import LSTNet

#load experiment configs
with open('/Users/user/PycharmProjects/SensorForecasts/Experiment_config.yaml', 'r') as f:
    config = list(yaml.load_all(f,  Loader=yaml.SafeLoader))[0]
    
def train(model, model_type = ""):
    model.train()
    batch_losses = [] 

    for batch in train_loader:      
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        if config["features"] == "single":
            x_batch = x_batch.unsqueeze(-1)
            y_batch = y_batch.unsqueeze(1)
      
        # Make predictions
        y_hat = model(x_batch)        
       
        # Computes loss
        loss = model.loss(y_hat, y_batch)
        
        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()

        batch_losses.append(loss.item())
    
    return batch_losses 

def val(model, model_type = ""):
    model.eval()
    batch_val_losses = []    

    for batch in val_loader:
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        if config["features"] == "single":
            x_batch = x_batch.unsqueeze(-1)
            y_batch = y_batch.unsqueeze(1)
        

        y_hat = model(x_batch)

        # Computes loss
        loss = model.loss(y_hat, y_batch)
        
        batch_val_losses.append(loss.item())
        
    return np.mean(batch_val_losses)

def test(test_loader, load_state = True, model_loc = "", return_graphs = False):
    if load_state:
        model.load_state_dict(torch.load(model_loc))
      
    with torch.no_grad():
        model.eval()
        
        predictions = []
        values = []
        
        for batch in test_loader:
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            if config["features"] == "single":
                x_batch = x_batch.unsqueeze(-1)
                y_batch = y_batch.unsqueeze(1)

            y_hat = model(x_batch)
            
            y_hat = y_hat.cpu().detach().numpy()
            predictions.append(y_hat)
            values.append(y_batch.cpu().detach().numpy())

    return predictions, values

#Recording args
model_type = "LSTNet"
output_dir = config["output_dir"]
experiment_number = None #Leave as None to autogenerate a new folder. Change to write into a specific folder
save_dir, model_dir = create_directories(model_type, output_dir, experiment_number)
print(save_dir)

#Dataset args    
input_dim, output_dim = get_dataset_dims(config["dataset"],config["features"])

#Load data
train_loader, val_loader, test_loader, test_loader_one, scaler, col_names = get_dataloaders(config["dataset"], seq_len = config["seq_len"],
                                                                                 horizon = config["horizon"], features = config["features"], 
                                                                                 cut = config["cut"])

#Model Args 
args = {
    "hidCNN" : 50,
    "hidRNN" : 50,
    "CNN_kernel" : 5,

    "clip" : 10,
    "dropout" : 0.2,

    "hidSkip" : 5,
    "normalize" : 2,
    "output_fun" : "None",

    "seq_len" : config["seq_len"],
    "input_dim" : input_dim,
    "highway_window" : 24,
    "skip" : 24,
    "horizon" : 1,
}

loss = torch.nn.MSELoss()
 
#Training args
train_losses = [] 
val_losses = []
val_epoch = [] 
metrics_last = {}
metrics_best = {}
results_last = []
results_best = [] 

#Save args
with open(save_dir + model_type + "_args.yaml", 'w') as f:
    yaml.dump(args, f, sort_keys=False, default_flow_style=False)
    
with open(save_dir + "Experiment_config.yaml", "w") as f:
    yaml.dump(config, f, sort_keys=False, default_flow_style=False)
    
#Training
did_nan = False 
n_epochs = config["n_epochs"]
for i in range(0, config["runs"]):
    print("Run " + str(i))
    best_val = float('inf')
    
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    model = LSTNet(loss, args).double().to(device)
    optimizer = optim.Adam(model.parameters())
    
    train_time = []    
    for epoch in range(n_epochs):
        t0 = time.time()
        batch_losses = train(model)
        t1 = time.time()
        
        train_losses.append(np.mean(batch_losses))
        train_time.append(t1-t0)

        if epoch % config["val_interval"] == 0:
            val_losses.append(val(model))
            val_epoch.append(epoch)
            print(
              f"[{epoch}/{n_epochs}] Training loss: {train_losses[-1]:.4f}\t Validation loss: {val_losses[-1]:.4f} \t Time: {t1-t0:.2f}"
          )
            if val_losses[-1] <= best_val and not np.isnan(val_losses[-1]):
                best_val = val_losses[-1]
                torch.save(model.state_dict(), model_dir + "best_run" + str(i) + ".pt")   
        else:
            print(f"[{epoch}/{n_epochs}] Training loss: {train_losses[-1]:.4f} \t Time: {t1-t0:.2f}")


    torch.save(model.state_dict(), model_dir + "last_run" + str(i) + ".pt")

    print("Last model this run: ")
    t0 = time.time()
    predictions, values = test(test_loader_one, load_state=False)
    t1 = time.time()
    inf_time = t1 - t0
    metrics_last, df_results_last = metrics(predictions, values, metrics_best, scaler, test_loader_one.dataset.start,
                                            config["features"], train_time, inf_time)


    results_last.append(df_results_last)

    print("")
    print("Best model this run: ")
    t0 = time.time()
    predictions, values = test(test_loader_one, load_state=True, model_loc=model_dir + "best_run" + str(i) + ".pt")
    t1 = time.time()
    inf_time = t1 - t0
    metrics_best, df_results_best = metrics(predictions, values, metrics_best, scaler, test_loader_one.dataset.start,
                                            config["features"], train_time, inf_time)

    fig, axs = plt.subplots(9, 1, figsize=(20, 50))

    plt.rcParams.update({
        'font.size': 20,  # Размер шрифта для текста по умолчанию
        'axes.labelsize': 20,  # Размер шрифта для меток осей
        'axes.titlesize': 20,  # Размер шрифта для заголовков осей
        'xtick.labelsize': 15,  # Размер шрифта для меток по оси X
        'ytick.labelsize': 20,  # Размер шрифта для меток по оси Y
        'legend.fontsize': 15,  # Размер шрифта для легенды
    })

    rus = ['CO2 на высоте влажный', 'CH4 на высоте влажный', 'CO на высоте', 'H2O на высоте', 'Температура',
           'Влажность', 'Давление', 'Скорость ветра', 'Направление ветра']
    mesures = ['z_score PPM', 'z_score PPM', 'z_score PPM', 'z_score PPM', 'z_score Градусы цельсия',
               'z_score Проценты', 'z_score 1e2 паскаль', 'z_score Метр/секунду', 'z_score Градусы']

    # Отображение изображения
    for wind in range(output_dim):
        axs[wind].plot([value[0][0][wind] for value in values], label='Observed values', color='black')
        axs[wind].plot([prediction[0][0][wind] for prediction in predictions], label='Predicted values', color='red')
        axs[wind].set_title(f'{rus[wind]}')
        axs[wind].legend()
        axs[wind].set_xlabel('Time')
        axs[wind].set_ylabel(mesures[wind])
    # Показать график
    plt.tight_layout()

    # Настройка заголовков и меток

    # Отображение
    plt.savefig(save_dir + f'Run {i}_best_model.png')

    results_last.append(df_results_last)

    results_best.append(df_results_best)
    print("")

print("DONE")
print(model_type)
print(config["dataset"])
print("horizon " + str(config["horizon"]))

print("\nBest Models Aggregate:")
print_metrics(metrics_best)
print("\nLast Models Aggregate:")
print_metrics(metrics_last)

plt.close()

with open(save_dir + 'out.txt', 'w') as f:
    with redirect_stdout(f):
        print(model_type)
        print(config["dataset"])
        print("horizon " + str(config["horizon"]))

        print("\nBest Models Aggregate:")
        print_metrics(metrics_best)
        print("\nLast Models Aggregate:")
        print_metrics(metrics_last)

with open(save_dir + model_type + "_best.pickle", 'wb') as handle:
    pickle.dump(results_best, handle)

with open(save_dir + model_type + "_last.pickle", 'wb') as handle:
    pickle.dump(results_last, handle)
