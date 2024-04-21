import numpy as np
import matplotlib.pyplot as plt

def split_train_val_test_data(inputs, outputs, train_ratio = 0.8, val_ratio = 0.1, test_ratio = 0.1, shuffle = False):

    '''
    Function to split the data into training, validation and test datasets

    Parameters:
    inputs: array_like
        Inputs of the data
    outputs: array_like
        Outputs of the data
    train_ratio: float
        Ratio of the training data
    val_ratio: float
        Ratio of the validation data
    test_ratio: float
        Ratio of the test data
    shuffle: bool
        Shuffle the data before splitting
    '''

    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("The sum of the ratios must be equal to 1")

    train_idx = int((train_ratio)*len(inputs))
    val_idx   = int((train_ratio + val_ratio)*len(inputs))
    test_idx  = int((train_ratio + val_ratio + test_ratio)*len(inputs))

    train_idxs = np.arange(0, train_idx)
    val_idxs = np.arange(train_idx, val_idx)
    test_idxs = np.arange(val_idx, test_idx)

    if shuffle:
        idx = np.random.permutation(len(inputs))
        inputs = inputs[idx]
        outputs = outputs[idx]

    # train, validation and test data
    train_in, train_out = inputs[train_idxs], outputs[train_idxs]
    val_in, val_out  = inputs[val_idxs], outputs[val_idxs]
    test_in, test_out = inputs[test_idxs], outputs[test_idxs]

    return train_in, train_out, val_in, val_out, test_in, test_out


def convert_to_real_loss(loss, norm_scale):
    '''
    Function to convert the trainiment loss to the real loss, considering the normalization of the data in FrequencyCombDataset

    Parameters:
    loss: array_like
        Training loss
    norm_scale: float
        Normalization scale of the data

    '''
    loss = np.array(loss)
    loss = loss * (2 * norm_scale)**2
    return loss.squeeze()


def plot_training_progress(train_losses, val_losses, title = "Training and Validation Losses", ylabel = "Loss", average_curves = False, M = 200):

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    plt.figure(figsize=(15,5))
    plt.plot(train_losses, label=f'Training Loss: {train_losses[-1]:.4f}', color = "C0")
    plt.plot(val_losses, label=f'Validation Loss: {val_losses[-1]:.4f}', color='C3')

    if train_losses.size > M and average_curves:
        def moving_average(x, w):
            return np.convolve(x, np.ones(w), 'valid')/w
        plt.plot(moving_average(train_losses, M), color='blue', label='Training Loss (Moving Average)')
        plt.plot(moving_average(val_losses, M), color='red', label='Validation Loss (Moving Average)')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(which='both', alpha=0.5)
    plt.minorticks_on()
    plt.show()

def get_num_params(model):
    '''
    Function to calculate the number of parameters of a neural network model from the torch model

    Parameters:
    model: torch model
        Neural network model

    Returns:
    n_params: int
        Number of parameters of the neural network model
    '''
    n_params = 0
    for param in model.parameters():
        n_params += param.numel()
    return n_params

def calc_num_params(architecture):
    '''
    Function to calculate the number of parameters of a neural network model from the architecture list

    Formula: for architecture of [I, H1, H2, O], the number of parameters of the model is = I*(H1+1) + H1*(H2+1) + H2*(O+1)

    Parameters:
    architecture: list
        Architecture of the neural network model

    Returns:
    n_params: int
        Number of parameters of the neural network model
    '''
    n_params = 0
    for i in range(len(architecture) - 1):
        n_params += architecture[i]*architecture[i + 1] + architecture[i + 1]
    return n_params

def get_architecture(loaded_dict_data):
    '''
    Get the architecture of the model from the loaded dictionary data
    
    Parameters:
    loaded_dict_data: dictionary with the model data

    Returns:
    architecture: list with the architecture of the model
    '''
    architecture = []
    for key in loaded_dict_data.keys():
        if "weight" in key:
            architecture.append(loaded_dict_data[key].shape[1])
    architecture.append(loaded_dict_data[key].shape[0])
    return architecture


def plot_comparison(target, output, freqs_GHz, loss, figname, title, ylim = (-35,35), xlabel = "Frequency (GHz) - Normalized to symbol rate", ylabel = "Power Spectral Density (dB/Hz)"):
    with plt.style.context(['science', 'ieee', "grid", 'no-latex']):
        fig, ax = plt.subplots()
        ax.plot(freqs_GHz, target, "s", label='Target')
        ax.plot(freqs_GHz, output, "o", label='Predicted')
        ax.legend()
        ax.autoscale(tight=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(freqs_GHz)
        ax.set_title(title)
        ax.set_xlim(freqs_GHz[0]-0.5,freqs_GHz[-1]+0.5)
        ax.set_ylim(ylim)
        ax.text(0, ylim[0]*0.88, f"AVG Loss: {loss:.3f} (dB/Hz)^2\nMax - Min: {np.max(output) - np.min(output):.3f} dB", ha = 'center', bbox=dict(facecolor='white', alpha=1, edgecolor='silver', boxstyle='round,pad=0.3'))
        fig.savefig(figname, dpi=300)
        plt.show()
        plt.close()


def run_one_epoch_forward(mode, loader, model, loss_fn, device="cpu", optimizer=None):
    if mode == 'train':
        model.train()
    elif mode == 'val' or mode == "test":
        model.eval()
    else:
        raise ValueError("Invalide mode. Try to use 'train', 'val' or 'test'.")

    total_loss = 0.0
    n_loops = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs) # Calculate outputs
        loss = loss_fn(outputs, targets) # Calculate loss
        total_loss += loss.item()

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        n_loops += 1

    avg_loss = total_loss / n_loops
    return avg_loss, outputs, targets

def run_one_epoch_inverse(mode, loader, forward_model, inverse_model, loss_fn, device="cpu", optimizer=None):
    if mode == 'train':
        inverse_model.train()
    elif mode == 'val' or mode == "test":
        inverse_model.eval()
    else:
        raise ValueError("Invalide mode. Try to use 'train', 'val' or 'test'.")

    total_loss = 0.0
    n_loops = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        inverse_outputs = inverse_model(targets) # Forward pass through the inverse model
        forward_outputs = forward_model(inverse_outputs) # Forward pass through the forward model

        # Calculate loss
        loss = loss_fn(forward_outputs, targets)
        total_loss += loss.item()

        if mode == 'train':
            optimizer.zero_grad()  # Reset gradients tensors
            loss.backward()  # Calculate gradients
            optimizer.step()  # Update weights

        n_loops += 1

    avg_loss = total_loss / n_loops
    return avg_loss, forward_outputs, inverse_outputs, targets, inputs