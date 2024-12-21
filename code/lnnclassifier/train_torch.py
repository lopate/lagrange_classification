import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian
import torch.func
from torch.func import vmap
from torch import optim
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from tqdm import tqdm
import scipy.linalg
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
import locale
TRAIN_DATASET_PATH = Path("trajectories")
TEST_DATASET_PATH = Path("trajectories")
LOG_DIR = Path("logs")
BATCH_SIZE = 1000
LEARNING_RATE = 1e-3
WEIGHTDECAY = 1e-2
DEGREES_OF_FREEDOM = 3
DIVIDE_BY = 3
EPOCHS = 1000
LAMBERTWFUNCMIN = -0.27846
class LNNDataset(Dataset):
    def __init__(self, features, target):
        super().__init__()
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        X = self.features[idx]
        y = self.target[idx]

        return X, y

class NeuralNetwork(nn.Module):
    def __init__(self, deg_of_freedom = DEGREES_OF_FREEDOM):
        super().__init__()
        n = 100

        self.first_layer = nn.Sequential(
            nn.Linear(2 * deg_of_freedom, 100), 
            nn.Softplus(),
        )

        self.general_layer1 = nn.Sequential(
            nn.Softplus(),
            nn.Linear(100, 100)
        )


        self.general_layer2 = nn.Sequential(
            nn.Softplus(),
            nn.Linear(100, 100)
        )


        self.last_layer = nn.Sequential(
            nn.Linear(100, 1)
        )
        

    def forward(self, x):
        if(len(x.shape) < 2):
            x = x.unsqueeze(0)
        x = self.first_layer(x)
        x =  x + self.general_layer1(x)
        x =  x + self.general_layer2(x)
        l = x + self.last_layer(x)
        return (l).sum()



def train_loop(dataloader, model, loss_fn, optimizer, scaler, deg_of_freedom = DEGREES_OF_FREEDOM, device = torch.device("cpu"), log = LOG_DIR):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    
    for (X, y) in dataloader :
        #print(batch)
        # Compute prediction and loss
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)
        #with torch.autocast(device_type = "cuda"):
        L = model
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss = loss_fn(L, X, y, deg_of_freedom, device)
        #print(loss)
        # Backpropagation
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    return loss.item()


def test_loop(dataloader, model, loss_fn, deg_of_freedom = DEGREES_OF_FREEDOM, device = torch.device("cpu"), log = LOG_DIR):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    test_loss= 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            with torch.autocast(device_type="cuda"):
                X = X.to(device)
                y = y.to(device)
                L = model
                
                test_loss += loss_fn(L, X, y, deg_of_freedom, device).item()
    test_loss /= num_batches
    return test_loss

def get_func_coords(model):
    return model

def get_transformation_matrix_to_q(deg_of_freedom = DEGREES_OF_FREEDOM, device = torch.device("cpu")):
    Aq = torch.eye(deg_of_freedom, 2 * deg_of_freedom, device=device, dtype = torch.float32)
    return Aq
def get_transformation_matrix_to_qt(deg_of_freedom, device = torch.device("cpu")):
    Aq_t = torch.diag(torch.ones(deg_of_freedom, device = device, dtype = torch.float32), diagonal=deg_of_freedom)[:-deg_of_freedom]
    return Aq_t

def get_acc(model, x, deg_of_freedom = DEGREES_OF_FREEDOM, device = torch.device("cpu")):
    x = x.to(device)
    Aq = get_transformation_matrix_to_q(deg_of_freedom, device)
    Aqt = get_transformation_matrix_to_qt(deg_of_freedom, device)
    #print(x)
    #print(model(x))
    grad = jacobian(model , x, create_graph=True, vectorize = True) #полный градиент
    #print(grad)
    #grad_q = torch.einsum('ij, ...j->...i', Aq, grad)
    grad_q = (Aq @ x.T).T
    #print(grad_q)
    H = get_fullhess(model, x, deg_of_freedom, device)
    #print(H)
    #print(Aq.shape)
    #H_qqt = torch.einsum('im, ...mk, kj ->...ij', Aq, H, Aqt.T)
    H_qqt = torch.matmul(torch.matmul(Aq, H), Aqt.T)
    #qtt_pred = grad_q - torch.einsum('...im, mk, ...k ->...i', H_qqt, Aqt, x)
    qtt_pred = grad_q - torch.matmul(H_qqt,torch.matmul(Aqt, x.T).T.unsqueeze(-1)).squeeze(-1)
    return qtt_pred

def get_hess(model, x, deg_of_freedom = DEGREES_OF_FREEDOM, device = torch.device("cpu")):
    x = x.to(device)
    Aqt = get_transformation_matrix_to_qt(deg_of_freedom, device)
    H = get_fullhess(model, x, deg_of_freedom, device)
    #H = torch.zeros([batch_size, 2 * deg_of_freedom,  2 * deg_of_freedom], dtype=torch.float32, device=device)
    #for t in range(batch_size):
        #H[t] = hessian(model, x[t], create_graph=True, vectorize = True)
    #H_qtqt = torch.einsum('im, ...mk, kj ->...ij', Aqt, H, Aqt.T)
    H_qtqt = torch.matmul(torch.matmul(Aqt, H), Aqt.T)
    return H_qtqt

def get_fullhess(model, x, deg_of_freedom = DEGREES_OF_FREEDOM, device = torch.device("cpu")):
    x = x.to(device)
    H = vmap(torch.func.jacrev(torch.func.jacrev(model)), in_dims= 0)(x)
    return H


def loss_fn(L, x, y, deg_of_freedom = DEGREES_OF_FREEDOM, device = torch.device("cpu")):
    gloss_fn = nn.MSELoss()
    pred_qtt = get_acc(L, x, deg_of_freedom, device)
    with torch.autocast(device_type= "cuda", enabled=False):
        H_qtqt = get_hess(L, x, deg_of_freedom, device)
    #print(x)
    #print(y)
    #tranformed_y = torch.einsum('...im, ...m ->...i', H, y)
    tranformed_y = torch.bmm(H_qtqt,y.unsqueeze(-1)).squeeze(-1)
    #print(pred_qtt)
    #print(tranformed_y)
    #det = torch.det(H_qtqt)
    #weak regulerization
    #alpha = 1
    #beta = 1
    #strong regulerization
    #alpha = 5
    #beta = 12
    alpha = 5
    beta = 12
    gamma = 1
    act = nn.SiLU()
    #print(torch.mean(torch.det(H)))
    #print(torch.linalg.eigvalsh(H))
    #print(act(beta*(1 - torch.linalg.eigvalsh(H))))
    #print(torch.exp(alpha * act(beta*(1 - torch.linalg.eigvalsh(H)))))
    #print(torch.exp(alpha * act(beta*(1 - torch.linalg.eigvalsh(H)))) - np.exp(alpha * LAMBERTWFUNCMIN))
    #print(torch.einsum('...i->...',torch.exp(alpha * act(beta*(1 - torch.linalg.eigvalsh(H)))) - np.exp(alpha * LAMBERTWFUNCMIN)))
    
    adj_vect = alpha * act(beta*(1 - torch.linalg.eigvalsh(H_qtqt))) - alpha * LAMBERTWFUNCMIN
    #adj_vect = torch.sum(torch.exp(alpha * act(beta*(1 - torch.linalg.eigvalsh(H_qtqt)))) - np.exp(alpha * LAMBERTWFUNCMIN), dim = -1)
    adj = torch.mean(adj_vect)
    #print(torch.linalg.eigvalsh(H_qtqt))
    #print(torch.mean(torch.einsum('...i->...', torch.exp(alpha * act(beta*(1 - torch.linalg.eigvalsh(H)))) - np.exp(alpha * LAMBERTWFUNCMIN))))
    #torch.mean(torch.exp(alpha * act(beta * (1 - torch.det(H)))))
    general_mse_loss = gloss_fn(torch.linalg.solve(H_qtqt,pred_qtt.float()), y)
    return gloss_fn(pred_qtt, tranformed_y) + adj


def plot_loss(train_losses, test_losses, log_dir, name):
    """
    plot of loss
    """
    k = 0.01
    plt.rcParams['axes.formatter.use_locale'] = True
    plt.rcParams['axes.formatter.use_locale'] = True
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['font.size'] = 18

    train_losses, test_losses = train_losses[int(k*len(test_losses)):], test_losses[int(k*len(test_losses)):]
    plt.figure(figsize=(12, 6))
    locale.setlocale(locale.LC_NUMERIC,"ru_RU.utf8")
    plt.plot(train_losses, label='Обучающая выборка', linestyle = 'solid', color = "black")
    plt.plot(test_losses, label='Валидационная выборка', linestyle = 'dashed', color = "black")
    plt.xlabel("Шаг обучения")
    plt.ylabel("Функция потерь")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(log_dir, name + "_loss.png"), bbox_inches="tight")
    plt.savefig(Path(log_dir, name + "_loss.pdf"), bbox_inches="tight")

def plot_diff(y_real, y_pred, x, log_dir, name, cutoff = None):
    
    plt.rcParams['axes.formatter.use_locale'] = True
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['font.size'] = 18
    fig1 = plt.figure(figsize=(12, 6))
    fig2 = plt.figure(figsize=(12, 6))
    locale.setlocale(locale.LC_NUMERIC,"ru_RU.utf8")
    
    # метод  PCA
    x = x.to("cpu")
    t = np.linspace(1, y_real[:, 0].shape[0], y_real[:, 0].shape[0])
    def HankelMatrix(X, L):
            N = X.shape[0]
            return scipy.linalg.hankel(X[ : N - L + 1], X[N - L : N])
    from sklearn.decomposition import KernelPCA, PCA
    pca = PCA(n_components=3)
    #X = HankelMatrix(y_real.detach().numpy() , 500)
    #print(X.shape)
    #X_PCA = pca.fit_transform(X)
    ax1 = fig1.add_subplot(1, 1, 1)
    line_styles = ['solid', 'dotted', 'dashed', 'dashdot']
    if (cutoff == None):
        for i in range(0, min(y_real.shape[1], 2)):
            ax1.plot(t, y_real[:, i],  linestyle = line_styles[i], color="black", label=f'{i} компонента ускорения')
    else:
        for i in range(0, min(y_real.shape[1], 2)):
            ax1.plot(t[:cutoff], y_real[:cutoff, i],  linestyle = line_styles[i], color="black", label=f'{i} компонента ускорения')

    #ax1.plot(X_PCA[:, 0], X_PCA[:, 1], X_PCA[:, 2], label='Реальное ускорение')
    #ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    #X = HankelMatrix(y_pred.detach().numpy() , 500)
    #print(X.shape)
    #X_PCA = pca.fit_transform(X)
    ax2 = fig2.add_subplot(1, 1, 1)
    if (cutoff == None):
        for i in range(0, min(y_pred.shape[1], 2)):
            ax2.plot(t, y_pred[:, i],  linestyle = line_styles[i], color="black", label=f'{i} компонента ускорения')
    else:
        for i in range(0, min(y_pred.shape[1], 2)):
            ax2.plot(t[:cutoff], y_pred[:cutoff, i],  linestyle = line_styles[i], color="black", label=f'{i} компонента ускорения')
    
    #ax2.plot(X_PCA[:, 0], X_PCA[:, 1], X_PCA[:, 2], label='Предсказанное ускорение')

    ax1.set_xlabel("Шаг по времени")
    ax1.set_ylabel("Ускорение")

    ax2.set_xlabel("Шаг по времени")
    ax2.set_ylabel("Ускорение")

    ax1.set_ylim(torch.min(y_real), torch.max(y_real))
    ax2.set_ylim(torch.min(y_real), torch.max(y_real))

    fig1.legend()
    fig1.tight_layout()
    fig2.legend()
    fig2.tight_layout()
    fig1.savefig(Path(log_dir, name + "_results_a.png"), bbox_inches="tight")
    fig1.savefig(Path(log_dir, name + "_results_a.pdf"), bbox_inches="tight")
    fig2.savefig(Path(log_dir, name + "_results_b.png"), bbox_inches="tight")
    fig2.savefig(Path(log_dir, name + "_results_b.pdf"), bbox_inches="tight")


def train_model(name, training_dataloader, test_dataloader, epochs = 10, deg_of_freedom = DEGREES_OF_FREEDOM, device = torch.device("cpu"), log = LOG_DIR):
    
    model = NeuralNetwork(deg_of_freedom).to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.01,nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    train_losses = []
    test_losses = []
    tepoch = tqdm(range(epochs), desc="Learning", unit="epoch")
    scaler = torch.cuda.amp.GradScaler()
    for t in tepoch:
        train_loss = train_loop(training_dataloader, model, loss_fn, optimizer, scaler, deg_of_freedom, device, log)
        test_loss = test_loop(test_dataloader, model, loss_fn, deg_of_freedom, device, log)
        train_losses += [train_loss]
        test_losses += [test_loss]
        tepoch.set_postfix(train_loss = train_loss, test_loss = test_loss)
    
    """
    x = torch.ones((1, 6), device = device)
    print(x)
    
    

    Aq = get_transformation_matrix_to_q(deg_of_freedom, device)

    q_tt_real = -torch.einsum('ij, ...j->...i', Aq, x)

    
    for param in model.parameters():
        print(param)
    H = get_hess(model, x, deg_of_freedom, device)
    print(H)
    print(torch.linalg.eigvalsh(H))
    print(torch.einsum('...ik, ...k->...i', torch.inverse(H), q_tt_pred))
    print(loss_fn(model, x, q_tt_real, deg_of_freedom, device))
    """
    if log is not None:
        with open(Path(log, f"{name}"), 'wb') as handle:
            torch.save(model.state_dict(), handle)
        x, y_real = test_dataloader.dataset[:]
        y_real = y_real.to(device)
        x = x.to(device)
        y_pred = get_acc(model, x, deg_of_freedom, device)
        Hqtqt = get_hess(model, x, deg_of_freedom, device)

        #print(x)
        #print(y)
        #print("Hqtqt")
        #print(Hqtqt)
        #y_real = torch.einsum('...im, ...m ->...i', Hqtqt, y_real)
        y_pred = torch.linalg.solve(Hqtqt,y_real.unsqueeze(-1)).squeeze(-1)
        y_real = y_real.to(torch.device("cpu")).detach()
        y_pred = y_pred.to(torch.device("cpu")).detach()
        torch.save(y_real, Path(log, name + "_results_real.pt"))
        torch.save(y_pred, Path(log, name + "_results_predict.pt"))
        torch.save(train_losses, Path(log, name + "_train_loss.pt"))
        torch.save(test_losses, Path(log, name + "_test_loss.pt"))
        plot_diff(y_real, y_pred, x, log, name)
        plot_loss(train_losses, test_losses, log,name)
    return model


def get_sample_model(x, deg_of_freedom = DEGREES_OF_FREEDOM, device = torch.device("cpu")):
    Aq = get_transformation_matrix_to_q(deg_of_freedom, device)
    Aqt = get_transformation_matrix_to_qt(deg_of_freedom, device)
    #q = torch.einsum('...ik, ...k->...i', Aq, x)
    q = torch.bmm(Aq, x)
    #qt = torch.einsum('...ik, ...k->...i', Aqt, x)
    qt = torch.bmm(Aqt, x)
    #print(x)
    #print(q)
    #print(qt)
    return (qt*qt - q*q).sum()

def main():
    
    # Set to German locale to get comma decimal separater
    
    #device = torch.device("cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    experiment_name = str(sys.argv[1])
    deg_of_freedom = int(sys.argv[2])
    n = int(sys.argv[3])
    start_idx = int(sys.argv[4])
    epochs = EPOCHS
    batch_size = BATCH_SIZE
    log_dir = LOG_DIR
    for i in range(start_idx, start_idx+n):
        
        name = experiment_name + f'_{i + 1}.pickle'
        
        path = Path(TRAIN_DATASET_PATH, name)

        with open(path, 'rb') as f:
            training_data = pickle.load(f)
            
        with open(path, 'rb') as f:
            test_data = pickle.load(f)
        '''
        L = lambda x: get_sample_model(x, deg_of_freedom)
        Aq = get_transformation_matrix_to_q(deg_of_freedom)
        x_train = 3 * torch.rand((2**8, 2 * deg_of_freedom))
        y_train = -torch.einsum('ij, ...j->...i', Aq, x_train)
        print(x_train)
        print(y_train)
        #print(loss_fn(L, x_train, y_train, 3))

        x_train= x_train.to(torch.device("cpu"))
        
        y_train= y_train.to(torch.device("cpu"))
        training_data = [x_train.detach().numpy(), y_train.detach().numpy()]
        print(nn.CosineSimilarity(dim=1, eps =1e-6)(x_train, x_train))
        x_test = 3 * torch.rand((2**5, 2 * deg_of_freedom))
        y_test = -torch.einsum('ij, ...j->...i', Aq, x_test)
        x_test = x_test.to(torch.device("cpu"))
        y_test = y_test.to(torch.device("cpu"))
        print(x_test.shape)
        print(y_test.shape)
        test_data = [x_test.detach().numpy(), y_test.detach().numpy()]
        '''
        acc_scaler = sklearn.preprocessing.MaxAbsScaler()
        training_data[1] = acc_scaler.fit_transform(training_data[1])
        test_data[1] = acc_scaler.transform(test_data[1])

        vel_scaler = sklearn.preprocessing.MaxAbsScaler()
        training_data[0] = vel_scaler.fit_transform(training_data[0])
        test_data[0] = vel_scaler.transform(test_data[0])

        training_data[0], training_data[1] = torch.FloatTensor(training_data[0]), torch.FloatTensor(training_data[1])
        test_data[0], test_data[1] = torch.FloatTensor(test_data[0]), torch.FloatTensor(test_data[1])

        training_dataset = LNNDataset(training_data[0], training_data[1])
        test_dataset = LNNDataset(test_data[0], test_data[1])
        
        #print(training_dataset.features)
        
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        train_model(name, training_dataloader, test_dataloader, epochs, deg_of_freedom, device, log_dir)
        
        print(f"======== {name} ========")
        
        
    print("======== done! =========")

if __name__ == '__main__':
    main()
