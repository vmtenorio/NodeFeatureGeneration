from utils import *
import torch

from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GINConv, global_mean_pool

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_synthdata_params(config_name:str):
    path = os.getcwd()
    config_path = path + config_name
    assert os.path.exists(config_path), "Configuration file does not exist."
    with open(config_path,'r') as f:
        params = json.load(f)
    
    # Assert all parameters correct
    assert type(params['seed'])==int, 'Invalid seed.'
    assert type(params['num_nodes'])==int and params['num_nodes']>0, 'Invalid number of nodes.'
    # assert params['edge_prob']>=0 and params['edge_prob']<=1, 'Invalid Erdos-Renyi edge probability.'
    # assert params['in_prob']>=0 and params['in_prob']<=1, 'Invalid within-cluster edge probability.'
    # assert params['out_prob']>=0 and params['out_prob']<=1, 'Invalid across-cluster edge probability.'
    assert type(params['filt_order'])==int and params['filt_order']>0, 'Invalid filter order.'
    # assert type(params['num_blocks'])==int and params['num_blocks']>0, 'Invalid number of blocks.'
    assert type(params['num_samples'])==int and params['num_samples']>0, 'Invalid number of samples.'
    assert type(params['num_classes'])==int and params['num_classes']>0, 'Invalid number of classes.'
    assert type(params['num_trials'])==int and params['num_trials']>0, 'Invalid number of trials.'
    assert type(params['LINK_LAB_SIG'])==bool and type(params['LINK_LAB_STRUC'])==bool and type(params['LINK_SIG_STRUC'])==bool, 'Invalid choice of relationship.'
    return params

def load_model_params(config_name:str):
    path = os.getcwd()
    config_path = path + config_name
    assert os.path.exists(config_path), "Configuration file does not exist."
    with open(config_path,'r') as f:
        params = json.load(f)

    # Assert all parameters correct
    assert type(params['seed'])==int, 'Invalid seed.'
    assert params['train_ratio'] + params['miss_ratio'] + params['val_ratio']<1, 'Invalid data split ratios.'
    assert params['train_ratio']>0 and params['miss_ratio']>0 and params['val_ratio']>0, 'Invalid data split ratios.'
    assert type(params['gae_hid_dim'])==int and params['gae_hid_dim']>0, 'Invalid hidden dimension for GAE.'
    assert type(params['gae_n_layers'])==int and params['gae_n_layers']>0, 'Invalid number of layers for GAE.'
    assert params['gae_lmbda']>=0, 'Invalid weight penalty for GAE.'
    assert type(params['gae_epochs'])==int and params['gae_epochs']>0, 'Invalid number of epochs for GAE.'
    assert params['gae_lr']>0, 'Invalid learning rate for GAE.'
    assert params['gae_gamma']>0 and params['gae_gamma']<1, 'Invalid learning rate decay for GAE.'
    assert type(params['gae_patience'])==int and params['gae_patience']>0, 'Invalid patience length for GAE.'
    assert type(params['gnn_hid_dim'])==int and params['gnn_hid_dim']>0, 'Invalid hidden dimension for GNN.'
    assert params['gnn_lr']>0, 'Invalid learning rate for GNN.'
    assert type(params['gnn_epochs'])==int and params['gnn_epochs']>0, 'Invalid number of epochs for GNN.'
    assert params['gnn_lmbda']>=0, 'Invalid weight penalty for GNN.'
    assert params['gnn_gamma']>0 and params['gnn_gamma']<1, 'Invalid learning rate decay for GNN.'
    assert type(params['gnn_patience'])==int and params['gnn_patience']>0, 'Invalid patience length for GNN.'
    assert type(params['batch_train_size'])==int and params['batch_train_size']>0, 'Invalid training batch size.'
    assert type(params['batch_val_size'])==int and params['batch_val_size']>0, 'Invalid validation batch size.'
    assert type(params['batch_test_size'])==int and params['batch_test_size']>0, 'Invalid testing batch size.'
    assert type(params['k_graph'])==int and params['k_graph']>0, 'Invalid number of graph neighbors.'
    assert type(params['k_node'])==int and params['k_node']>0, 'Invalid number of node neighbors.'
    assert type(params['num_trials'])==int and params['num_trials']>0, 'Invalid number of trials.'
    return params

def batch_inds(list_inds,batch_size:int):
    return list(map(lambda i:list_inds[i],np.random.permutation(len(list_inds))[:batch_size]))

def tud_realdata(data_path:str='data',dataset_name:str='MUTAG',num_gae_feats=21):
    dataset = TUDataset(data_path+'/'+dataset_name, name=dataset_name)

    ret = dict()
    ret['num_classes'] = len(np.unique(list(map(lambda s:int(s.y),dataset))))
    ret['num_data_feats'] = dataset[0].x.shape[1]
    ret['num_samples'] = len(dataset)
    ret['all_num_nodes'] = list(map(lambda s:s.num_nodes,dataset))
    ret['median_num_nodes'] = int(np.median(ret['all_num_nodes']))
    ret['dataset'] = list(map(lambda data:NodeFeatData(N=data.num_nodes,edge_index=data.edge_index,X=data.x,Y=data.y),dataset))

    return ret

def generate_synthdata(num_samples:int, num_feats:int, num_nodes:int, filt_order:int,
                       LINK_LAB_SIG:bool, LINK_LAB_STRUC:bool, LINK_SIG_STRUC:bool):
    Y = torch.tensor(np.random.binomial(1,.5,num_samples))

    if LINK_LAB_SIG:
        X = [np.random.normal((Y[i]*2-1)*.03,1,(num_nodes,num_feats)) for i in range(num_samples)]
    else:
        Z = torch.tensor(np.random.binomial(1,.5,num_samples))
        X = [np.random.normal((Z[i]*2-1)*.04,1,(num_nodes,num_feats)) for i in range(num_samples)]
    
    def W1(N:int=num_nodes):
        u = np.random.rand(num_nodes)
        prob_mat = .7*(u[:,None]*u[None]) + .3*((u[:,None]+u[None])/2)
        return lowtri2mat(np.random.binomial(1,mat2lowtri(prob_mat)))
    def W2(N:int=num_nodes):
        u = np.random.rand(num_nodes)
        prob_mat = .3*(u[:,None]*u[None]) + .7*((u[:,None]+u[None])/2)
        return lowtri2mat(np.random.binomial(1,mat2lowtri(prob_mat)))
        # return lowtri2mat(np.random.binomial(1,mat2lowtri((u[:,None]*u[None])**2)))

    if LINK_LAB_STRUC:
        A = [W1(num_nodes) if not Y[i] else W2(num_nodes) for i in range(num_samples)]
    else:
        Z = torch.tensor(np.random.binomial(1,.5,num_samples))
        A = [W1(num_nodes) if not Z[i] else W2(num_nodes) for i in range(num_samples)]

    if LINK_SIG_STRUC:
        
        filt_taps = np.random.rand(filt_order)*.1
        filt_taps[0] = 1
        filt_taps = filt_taps/np.sum(filt_taps)
        H = [np.sum([filt_taps[l]*np.linalg.matrix_power(A[i],l) for l in range(filt_order)],axis=0) for i in range(num_samples)]
        X = [H[i]@X[i] for i in range(num_samples)]

    to_tensor = lambda x:torch.tensor(x).float()
    A = list(map(to_tensor,A))
    X = list(map(to_tensor,X))

    ret = dict()
    ret['num_classes'] = 2
    ret['num_data_feats'] = num_feats
    ret['num_samples'] = num_samples
    ret['all_num_nodes'] = [num_nodes]*num_samples
    ret['median_num_nodes'] = int(np.median(ret['all_num_nodes']))
    ret['dataset'] = list(map(lambda i:NodeFeatData(A[i],F=None,X=X[i],Y=Y[i]),np.arange(num_samples)))

    return ret

def generate_synthdata_er_ksbm(num_samples:int, num_feats:int, num_nodes:int,
                               edge_prob, num_blocks:int, in_prob, out_prob, filt_order:int,
                               LINK_LAB_SIG:bool, LINK_LAB_STRUC:bool, LINK_SIG_STRUC:bool):
    Y = torch.tensor(np.random.binomial(1,.5,num_samples))

    if LINK_LAB_SIG:
        X = [np.random.normal((Y[i]*2-1)*.2,1,(num_nodes,num_feats)) for i in range(num_samples)]
    else:
        X = [np.random.normal(0,1,(num_nodes,num_feats)) for i in range(num_samples)]

    if LINK_LAB_STRUC:
        A = [erdos_renyi(N=num_nodes,edge_prob=edge_prob) if Y[i] else ksbm(N=num_nodes,k=num_blocks,in_prob=in_prob,out_prob=out_prob) for i in range(num_samples)]
    else:
        A = [erdos_renyi(N=num_nodes,edge_prob=edge_prob) for i in range(num_samples)]

    if LINK_SIG_STRUC:
        filt_taps = np.random.rand(filt_order)
        filt_taps /= np.sum(filt_taps)
        H = [np.sum([filt_taps[l]*np.linalg.matrix_power(A[i],l) for l in range(filt_order)],axis=0) for i in range(num_samples)]
        X = [H[i]@X[i] for i in range(num_samples)]
    
    to_tensor = lambda x:torch.tensor(x).float()
    A = list(map(to_tensor,A))
    X = list(map(to_tensor,X))

    ret = dict()
    ret['num_classes'] = 2
    ret['num_data_feats'] = num_feats
    ret['num_samples'] = num_samples
    ret['all_num_nodes'] = [num_nodes]*num_samples
    ret['median_num_nodes'] = int(np.median(ret['all_num_nodes']))
    ret['dataset'] = list(map(lambda i:NodeFeatData(A[i],F=None,X=X[i],Y=Y[i]),np.arange(num_samples)))

    return ret

# ------------------------------------------------
# LSE GAE

class NodeFeatData():
    def __init__(self,N=None,A=None,edge_index=None,F=None,X=None,Y=None):
        assert A is not None or edge_index is not None, "Must provide graph structure via adjacency matrix `A` or edge list `edge_index`."
        assert ((A is None)-.5)*((edge_index is None)-.5) < 0, "Either provide `A` or `edge_index`, but not both."
        assert X is not None, "Features `X` are missing."
        assert Y is not None, "Label `Y` is missing."

        if edge_index is not None:
            A = to_dense_adj(edge_index, max_num_nodes =N)[0]
        self.A = A
        self.num_nodes = A.shape[0]
        assert self.num_nodes == N
        self.Ah = A + torch.eye(self.num_nodes)

        if F is None:
            self.F = torch.tensor(compute_features(A.double().numpy())).float()

        self.X = X
        self.num_feats = X.shape[1]

        self.Y = Y

    def plot_F(self, fig_title:str=''):
        fig = plt.figure()
        ax = fig.subplots()
        ax.imshow(self.F,mycmap)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Node')
        if len(fig_title)>0:
            ax.set_title(fig_title)
        fig.tight_layout()

def nodefeat_to_pyg(sample:NodeFeatData):
    pyg_graph = Data()
    pyg_graph.edge_index = dense_to_sparse(sample.A)[0]
    pyg_graph.x = sample.X.clone()
    pyg_graph.y = sample.Y.clone()
    pyg_graph.num_nodes = sample.num_nodes
    return pyg_graph

def gae_loss(F,Fh):
    return torch.linalg.norm(F-Fh,"fro")**2/(F.shape[0]*F.shape[1])

class MLPLayer(torch.nn.Module):
    def __init__(self, in_dim:int, out_dim:int):
        super().__init__()
        assert in_dim>0 and out_dim>0, 'Invalid dimension.'

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = torch.nn.Parameter(torch.empty(self.in_dim, self.out_dim))
        self.b = torch.nn.Parameter(torch.empty(self.out_dim))
        torch.nn.init.kaiming_uniform_(self.W.data)
        torch.nn.init.uniform_(self.b.data)
    
    def forward(self,H):
        return H@self.W + self.b
        
class GCNLayer(torch.nn.Module):
    def __init__(self, in_dim:int, out_dim:int):
        super().__init__()
        assert in_dim>0 and out_dim>0, 'Invalid dimension.'

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = torch.nn.Parameter(torch.empty(self.in_dim, self.out_dim))
        torch.nn.init.kaiming_uniform_(self.W.data)

    def forward(self, H, A):
        assert (A.shape[0], self.in_dim)==H.shape, 'Invalid dimension.'
        return A@H@self.W

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_dim:int, hid_dim:int, out_dim:int, n_layers:int, nonlin=torch.nn.ReLU):
        super().__init__()
        assert in_dim>0 and out_dim>0 and hid_dim>0, 'Invalid dimension.'
        assert n_layers>0, 'Invalid number of layers.'

        self.n_layers = n_layers
        self.convs = torch.nn.ModuleList()
        self.nonlin = nonlin()

        if self.n_layers<=1:
            self.convs.append(GCNLayer(in_dim,out_dim))
        else:
            self.convs.append(GCNLayer(in_dim,hid_dim))
            for _ in range(n_layers-2):
                self.convs.append(GCNLayer(hid_dim,hid_dim))
            self.convs.append(GCNLayer(hid_dim,out_dim))

    def forward(self,H,A):
        for i_layer in range(self.n_layers-1):
            H = self.nonlin(self.convs[i_layer](H,A))
        H = self.convs[-1](H,A)
        return H
    
class GCNDecoder(torch.nn.Module):
    def __init__(self, in_dim:int, hid_dim:int, out_dim:int, n_layers:int, nonlin=torch.nn.ReLU):
        super().__init__()
        assert in_dim>0 and out_dim>0 and hid_dim>0, 'Invalid dimension.'
        assert n_layers>0, 'Invalid number of layers.'

        self.n_layers = n_layers
        self.convs = torch.nn.ModuleList()
        self.nonlin = nonlin()

        if self.n_layers<=1:
            self.convs.append(MLPLayer(in_dim,out_dim))
        else:
            self.convs.append(MLPLayer(in_dim,hid_dim))
            for _ in range(n_layers-2):
                self.convs.append(MLPLayer(hid_dim,hid_dim))
            self.convs.append(MLPLayer(hid_dim,out_dim))

    def forward(self,H):
        for i_layer in range(self.n_layers-1):
            H = self.nonlin(self.convs[i_layer](H))
        H = self.convs[-1](H)
        return H

class GCNAutoencoder(torch.nn.Module):
    def __init__(self, in_dim:int, hid_dim:int, out_dim:int, lat_dim:int,
                 n_layers_enc:int, n_layers_dec:int, nonlin=torch.nn.ReLU):
        super().__init__()

        self.n_layers_enc = n_layers_enc
        self.n_layers_dec = n_layers_dec
        self.encoder = GCNEncoder(in_dim,hid_dim,lat_dim,n_layers_enc,nonlin=nonlin)
        self.decoder = GCNDecoder(lat_dim,hid_dim,out_dim,n_layers_dec,nonlin=nonlin)
    
    def encode(self,F,A):
        return self.encoder(F,A)
    def decode(self,Z):
        return self.decoder(Z)
    def forward(self,F,A):
        return self.decoder(self.encoder(F,A))

def train_gae_model(train_data,val_data,
                    in_dim:int,out_dim:int,hid_dim:int,lat_dim:int,n_layers_enc:int,n_layers_dec:int,
                    nonlin,batch_train_size:int,batch_val_size:int,
                    lr,lmbda,gamma,patience:int,epochs:int,verbose:bool):
    autoenc = GCNAutoencoder(in_dim,hid_dim,out_dim,lat_dim,n_layers_enc,n_layers_dec,nonlin)
    optimizer = torch.optim.Adam(autoenc.parameters(), lr=lr, weight_decay=lmbda)
    sched = ReduceLROnPlateau(optimizer, factor=gamma, patience=patience)
    autoenc.train()

    train_loss_iters = []
    val_loss_iters = []
    min_val_loss = np.inf
    min_val_ind = -1
    min_val_model = GCNAutoencoder(in_dim,hid_dim,out_dim,lat_dim,n_layers_enc,n_layers_dec,nonlin)

    for epoch in range(epochs):
        gae_train_batch = batch_inds(train_data,batch_train_size)
        gae_val_batch = batch_inds(val_data,batch_val_size)

        optimizer.zero_grad()
        train_loss = 0
        for i in range(len(gae_train_batch)):
            train_loss = train_loss + gae_loss(gae_train_batch[i].F,autoenc(gae_train_batch[i].F,gae_train_batch[i].Ah))
        train_loss = train_loss/batch_train_size
        train_loss.backward()
        optimizer.step()
        train_loss_iters.append(train_loss.item())

        val_loss = 0
        for i in range(len(gae_val_batch)):
            val_loss = val_loss + gae_loss(gae_val_batch[i].F,autoenc(gae_val_batch[i].F,gae_val_batch[i].Ah))
        val_loss = val_loss/batch_val_size
        val_loss_iters.append(val_loss.item())
        sched.step(val_loss_iters[-1])

        if val_loss.item()<=min_val_loss:
            min_val_loss = val_loss.item()
            min_val_ind = epoch
            min_val_model.load_state_dict(autoenc.state_dict())

        if verbose and epoch%(epochs//5)==0:
            print(f"Epoch: {epoch:03d} | " + 
                  f"Train loss: {train_loss_iters[-1]:.3f}, " + 
                  f"Val loss: {val_loss_iters[-1]:.3f}")
    
    train_pred_loss = train_loss_iters[min_val_ind]
    val_pred_loss = val_loss_iters[min_val_ind]

    if verbose:
        print(f"Final: {min_val_ind:03d} | " + 
                f"Train loss: {train_pred_loss:.3f}, " + 
                f"Val loss: {val_pred_loss:.3f}")
        print('')
    
    autoenc.load_state_dict(min_val_model.state_dict())
    return autoenc, train_loss_iters, val_loss_iters

# ------------------------------------------------
# LSE learning

def pred_node_feats_simple(sample:NodeFeatData,PRED_FEAT_TYPE:str='true',verbose=True):
    if PRED_FEAT_TYPE=='true':
        X_pred = sample.X
    elif PRED_FEAT_TYPE=='zeros':
        X_pred = torch.zeros((sample.num_nodes,sample.num_feats))
    elif PRED_FEAT_TYPE=='ones':
        X_pred = torch.ones((sample.num_nodes,sample.num_feats))
    elif PRED_FEAT_TYPE=='random':
        X_pred = torch.rand((sample.num_nodes,sample.num_feats))
    elif PRED_FEAT_TYPE=='degree':
        X_pred = torch.cat([(sample.A.sum(dim=0)).view(-1,1)/sample.num_nodes]*sample.num_feats,dim=1)
    else:
        if verbose:
            print('Invalid node feature prediction type.')
        return
    sample.X = X_pred
    return sample

def pred_node_feats_lse(sample:NodeFeatData,train_data,autoenc:GCNAutoencoder,
                        k_graph=2,k_node=2):
    Z_samp = autoenc.encode(sample.F,sample.Ah).detach().double().numpy()
    Y_train_inds = [i for i in range(len(train_data)) if train_data[i].Y==sample.Y]

    Z_train_samps = list(map(lambda i:autoenc.encode(train_data[i].F,train_data[i].Ah).detach().double().numpy(),Y_train_inds))
    X_train_samps = list(map(lambda i:train_data[i].X.double().numpy(),Y_train_inds))

    graph_samp_dists = np.linalg.norm(list(map(lambda Z_train_samp:np.mean(Z_samp,axis=0)-np.mean(Z_train_samp,axis=0),Z_train_samps)),axis=1)
    graph_sim_inds = np.argsort(graph_samp_dists)[:k_graph]

    X_similar = []
    for i_graph in range(k_graph):
        Z_train_samp = Z_train_samps[graph_sim_inds[i_graph]]
        node_samp_dists = np.array([[np.linalg.norm(Z_train_samp[i1]-Z_samp[i2]) for i2 in range(len(Z_samp))] for i1 in range(len(Z_train_samp))])
        node_sim_inds = np.argsort(node_samp_dists,axis=0)[:k_node]
        X_similar.append( np.mean(X_train_samps[graph_sim_inds[i_graph]][node_sim_inds],axis=0) )
    X_samp_pred = np.mean(X_similar,axis=0)

    sample.X = torch.tensor(X_samp_pred).float()

    return sample 

# ------------------------------------------------
# GNN for graph classification

class GIN(torch.nn.Module):
    def __init__(self, num_feats:int, num_classes:int, num_hid:int=32):
        super().__init__()
        assert num_feats>0, 'Invalid number of features.'
        assert num_classes>0, 'Invalid number of classes.'
        assert num_hid>0, 'Invalid dimension.'

        nn1 = Sequential(Linear(num_feats, num_hid), ReLU(), Linear(num_hid, num_hid))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(num_hid)

        nn2 = Sequential(Linear(num_hid, num_hid), ReLU(), Linear(num_hid, num_hid))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(num_hid)

        nn3 = Sequential(Linear(num_hid, num_hid), ReLU(), Linear(num_hid, num_hid))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(num_hid)

        nn4 = Sequential(Linear(num_hid, num_hid), ReLU(), Linear(num_hid, num_hid))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(num_hid)

        nn5 = Sequential(Linear(num_hid, num_hid), ReLU(), Linear(num_hid, num_hid))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(num_hid)

        self.fc1 = Linear(num_hid, num_hid)
        self.fc2 = Linear(num_hid, num_classes)

    def forward(self, x, edge_index, batch):
        x = torch.nn.functional.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = torch.nn.functional.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = torch.nn.functional.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = torch.nn.functional.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = torch.nn.functional.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        # x = global_add_pool(x, batch)
        x = global_mean_pool(x, batch)
        x = torch.nn.functional.relu(self.fc1(x))
        # x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=-1)


class MLPGClas(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=32):
        super(MLPGClas, self).__init__()
        dim = num_hidden

        self.conv1 = Sequential(Linear(num_features, dim), ReLU())
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.conv2 = Sequential(Linear(dim, dim), ReLU())
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.conv3 = Sequential(Linear(dim, dim), ReLU())
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.conv4 = Sequential(Linear(dim, dim), ReLU())
        self.bn4 = torch.nn.BatchNorm1d(dim)

        self.conv5 = Sequential(Linear(dim, dim), ReLU())
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, x, _, batch):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.bn1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.bn2(x)
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.bn3(x)
        x = torch.nn.functional.relu(self.conv4(x))
        x = self.bn4(x)
        x = torch.nn.functional.relu(self.conv5(x))
        x = self.bn5(x)
        # x = global_add_pool(x, batch)
        x = global_mean_pool(x, batch)
        x = torch.nn.functional.relu(self.fc1(x))
        # x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=-1)


def train_gin(model,loader,optimizer,loss_fn):
    model.train()
    correct_all = 0
    loss_all = 0
    graph_all = 0
    model = model.to(DEVICE)
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        output = model(data.x,data.edge_index,data.batch)
        pred = output.max(dim=1)[1]
        # y = data.y.view(-1,num_classes)
        y = data.y.view(-1)
        loss = loss_fn(output,y)
        # y = y.max(dim=1)[1]
        correct_all += pred.eq(y).sum().item()
        loss.backward()
        loss_all += loss.item()*data.num_graphs
        graph_all += data.num_graphs
        optimizer.step()
    train_acc = correct_all/graph_all
    train_loss = loss_all/graph_all
    return model, train_acc, train_loss

def test_gin(model,loader,loss_fn):
    model.eval()
    correct_all = 0
    graph_all = 0
    loss_all = 0
    for data in loader:
        data = data.to(DEVICE)
        output = model(data.x,data.edge_index,data.batch)
        pred = output.max(dim=1)[1]
        # y = data.y.view(-1,num_classes)
        y = data.y.view(-1)
        loss_all += loss_fn(output,y).item()*data.num_graphs
        # y = y.max(dim=1)[1]
        correct_all += pred.eq(y).sum().item()
        graph_all += data.num_graphs
    test_acc = correct_all/graph_all
    test_loss = loss_all/graph_all
    return test_acc, test_loss

def train_gin_model(train_data,val_data,test_data,
                    model_name: str,
                    batch_train_size:int, batch_val_size:int, batch_test_size:int,
                    num_feats:int, num_classes:int, num_hid:int,
                    lr, lmbda, gamma, patience:int, epochs:int, verbose:bool):
    train_loader = DataLoader(train_data, batch_size=batch_train_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_val_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_test_size, shuffle=False)

    if model_name == "GIN":
        model = GIN(num_feats=num_feats, num_classes=num_classes, num_hid=num_hid)
        min_val_model = GIN(num_feats=num_feats, num_classes=num_classes, num_hid=num_hid)
    elif model_name == "MLP":
        model = MLPGClas(num_features=num_feats, num_classes=num_classes, num_hidden=num_hid)
        min_val_model = MLPGClas(num_features=num_feats, num_classes=num_classes, num_hidden=num_hid)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lmbda)
    sched = ReduceLROnPlateau(optimizer, factor=gamma, patience=patience)
    gin_loss = torch.nn.CrossEntropyLoss()

    train_loss_iters = []
    val_loss_iters = []
    test_loss_iters = []
    train_acc_iters = []
    val_acc_iters = []
    test_acc_iters = []

    min_val_loss = np.inf
    min_val_ind = -1

    for epoch in range(epochs):
        model, train_acc, train_loss = train_gin(model,train_loader,optimizer,gin_loss)
        val_acc, val_loss = test_gin(model,val_loader,gin_loss)
        test_acc, test_loss = test_gin(model,test_loader,gin_loss)
        sched.step(val_loss)
        if verbose and epoch%(epochs//5)==0:
            print(f"Epoch: {epoch:03d} | "+
                f"Train loss: {train_loss:.3f}, "+
                f"Val loss: {val_loss:.3f}, "+
                f"Test loss: {test_loss:.3f} | "+
                f"Train acc: {train_acc:.3f}, "+
                f"Val acc: {val_acc:.3f}, "+
                f"Test acc: {test_acc:.3f}")
        train_loss_iters.append(train_loss)
        val_loss_iters.append(val_loss)
        test_loss_iters.append(test_loss)
        train_acc_iters.append(train_acc)
        val_acc_iters.append(val_acc)
        test_acc_iters.append(test_acc)

        if val_loss<=min_val_loss:
            min_val_loss = val_loss
            min_val_ind = epoch
            min_val_model.load_state_dict(model.state_dict())

    train_pred_loss = train_loss_iters[min_val_ind]
    val_pred_loss = val_loss_iters[min_val_ind]
    test_pred_loss = test_loss_iters[min_val_ind]
    train_pred_acc = train_acc_iters[min_val_ind]
    val_pred_acc = val_acc_iters[min_val_ind]
    test_pred_acc = test_acc_iters[min_val_ind]

    if verbose:
        print(f"Final: {min_val_ind:03d} | " + 
            f"Train loss: {train_pred_loss:.3f}, "+
            f"Val loss: {val_pred_loss:.3f}, "+
            f"Test loss: {test_pred_loss:.3f} | "+
            f"Train acc: {train_pred_acc:.3f}, "+
            f"Val acc: {val_pred_acc:.3f}, "+
            f"Test acc: {test_pred_acc:.3f}")
        print('')

    ret = {
        'model':model,
        'train_loss_iters':train_loss_iters,
        'val_loss_iters':val_loss_iters,
        'test_loss_iters':test_loss_iters,
        'train_acc_iters':train_acc_iters,
        'val_acc_iters':val_acc_iters,
        'test_acc_iters':test_acc_iters,
        'train_pred_loss':train_pred_loss,
        'val_pred_loss':val_pred_loss,
        'test_pred_loss':test_pred_loss,
        'train_pred_acc':train_pred_acc,
        'val_pred_acc':val_pred_acc,
        'test_pred_acc':test_pred_acc
    }
    return ret

# ------------------------------------------------
# Experiments for node feature prediction comparison

class GAE_Experiment:
    def __init__(self, train_data, val_data, verbose:bool=False):
        self.train_data = train_data
        self.val_data = val_data
        self.num_train_samples = len(train_data)
        self.num_val_samples = len(val_data)
        self.verbose = verbose

        self.num_feats = train_data[0].num_feats

    def train_model(self,
                    in_dim:int, out_dim:int, hid_dim:int, lat_dim:int,
                    n_layers_enc:int, n_layers_dec:int, nonlin,
                    batch_train_size:int, batch_val_size:int,
                    lr, lmbda, gamma, patience:int, epochs:int):
        args = {
            'train_data':self.train_data,
            'val_data':self.val_data,
            'in_dim':in_dim,
            'out_dim':out_dim,
            'hid_dim':hid_dim,
            'lat_dim':lat_dim,
            'n_layers_enc':n_layers_enc,
            'n_layers_dec':n_layers_dec,
            'nonlin':nonlin,
            'batch_train_size':batch_train_size,
            'batch_val_size':batch_val_size,
            'lr':lr,
            'lmbda':lmbda,
            'gamma':gamma,
            'patience':patience,
            'epochs':epochs,
            'verbose':self.verbose
        }
        autoenc, train_loss_iters, val_loss_iters = train_gae_model(**args)
        self.autoenc = autoenc
        self.train_loss_iters = train_loss_iters
        self.val_loss_iters = val_loss_iters
        self.F_train_pred = list(map(lambda i:autoenc(self.train_data[i].F,self.train_data[i].Ah).detach().double().numpy(),np.arange(self.num_train_samples)))
        self.F_val_pred = list(map(lambda i:autoenc(self.val_data[i].F,self.val_data[i].Ah).detach().double().numpy(),np.arange(self.num_val_samples)))
    
    def plot_loss_iters(self,fig_title:str=''):
        assert hasattr(self,'train_loss_iters'), 'Model not yet trained.'
        fig = plt.figure(figsize=(2*6,4))
        ax = fig.subplots(1,2)
        ax[0].plot(self.train_loss_iters, c=reds[10])
        ax[0].grid(True)
        ax[0].set_xlabel('Training iteration')
        ax[0].set_ylabel('Training loss')
        ax[1].plot(self.val_loss_iters, c=blues[10])
        ax[1].grid(True)
        ax[1].set_xlabel('Training iteration')
        ax[1].set_ylabel('Validation loss')
        if len(fig_title)>0:
            fig.suptitle(fig_title)
        fig.tight_layout()

    def plot_F_train_pred(self, ind:int=0, fig_title:str=''):
        assert ind in range(self.num_train_samples)
        fig = plt.figure()
        ax = fig.subplots()
        ax.imshow(self.F_train_pred[ind],mycmap)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Node')
        if len(fig_title)>0:
            ax.set_title(fig_title)
        fig.tight_layout()

    def plot_F_val_pred(self, ind:int=0, fig_title:str=''):
        assert ind in range(self.num_val_samples)
        fig = plt.figure()
        ax = fig.subplots()
        ax.imshow(self.F_val_pred[ind],mycmap)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Node')
        if len(fig_title)>0:
            ax.set_title(fig_title)
        fig.tight_layout()

class GIN_Experiment:
    def __init__(self, train_data, val_data, test_data, model_name: str, verbose:bool=False):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.model_name = model_name
        self.verbose = verbose

        self.num_train_samples = len(train_data)
        self.num_val_samples = len(val_data)
        self.num_test_samples = len(test_data)

    def train_model(self,
                    batch_train_size:int, batch_val_size:int, batch_test_size:int,
                    num_feats:int, num_classes:int, num_hid:int,
                    lr, lmbda, gamma, patience:int, epochs:int
                    ):
        args = {
            'train_data':self.train_data,
            'val_data':self.val_data,
            'test_data':self.test_data,
            'model_name':self.model_name,
            'batch_train_size':batch_train_size,
            'batch_val_size':batch_val_size,
            'batch_test_size':batch_test_size,
            'num_feats':num_feats,
            'num_classes':num_classes,
            'num_hid':num_hid,
            'lr':lr,
            'lmbda':lmbda,
            'gamma':gamma,
            'patience':patience,
            'epochs':epochs,
            'verbose':self.verbose
        }
        results = train_gin_model(**args)
        for key,value in results.items():
            setattr(self, key, value)

    def plot_loss_iters(self, fig_title:str=''):
        assert hasattr(self,'train_loss_iters') and hasattr(self,'val_loss_iters') and hasattr(self,'test_loss_iters') , 'Model not yet trained.'

        fig = plt.figure(figsize=(3*6,4))
        ax = fig.subplots(1,3)
        ax[0].plot(self.train_loss_iters, c=reds[10])
        ax[0].grid(True)
        ax[0].set_xlabel('Training iteration')
        ax[0].set_ylabel('Training loss')
        ax[1].plot(self.val_loss_iters, c=blues[10])
        ax[1].grid(True)
        ax[1].set_xlabel('Training iteration')
        ax[1].set_ylabel('Validation loss')
        ax[2].plot(self.test_loss_iters, c=greens[10])
        ax[2].grid(True)
        ax[2].set_xlabel('Training iteration')
        ax[2].set_ylabel('Testing loss')
        if len(fig_title)>0:
            fig.suptitle(fig_title)
        fig.tight_layout()

    def plot_acc_iters(self, fig_title:str=''):
        assert hasattr(self,'train_acc_iters') and hasattr(self,'val_acc_iters') and hasattr(self,'test_acc_iters') , 'Model not yet trained.'

        fig = plt.figure(figsize=(3*6,4))
        ax = fig.subplots(1,3)
        ax[0].plot(self.train_acc_iters, c=reds[10])
        ax[0].grid(True)
        ax[0].set_xlabel('Training iteration')
        ax[0].set_ylabel('Training accuracy')
        ax[1].plot(self.val_acc_iters, c=blues[10])
        ax[1].grid(True)
        ax[1].set_xlabel('Training iteration')
        ax[1].set_ylabel('Validation accuracy')
        ax[2].plot(self.test_acc_iters, c=greens[10])
        ax[2].grid(True)
        ax[2].set_xlabel('Training iteration')
        ax[2].set_ylabel('Testing accuracy')
        if len(fig_title)>0:
            fig.suptitle(fig_title)
        fig.tight_layout()
