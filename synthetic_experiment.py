# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from model import *
from time import perf_counter

verbose = True

# ---------------------------
# Load parameters
data_params = load_synthdata_params('/config/synthdataconfig.jsonc')
model_params = load_model_params('/config/modelconfig.jsonc')
num_classes = 2
num_samples = data_params['num_samples']
num_ls_feats = 21
# ---------------------------


# ---------------------------
data_args = {
    "num_samples":data_params['num_samples'],
    "num_feats":data_params['num_data_feats'],
    "num_nodes":data_params['num_nodes'],
    "filt_order":data_params['filt_order'],
    "LINK_LAB_SIG":data_params['LINK_LAB_SIG'],
    "LINK_LAB_STRUC":data_params['LINK_LAB_STRUC'],
    "LINK_SIG_STRUC":data_params['LINK_SIG_STRUC']
}

gae_args = {
    'in_dim':num_ls_feats,
    'out_dim':num_ls_feats,
    'hid_dim':model_params['gae_hid_dim'],
    'lat_dim':model_params['gae_lat_dim'],
    'n_layers_enc':model_params['gae_n_layers'],
    'n_layers_dec':model_params['gae_n_layers'],
    'nonlin':torch.nn.Sigmoid,
    'batch_train_size':model_params['batch_train_size'],
    'batch_val_size':model_params['batch_val_size'],
    'lr':model_params['gae_lr'],
    'lmbda':model_params['gae_lmbda'],
    'gamma':model_params['gae_gamma'],
    'patience':model_params['gae_patience'],
    'epochs':model_params['gae_epochs'],
}

gnn_args = {
    "batch_train_size":model_params['batch_train_size'],
    "batch_val_size":model_params['batch_val_size'],
    "batch_test_size":model_params['batch_test_size'],
    "num_feats":data_params['num_data_feats'],
    "num_classes":num_classes,
    "num_hid":model_params['gnn_hid_dim'],
    "lr":model_params['gnn_lr'],
    "lmbda":model_params['gnn_lmbda'],
    "gamma":model_params['gnn_gamma'],
    "patience":model_params['gnn_patience'],
    "epochs":model_params['gnn_epochs'],
}

num_train_samples = int(num_samples*model_params['train_ratio'])
num_val_samples = int(num_samples*model_params['val_ratio'])
num_miss_samples = int(num_samples*model_params['miss_ratio'])
num_test_samples = num_samples - num_train_samples - num_val_samples - num_miss_samples
# ---------------------------


# ---------------------------
# Train models for graph classification
gae_list = [[[] for _ in range(model_params['num_trials'])] for _ in range(data_params['num_trials'])]
gin_list = [[[] for _ in range(model_params['num_trials'])] for _ in range(data_params['num_trials'])]
exp_list = [[[] for _ in range(model_params['num_trials'])] for _ in range(data_params['num_trials'])]

tic = perf_counter()
for dt in range(data_params['num_trials']):
    if verbose:
        print(f"Data trial {dt+1} of {data_params['num_trials']}")

    # ---------------------------
    # Synthetic data
    ret = generate_synthdata(**data_args)
    gae_data = ret['dataset']
    # ---------------------------


    # ---------------------------
    for mt in range(model_params['num_trials']):
        if verbose:
            print(f"Model trial {mt+1} of {model_params['num_trials']}")
        # ---------------------------
        # Set up GAE and GNN data splits
        shuff_inds = np.random.permutation(num_samples)
        gae_train_data = list(map(lambda i:gae_data[i],shuff_inds[:num_train_samples]))
        gae_miss_data = list(map(lambda i:gae_data[i],shuff_inds[num_train_samples:num_train_samples+num_miss_samples]))
        gae_val_data = list(map(lambda i:gae_data[i],shuff_inds[num_train_samples+num_miss_samples:num_train_samples+num_miss_samples+num_val_samples]))
        gae_test_data = list(map(lambda i:gae_data[i],shuff_inds[num_train_samples+num_miss_samples+num_val_samples:]))
        # ---------------------------


        # ---------------------------
        # Train GAE for structure-based node embeddings
        gae_exp = GAE_Experiment(gae_train_data,gae_val_data,verbose)
        gae_exp.train_model(**gae_args)
        gae_list[dt][mt] = gae_exp
        # ---------------------------


        # ---------------------------
        # Graph classification for different node feature prediction methods
        exp_featpreds = dict(zip(model_params['PRED_FEAT_TYPES'],[None]*len(model_params['PRED_FEAT_TYPES'])))
        for PRED_FEAT_TYPE in model_params['PRED_FEAT_TYPES']:
            if verbose:
                print(f'Prediction type: {PRED_FEAT_TYPE}')

            # ---------------------------
            # Predict missing features
            if PRED_FEAT_TYPE!='lse':
                gae_miss_data_pred = list(map(lambda sample:pred_node_feats_simple(sample,PRED_FEAT_TYPE,verbose),gae_miss_data))
            else:
                gae_miss_data_pred = list(map(lambda sample:pred_node_feats_lse(sample,gae_train_data,gae_exp.autoenc,model_params['k_graph'],model_params['k_node']),gae_miss_data))
            # ---------------------------


            # ---------------------------
            # Set up GNN data splits
            gnn_train_data = list(map(lambda sample:nodefeat_to_pyg(sample), gae_train_data + gae_miss_data_pred))
            gnn_val_data = list(map(lambda sample:nodefeat_to_pyg(sample), gae_val_data))
            gnn_test_data = list(map(lambda sample:nodefeat_to_pyg(sample), gae_test_data))
            # ---------------------------


            # ---------------------------
            # Train GIN for graph classification
            gin_exp = GIN_Experiment(gnn_train_data,gnn_val_data,gnn_test_data,verbose)
            gin_exp.train_model(**gnn_args)
            exp_featpreds[PRED_FEAT_TYPE] = gin_exp
            # ---------------------------
        gin_list[dt][mt] = exp_featpreds
        
        exp_list[dt][mt] = (gae_list[dt][mt], gin_list[dt][mt])

        print("")
        # ---------------------------
    # ---------------------------
# ---------------------------

toc = perf_counter()
if verbose:
    if (toc-tic)/60 < 1:
        print(f'{toc-tic:.2f}s')
    elif (toc-tic)/3600<1:
        print(f'{(toc-tic)//60:.0f}m {(toc-tic)%60:.2f}s')
    else:
        print(f'{(toc-tic)//3600:.0f}h {((toc-tic)%3600)//60:.0f}m {((toc-tic)%3600)%60:.2f}s')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot loss during training for GAE

gae_train_loss_avg = np.mean([gae_list[dt][mt].train_loss_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)
gae_train_loss_min = np.min([gae_list[dt][mt].train_loss_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)
gae_train_loss_max = np.max([gae_list[dt][mt].train_loss_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)

gae_val_loss_avg = np.mean([gae_list[dt][mt].val_loss_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)
gae_val_loss_min = np.min([gae_list[dt][mt].val_loss_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)
gae_val_loss_max = np.max([gae_list[dt][mt].val_loss_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)

fig = plt.figure(figsize=(2*6,4))
ax = fig.subplots(1,2)
ax[0].plot(gae_train_loss_avg,'-',c=reds[13])
ax[0].fill_between(np.arange(model_params['gae_epochs']),gae_train_loss_min,gae_train_loss_max,color=reds[4])
ax[0].grid(True)
ax[0].set_xlabel(f'Training iteration')
ax[0].set_ylabel(f'Training loss')
ax[1].plot(gae_val_loss_avg,'-',c=blues[12])
ax[1].fill_between(np.arange(model_params['gae_epochs']),gae_val_loss_min,gae_val_loss_max,color=blues[3])
ax[1].grid(True)
ax[1].set_xlabel(f'Training iteration')
ax[1].set_ylabel(f'Validation loss')

# Plot loss during training for GIN
for PRED_FEAT_TYPE in model_params['PRED_FEAT_TYPES']:
    gin_train_loss_avg = np.mean([gin_list[dt][mt][PRED_FEAT_TYPE].train_loss_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)
    gin_train_loss_min = np.min([gin_list[dt][mt][PRED_FEAT_TYPE].train_loss_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)
    gin_train_loss_max = np.max([gin_list[dt][mt][PRED_FEAT_TYPE].train_loss_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)

    gin_val_loss_avg = np.mean([gin_list[dt][mt][PRED_FEAT_TYPE].val_loss_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)
    gin_val_loss_min = np.min([gin_list[dt][mt][PRED_FEAT_TYPE].val_loss_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)
    gin_val_loss_max = np.max([gin_list[dt][mt][PRED_FEAT_TYPE].val_loss_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)

    gin_test_loss_avg = np.mean([gin_list[dt][mt][PRED_FEAT_TYPE].test_loss_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)
    gin_test_loss_min = np.min([gin_list[dt][mt][PRED_FEAT_TYPE].test_loss_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)
    gin_test_loss_max = np.max([gin_list[dt][mt][PRED_FEAT_TYPE].test_loss_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)

    fig = plt.figure(figsize=(3*6,4))
    ax = fig.subplots(1,3)
    ax[0].plot(gin_train_loss_avg,'-',c=reds[13])
    ax[0].fill_between(np.arange(model_params['gnn_epochs']),gin_train_loss_min,gin_train_loss_max,color=reds[4])
    ax[0].grid(True)
    ax[0].set_xlabel(f'Training iteration')
    ax[0].set_ylabel(f'Training loss')
    ax[1].plot(gin_val_loss_avg,'-',c=blues[12])
    ax[1].fill_between(np.arange(model_params['gnn_epochs']),gin_val_loss_min,gin_val_loss_max,color=blues[3])
    ax[1].grid(True)
    ax[1].set_xlabel(f'Training iteration')
    ax[1].set_ylabel(f'Validation loss')
    ax[2].plot(gin_test_loss_avg,'-',c=greens[12])
    ax[2].fill_between(np.arange(model_params['gnn_epochs']),gin_test_loss_min,gin_test_loss_max,color=greens[3])
    ax[2].grid(True)
    ax[2].set_xlabel(f'Training iteration')
    ax[2].set_ylabel(f'Testing loss')
    fig.suptitle(f'Prediction type: {PRED_FEAT_TYPE}')
    fig.tight_layout()


# Plot accuracy during training for GIN
for PRED_FEAT_TYPE in model_params['PRED_FEAT_TYPES']:
    gin_train_acc_avg = np.mean([gin_list[dt][mt][PRED_FEAT_TYPE].train_acc_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)
    gin_train_acc_min = np.min([gin_list[dt][mt][PRED_FEAT_TYPE].train_acc_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)
    gin_train_acc_max = np.max([gin_list[dt][mt][PRED_FEAT_TYPE].train_acc_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)

    gin_val_acc_avg = np.mean([gin_list[dt][mt][PRED_FEAT_TYPE].val_acc_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)
    gin_val_acc_min = np.min([gin_list[dt][mt][PRED_FEAT_TYPE].val_acc_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)
    gin_val_acc_max = np.max([gin_list[dt][mt][PRED_FEAT_TYPE].val_acc_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)

    gin_test_acc_avg = np.mean([gin_list[dt][mt][PRED_FEAT_TYPE].test_acc_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)
    gin_test_acc_min = np.min([gin_list[dt][mt][PRED_FEAT_TYPE].test_acc_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)
    gin_test_acc_max = np.max([gin_list[dt][mt][PRED_FEAT_TYPE].test_acc_iters for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])],axis=0)

    fig = plt.figure(figsize=(3*6,4))
    ax = fig.subplots(1,3)
    ax[0].plot(gin_train_acc_avg,'-',c=reds[13])
    ax[0].fill_between(np.arange(model_params['gnn_epochs']),gin_train_acc_min,gin_train_acc_max,color=reds[4])
    ax[0].grid(True)
    ax[0].set_xlabel(f'Training iteration')
    ax[0].set_ylabel(f'Training accuracy')
    ax[1].plot(gin_val_acc_avg,'-',c=blues[12])
    ax[1].fill_between(np.arange(model_params['gnn_epochs']),gin_val_acc_min,gin_val_acc_max,color=blues[3])
    ax[1].grid(True)
    ax[1].set_xlabel(f'Training iteration')
    ax[1].set_ylabel(f'Validation accuracy')
    ax[2].plot(gin_test_acc_avg,'-',c=greens[12])
    ax[2].fill_between(np.arange(model_params['gnn_epochs']),gin_test_acc_min,gin_test_acc_max,color=greens[3])
    ax[2].grid(True)
    ax[2].set_xlabel(f'Training iteration')
    ax[2].set_ylabel(f'Testing accuracy')
    fig.suptitle(f'Prediction type: {PRED_FEAT_TYPE}')
    fig.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot graphs in different classes
Y_inds = [torch.where(torch.tensor(list(map(lambda i_samp:gae_data[i_samp].Y,np.arange(num_samples))))==iy)[0] for iy in [0,1]]
i_samp = 0
cmap_per_class = ['Reds','Blues']
fig = plt.figure(figsize=(2*5,5))
ax = fig.subplots(1,2)
iy = 0
A_curr = gae_data[Y_inds[iy][i_samp]].A
sort_inds = torch.argsort(torch.sum(A_curr,axis=0))
A_curr = A_curr[sort_inds][:,sort_inds]
ax[iy].imshow(A_curr,cmap_per_class[iy],origin='lower')
ax[iy].axis('off')
iy = 1
A_curr = gae_data[Y_inds[iy][i_samp]].A
sort_inds = torch.argsort(torch.sum(A_curr,axis=0))
A_curr = A_curr[sort_inds][:,sort_inds]
ax[iy].imshow(A_curr,cmap_per_class[iy],origin='lower')
ax[iy].axis('off')
fig.tight_layout()

# Plot features versus predicted features for GAE
dt = 0
mt = 0
gae_curr = gae_list[dt][mt]
F_pred = list(map(lambda i_samp:gae_curr.autoenc(gae_data[i_samp].F,gae_data[i_samp].Ah).detach().double().numpy(),np.arange(num_samples)))
F_true = list(map(lambda i_samp:gae_data[i_samp].F.double().numpy(),np.arange(num_samples)))
vmax = np.maximum(np.max(F_pred),np.max(F_true))
vmin = np.minimum(np.min(F_pred),np.min(F_true))
for dt in [0]:
    for mt in [0]:
        gae_curr = gae_list[dt][mt]
        for i_samp in [1]:
            vmax = np.maximum(np.max(F_pred[i_samp]),np.max(F_true[i_samp]))
            vmin = np.minimum(np.min(F_pred[i_samp]),np.min(F_true[i_samp]))
            fig = plt.figure(figsize=(2*4,5))
            ax = fig.subplots(1,2)
            ax[0].imshow(gae_data[i_samp].F,madcmap)
            # ax[0].imshow(gae_curr.train_data[i_samp].F,mycmap,vmin=vmin,vmax=vmax)
            ax[0].set_xlabel('Feature')
            ax[0].set_ylabel('Node')
            ax[0].set_title('True features')
            ax[1].imshow(F_pred[i_samp],madcmap,vmin=vmin,vmax=vmax)
            ax[1].set_xlabel('Feature')
            ax[1].set_ylabel('Node')
            ax[1].set_title('Pred. features')
fig.tight_layout()

# Plot latent node and graph embeddings in different classes for GAE
dt = 0
mt = 0
gae_curr = gae_list[dt][mt]

Z = list(map(lambda i_samp:gae_curr.autoenc.encode(gae_data[i_samp].F,gae_data[i_samp].Ah).detach().double().numpy(),np.arange(num_samples)))
if Z[0].shape[1]>2:
    pca_basis = np.concatenate(Z,axis=0)
    pca_proj = np.linalg.eigh(pca_basis.T@pca_basis)[1][:,-2:]
    Z = list(map(lambda X:X@pca_proj,Z))
Y_inds = [torch.where(torch.tensor(list(map(lambda i_samp:gae_data[i_samp].Y,np.arange(num_samples))))==iy)[0] for iy in [0,1]]
Z_per_class = [list(map(lambda i_samp:Z[i_samp],Y_inds[iy])) for iy in [0,1]]

mkrs_per_class = ['o','x']
clrs_per_class = [reds[10],blues[10]]
plt_args = {
    'markersize':10,
    'linewidth':3,
    'markeredgewidth':2
}

# Plot latent node embeddings
fig = plt.figure(figsize=(6,4))
ax = fig.subplots()
ax.grid(True)
for iy in [0,1]:
    # for i_samp in range(len(Y_inds[iy])):
    # for i_samp in range(10):
    for i_samp in np.random.permutation(len(Y_inds[iy]))[:10]:
        ax.plot(Z_per_class[iy][i_samp][:,0],Z_per_class[iy][i_samp][:,1],mkrs_per_class[iy],c=clrs_per_class[iy],**plt_args)

# Plot latent graph embeddings
fig = plt.figure(figsize=(6,4))
ax = fig.subplots()
ax.grid(True)
for iy in [0,1]:
    for i_samp in range(len(Y_inds[iy])):
        ax.plot(*np.mean(Z_per_class[iy][i_samp],axis=0),mkrs_per_class[iy],c=clrs_per_class[iy],**plt_args)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# setting = 'FFF'

# Plot prediction error
mkr_args = {
    'marker':'o',
    'markersize':10,
    'markeredgewidth':2,
    'markerfacecolor':reds[3],
    'markeredgecolor':reds[10]
}
line_args = {
    'ls':'-',
    'linewidth':2,
    'c':reds[10]
}

test_pred_acc_list = {PRED_FEAT_TYPE:[gin_list[dt][mt][PRED_FEAT_TYPE].test_pred_acc for dt in range(data_params['num_trials']) for mt in range(model_params['num_trials'])] for PRED_FEAT_TYPE in model_params['PRED_FEAT_TYPES']}
fig = plt.figure(figsize=(6,4))
ax = fig.subplots()
ax.grid(True)
for i_pred,PRED_FEAT_TYPE in enumerate(model_params['PRED_FEAT_TYPES']):
    ax.plot([i_pred]*2,
            [np.mean(test_pred_acc_list[PRED_FEAT_TYPE])-np.std(test_pred_acc_list[PRED_FEAT_TYPE]),
            np.mean(test_pred_acc_list[PRED_FEAT_TYPE])+np.std(test_pred_acc_list[PRED_FEAT_TYPE])],
            **line_args)
    ax.plot([i_pred],np.mean(test_pred_acc_list[PRED_FEAT_TYPE]),**mkr_args)
ax.set_xticks(np.arange(len(model_params['PRED_FEAT_TYPES'])),model_params['PRED_FEAT_TYPES'])
ax.set_ylim([-.05,1.05])
fig.tight_layout()
# fig.savefig(f'testerr_{setting}_scaled.png',dpi=300)


fig = plt.figure(figsize=(6,4))
ax = fig.subplots()
ax.grid(True)
for i_pred,PRED_FEAT_TYPE in enumerate(model_params['PRED_FEAT_TYPES']):
    ax.plot([i_pred]*2,
            [np.mean(test_pred_acc_list[PRED_FEAT_TYPE])-np.std(test_pred_acc_list[PRED_FEAT_TYPE]),
            np.mean(test_pred_acc_list[PRED_FEAT_TYPE])+np.std(test_pred_acc_list[PRED_FEAT_TYPE])],
            **line_args)
    ax.plot([i_pred],np.mean(test_pred_acc_list[PRED_FEAT_TYPE]),**mkr_args)
ax.set_xticks(np.arange(len(model_params['PRED_FEAT_TYPES'])),model_params['PRED_FEAT_TYPES'])
# ax.set_ylim([-.05,1.05])
fig.tight_layout()
# fig.savefig(f'testerr_{setting}_unscaled.png',dpi=300)

print('Testing accuracy:')
for PRED_FEAT_TYPE in model_params['PRED_FEAT_TYPES']:
    print(f"{PRED_FEAT_TYPE}: {100*np.mean(test_pred_acc_list[PRED_FEAT_TYPE]):.2f} ({100*np.std(test_pred_acc_list[PRED_FEAT_TYPE]):.2f})")

# s = ''
# s += 'Testing accuracy:\n'
# for PRED_FEAT_TYPE in model_params['PRED_FEAT_TYPES']:
#     s += f"{PRED_FEAT_TYPE}: {100*np.mean(test_pred_acc_list[PRED_FEAT_TYPE]):.2f} ({100*np.std(test_pred_acc_list[PRED_FEAT_TYPE]):.2f})\n"
# with open(f'testerr_{setting}.txt','w') as f:
#     f.write(s)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%