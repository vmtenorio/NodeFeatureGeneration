import numpy as np
import matplotlib.pyplot as plt

madcmap = 'Reds'
num_total = 25

blue_base = np.array([.267,.467,.831])
blue_min = np.array([.800,.875,1.00])
blue_max = np.array([.090,.165,.302])
blues = [blue_min]
blues1 = list((blue_min[None] + (blue_base-blue_min)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
blues2 = list((blue_base[None] + (blue_max-blue_base)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
blues += blues1 + blues2

gray_base = np.array([.584,.588,.592])
gray_min = np.array([.894,.894,.894])
gray_max = np.array([.216,.220,.224])
grays = [gray_min]
grays1 = list((gray_min[None] + (gray_base-gray_min)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
grays2 = list((gray_base[None] + (gray_max-gray_base)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
grays += grays1 + grays2

red_base = np.array([.831,.267,.443])
red_min = np.array([.969,.835,.878])
red_max = np.array([.302,.090,.157])
reds = [red_min]
reds1 = list((red_min[None] + (red_base-red_min)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
reds2 = list((red_base[None] + (red_max-red_base)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
reds += reds1 + reds2

green_base = np.array([.337,0.761,0.620])
green_min = np.array([.792,0.933,0.886])
green_max = np.array([.059,0.333,0.243])
greens = [green_min]
greens1 = list((green_min[None] + (green_base-green_min)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
greens2 = list((green_base[None] + (green_max-green_base)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
greens += greens1 + greens2

yellow_base = np.array([.867,0.608,0.231])
yellow_min = np.array([.984,0.855,0.663])
yellow_max = np.array([.392,0.235,0.000])
yellows = [yellow_min]
yellows1 = list((yellow_min[None] + (yellow_base-yellow_min)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
yellows2 = list((yellow_base[None] + (yellow_max-yellow_base)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
yellows += yellows1 + yellows2

purple_base = np.array([.576,.463,.816])
purple_min = np.array([.812,.757,.980])
purple_max = np.array([.282,.125,.498])
purples = [purple_min]
purples1 = list((purple_min[None] + (purple_base-purple_min)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
purples2 = list((purple_base[None] + (purple_max-purple_base)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
purples += purples1 + purples2

def madimshow(mat,cmap:str=madcmap,xlabel:str='',ylabel:str='',set_axis=True,figsize=(4,4),vmin=None,vmax=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.subplots()
    args = {'cmap':cmap}
    if vmin is not None:
        args['vmin'] = vmin
    if vmax is not None:
        args['vmax'] = vmax
    ax.imshow(mat,**args)
    if len(xlabel)>0:
        ax.set_xlabel(xlabel)
    if len(ylabel)>0:
        ax.set_ylabel(ylabel)
    if not set_axis:
        ax.axis('off')
    fig.tight_layout()

# ----------------------------------------

def mat2lowtri(A,N=None):
    if N is None:
        N=A.shape[0] if type(A)==np.ndarray else A.type.shape[0]
    low_tri_indices=np.where(np.triu(np.ones((N,N)))-np.eye(N))
    return A[low_tri_indices[1],low_tri_indices[0]]

def lowtri2mat(a,L=None):
    if L is None:
        L=len(a) if type(a)==np.ndarray else a.type.shape[0]
    N=int(.5+np.sqrt(2*L+.25))
    A=np.full((N,N),0,dtype=type(a[0]))
    low_tri_indices=np.where(np.triu(np.ones((N,N)))-np.eye(N))
    A[low_tri_indices[1],low_tri_indices[0]]=a
    A+=A.T
    return A

def vec(X):
    x=np.linalg.vstack(X)
    return x

def kchoose2(k:int):
    assert k>0, 'Invalid integer.'
    return int(k*(k-1)/2)

def kchoose3(k:int):
    assert k>0, 'Invalid integer.'
    return int(k*(k-1)*(k-2)/6)

# ----------------------------------------

def erdos_renyi(N=20,edge_prob=.2):
    return lowtri2mat(np.random.binomial(1,edge_prob,int(N*(N-1)/2)))

def ksbm(N=20,k=2,in_prob=.8,out_prob=.1,block_assign=None):
    if block_assign is None:
        block_assign=np.random.choice(k,N)
    else:
        assert len(block_assign)==N
    A=np.zeros((N,N),dtype=int)
    in_inds=np.where(block_assign[:,None]==block_assign[None])
    in_inds=tuple(np.sort(in_inds,axis=0))
    out_inds=np.where(block_assign[:,None]!=block_assign[None])
    out_inds=tuple(np.sort(out_inds,axis=0))

    A[in_inds]=np.random.binomial(1,in_prob,len(in_inds[0]))
    A[out_inds]=np.random.binomial(1,out_prob,len(out_inds[0]))
    A=A+A.T
    A[np.eye(N)==1]=0
    return A

def W(num_nodes:int=20,graphon_func=None):
    if graphon_func is None:
        graphon_func = lambda u: .5*(u[:,None] + u[None])
    u = np.random.rand(num_nodes)
    prob_mat = graphon_func(u)
    return lowtri2mat(np.random.binomial(1,mat2lowtri(prob_mat)))

# ----------------------------------------

# TODO: Complete list of local structural features
LS_FEAT_LIST = ['degree','egonet_dens','egonet_deg','egonet_ratio','nonegonet_ratio','cliques3','clustcoef']
def compute_ls_feat(A,ls_type:str='degree'):
    assert ls_type in LS_FEAT_LIST, 'Invalid local structure feature type.'
    (num_nodes,num_nodes) = A.shape
    degs = np.sum(A,axis=0)
    egonet_inds = list(map(lambda i:np.concatenate(([i],np.where(A[i]==1)[0])),np.arange(num_nodes)))
    egonet = list(map(lambda inds:A[inds][:,inds],egonet_inds))

    if ls_type=='degree':
        return degs
    elif ls_type=='egonet_dens':
        return np.array(list(map(np.sum,egonet)))/2
    elif ls_type=='egonet_deg':
        return np.array(list(map(lambda inds:np.sum(degs[inds])/2,egonet_inds)))
    elif ls_type=='egonet_ratio':
        egonet_dens = np.array(list(map(np.sum,egonet)))/2
        egonet_deg = np.array(list(map(lambda inds:np.sum(degs[inds]),egonet_inds)))/2
        return np.array([egonet_dens[i]/egonet_deg[i] if egonet_deg[i]>0 else 0 for i in range(num_nodes)])
    elif ls_type=='nonegonet_ratio':
        egonet_dens = np.array(list(map(np.sum,egonet)))/2
        egonet_deg = np.array(list(map(lambda inds:np.sum(degs[inds]),egonet_inds)))/2
        return np.array([1 - egonet_dens[i]/egonet_deg[i] if egonet_deg[i]>0 else 0 for i in range(num_nodes)])
    elif ls_type=='cliques3':
        return np.diag(np.linalg.matrix_power(A,3))/2
    elif ls_type=='clustcoef':
        degs = np.sum(A,axis=0)
        cliques3 = np.diag(np.linalg.matrix_power(A,3))/2
        return np.array([2*cliques3[i]/(degs[i]*(degs[i]-1)) if degs[i]>1 else 0 for i in range(num_nodes)])
    else:
        print('Invalid local structure feature type.')
        return

# TODO: Complete list of global structural features. Options may include ratio of triangles, ratio of k-stars, eigendecomposition, pagerank
GS_FEAT_LIST = ['degree_dist','egonet_dens_ratio','egonet_deg_ratio','cliques3_ratio']
def compute_gs_feat(A,gs_type:str='degree_dist'):
    assert gs_type in GS_FEAT_LIST, 'Invalid global structure feature type.'
    (num_nodes,num_nodes) = A.shape
    degs = np.sum(A,axis=0)
    egonet_inds = list(map(lambda i:np.concatenate(([i],np.where(A[i]==1)[0])),np.arange(num_nodes)))
    egonet = list(map(lambda inds:A[inds][:,inds],egonet_inds))

    if gs_type=='degree_dist':
        return degs/(num_nodes-1)
    elif gs_type=='egonet_dens_ratio':
        return np.array(list(map(np.sum,egonet)))/2/kchoose2(num_nodes)
    elif gs_type=='egonet_deg_ratio':
        return np.array(list(map(lambda inds:np.sum(degs[inds])/2,egonet_inds)))/kchoose2(num_nodes)
    elif gs_type=='cliques3_ratio':
        return np.diag(np.linalg.matrix_power(A,3))/2/kchoose2(num_nodes-1)
    # TODO: Include global clustering coefficient
    else:
        print('Invalid global structure feature type.')
        return

# TODO: Normalize features by constant, consistent across all samples in graph dataset
def compute_features(A:np.ndarray):
    assert A.shape[0]==A.shape[1], 'Invalid adjacency matrix.'
    (num_nodes,num_nodes) = A.shape
    degs = np.sum(A,axis=0)
    egonet_inds = list(map(lambda i:np.concatenate(([i],np.where(A[i]==1)[0])),np.arange(num_nodes)))
    egonet = list(map(lambda inds:A[inds][:,inds],egonet_inds))
    egonet_dens = np.array(list(map(np.sum,egonet)))/2
    egonet_deg = np.array(list(map(lambda inds:np.sum(degs[inds]),egonet_inds)))/2
    cliques3 = np.diag(np.linalg.matrix_power(A,3))/2

    f = [None]*7
    f[0] = degs
    f[1] = egonet_dens
    f[2] = egonet_deg
    f[3] = [egonet_dens[i]/egonet_deg[i] if egonet_deg[i]>0 else 0 for i in range(num_nodes)]
    f[4] = [1-egonet_dens[i]/egonet_deg[i] if egonet_deg[i]>0 else 0 for i in range(num_nodes)]
    f[5] = cliques3
    f[6] = [2*cliques3[i]/(degs[i]*(degs[i]-1)) if degs[i]>1 else 0 for i in range(num_nodes)]
    Ft = np.array(f)

    Ah = A + np.eye(num_nodes)
    Dh = np.diag(degs+1)
    F = np.concatenate([Ft.T, np.linalg.inv(Dh)@Ah@Ft.T, Ah@Ft.T],axis=1)

    return F

