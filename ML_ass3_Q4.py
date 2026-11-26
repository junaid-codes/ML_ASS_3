!pip -q install torch torchvision scikit-learn matplotlib numpy pandas

import os, random, json, time
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

SEED=26001
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE)
OUT='q3_q4_results'
os.makedirs(OUT,exist_ok=True)
CLASS_NAMES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

transform=transforms.ToTensor()
train_full=datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
test_dataset=datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
y_all=np.array(train_full.targets)
splitter=StratifiedShuffleSplit(n_splits=1,test_size=5000,random_state=SEED)
train_idx,val_idx=next(splitter.split(np.zeros(len(y_all)),y_all))
train_ds=Subset(train_full,train_idx); val_ds=Subset(train_full,val_idx)
BATCH=128
train_loader=DataLoader(train_ds,batch_size=BATCH,shuffle=True,num_workers=2,pin_memory=True)
val_loader=DataLoader(val_ds,batch_size=BATCH,shuffle=False,num_workers=2,pin_memory=True)
test_loader=DataLoader(test_dataset,batch_size=BATCH,shuffle=False,num_workers=2,pin_memory=True)
print(len(train_ds),len(val_ds),len(test_dataset))

class ConvAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder_conv=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.to_latent=nn.Linear(128*4*4,latent_dim)
        self.from_latent=nn.Sequential(nn.Linear(latent_dim,128*4*4),nn.ReLU(),
            nn.Unflatten(1,(128,4,4)))
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(128,64,4,stride=2,padding=1),nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,stride=2,padding=1),nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,stride=2,padding=1),nn.Sigmoid())
    def encode(self,x):
        h=self.encoder_conv(x).flatten(1)
        return self.to_latent(h)
    def decode(self,z):
        return self.decoder(self.from_latent(z))
    def forward(self,x):
        z=self.encode(x)
        return self.decode(z)

def nparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_ae(model, train_loader, val_loader, epochs=10, noise_sigma=0.0):
    model.to(DEVICE); optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    history={'train_mse':[],'val_mse':[]}
    for epoch in range(epochs):
        model.train(); train_losses=[]
        for x,_ in train_loader:
            x=x.to(DEVICE)
            xin=(x+noise_sigma*torch.randn_like(x)).clamp(0,1) if noise_sigma else x
            optimizer.zero_grad()
            loss=F.mse_loss(model(xin),x)
            loss.backward(); optimizer.step()
            train_losses.append(loss.item())
        model.eval(); val_losses=[]
        with torch.no_grad():
            for x,_ in val_loader:
                x=x.to(DEVICE)
                xin=(x+noise_sigma*torch.randn_like(x)).clamp(0,1) if noise_sigma else x
                val_losses.append(F.mse_loss(model(xin),x).item())
        history['train_mse'].append(float(np.mean(train_losses)))
        history['val_mse'].append(float(np.mean(val_losses)))
        print(f'Epoch {epoch+1}/{epochs}: train={history["train_mse"][-1]:.5f}, val={history["val_mse"][-1]:.5f}')
    return history

EPOCHS_Q3=10                                                         
latent_sizes=[16,64,256]
ae_models={}; ae_histories={}; ae_results=[]
for zdim in latent_sizes:
    print('\nTraining bottleneck',zdim)
    model=ConvAE(zdim)
    hist=train_ae(model,train_loader,val_loader,epochs=EPOCHS_Q3)
    ae_models[zdim]=model; ae_histories[zdim]=hist
    ae_results.append({'latent_dim':zdim,'parameters':nparams(model),'best_val_mse':min(hist['val_mse']),'final_val_mse':hist['val_mse'][-1]})
    torch.save(model.state_dict(),f'{OUT}/ae_{zdim}.pt')
pd.DataFrame(ae_results)

plt.figure(figsize=(7,4))
for zdim,h in ae_histories.items(): plt.plot(h['val_mse'],marker='o',label=f'Latent {zdim}')
plt.xlabel('Epoch'); plt.ylabel('Validation reconstruction MSE'); plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(f'{OUT}/q3_bottleneck_val_mse.png',dpi=160); plt.show()

for zdim,h in ae_histories.items():
    plt.figure(figsize=(6,4)); plt.plot(h['train_mse'],label='Train'); plt.plot(h['val_mse'],label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.title(f'AE latent={zdim}'); plt.legend(); plt.tight_layout()
    plt.savefig(f'{OUT}/ae_{zdim}_curves.png',dpi=160); plt.show()

pd.DataFrame(ae_results).to_csv(f'{OUT}/q3_bottleneck_results.csv',index=False)

def show_reconstructions(model, loader, title, n=8):
    model.eval(); x,_=next(iter(loader)); x=x[:n].to(DEVICE)
    with torch.no_grad(): rec=model(x).cpu()
    x=x.cpu()
    fig,axs=plt.subplots(2,n,figsize=(2*n,4))
    for i in range(n):
        axs[0,i].imshow(x[i].permute(1,2,0)); axs[0,i].axis('off')
        axs[1,i].imshow(rec[i].permute(1,2,0)); axs[1,i].axis('off')
    axs[0,0].set_ylabel('Original'); axs[1,0].set_ylabel('Reconstructed')
    plt.suptitle(title); plt.tight_layout(); plt.show()
for zdim in latent_sizes: show_reconstructions(ae_models[zdim],test_loader,f'AE bottleneck {zdim}')

NOISE_SIGMA=0.25
denoiser=ConvAE(latent_dim=64)
denoise_history=train_ae(denoiser,train_loader,val_loader,epochs=EPOCHS_Q3,noise_sigma=NOISE_SIGMA)
torch.save(denoiser.state_dict(),f'{OUT}/denoising_ae_64.pt')
pd.DataFrame(denoise_history).to_csv(f'{OUT}/q3_denoising_history.csv',index=False)
plt.figure(figsize=(6,4)); plt.plot(denoise_history['train_mse'],label='Train'); plt.plot(denoise_history['val_mse'],label='Validation')
plt.xlabel('Epoch'); plt.ylabel('Clean-target MSE'); plt.title(f'Denoising AE, noise sigma={NOISE_SIGMA}'); plt.legend(); plt.tight_layout()
plt.savefig(f'{OUT}/q3_denoising_curve.png',dpi=160); plt.show()

denoiser.eval()
x,y=next(iter(test_loader)); x=x[:8].to(DEVICE)
noisy=(x+NOISE_SIGMA*torch.randn_like(x)).clamp(0,1)
with torch.no_grad(): reconstructed=denoiser(noisy)
fig,axs=plt.subplots(3,8,figsize=(16,6))
for i in range(8):
    for row,img in enumerate([x,noisy,reconstructed]):
        axs[row,i].imshow(img[i].cpu().permute(1,2,0)); axs[row,i].axis('off')
axs[0,0].set_ylabel('Clean'); axs[1,0].set_ylabel('Noisy'); axs[2,0].set_ylabel('Denoised')
plt.tight_layout(); plt.savefig(f'{OUT}/q3_denoising_examples.png',dpi=160); plt.show()

def extract_latents(model, dataset, limit=None):
    model.eval()
    loader=DataLoader(dataset,batch_size=256,shuffle=False,num_workers=2)
    feats=[]; labs=[]
    with torch.no_grad():
        for xb,yb in loader:
            feats.append(model.encode(xb.to(DEVICE)).cpu().numpy()); labs.append(yb.numpy())
            if limit is not None and sum(len(a) for a in labs)>=limit: break
    X=np.concatenate(feats); y=np.concatenate(labs)
    if limit is not None: X=X[:limit]; y=y[:limit]
    return X,y

def stratified_indices(labels, n, seed=SEED):
    if n>=len(labels): return np.arange(len(labels))
    splitter=StratifiedShuffleSplit(n_splits=1,train_size=n,random_state=seed)
    idx,_=next(splitter.split(np.zeros(len(labels)),labels))
    return idx

train_labels=np.array([train_full.targets[i] for i in train_idx])
val_labels=np.array([train_full.targets[i] for i in val_idx])
test_labels=np.array(test_dataset.targets)
Q3_PROBE_TRAIN=6000; Q3_PROBE_VAL=3000; Q3_PROBE_TEST=3000
tr_sub_idx=stratified_indices(train_labels,Q3_PROBE_TRAIN)
va_sub_idx=stratified_indices(val_labels,Q3_PROBE_VAL)
te_sub_idx=stratified_indices(test_labels,Q3_PROBE_TEST)
probe_train=Subset(train_ds,tr_sub_idx); probe_val=Subset(val_ds,va_sub_idx); probe_test=Subset(test_dataset,te_sub_idx)
probe_results=[]
for zdim,model in list(ae_models.items())+[('denoising',denoiser)]:
    Xtr,ytr=extract_latents(model,probe_train); Xv,yv=extract_latents(model,probe_val); Xt,yt=extract_latents(model,probe_test)
    scaler=StandardScaler().fit(Xtr); Xtr=scaler.transform(Xtr); Xv=scaler.transform(Xv); Xt=scaler.transform(Xt)
    best=None
    for C in [0.1,1,10]:
        probe=SVC(kernel='linear',C=C)
        probe.fit(Xtr,ytr); va=accuracy_score(yv,probe.predict(Xv))
        if best is None or va>best[0]: best=(va,C)
    probe=SVC(kernel='linear',C=best[1]).fit(Xtr,ytr)
    testacc=accuracy_score(yt,probe.predict(Xt))
    probe_results.append({'representation':str(zdim),'train_n':len(ytr),'val_n':len(yv),'test_n':len(yt),'best_C':best[1],'val_acc':best[0],'test_acc':testacc})
    print(probe_results[-1])
pd.DataFrame(probe_results).to_csv(f'{OUT}/q3_latent_probe_results.csv',index=False)
pd.DataFrame(probe_results)

def get_flat_images(dataset, indices=None):
    if indices is None: indices=np.arange(len(dataset))
    loader=DataLoader(Subset(dataset,indices),batch_size=256,shuffle=False,num_workers=2)
    X=[]; y=[]
    for xb,yb in loader: X.append(xb.flatten(1).numpy()); y.append(yb.numpy())
    return np.concatenate(X),np.concatenate(y)

Q4_TRAIN_N=5000; Q4_VAL_N=2000; Q4_TEST_N=2000
q4_tr_idx=stratified_indices(train_labels,Q4_TRAIN_N)
q4_va_idx=stratified_indices(val_labels,Q4_VAL_N)
q4_te_idx=stratified_indices(test_labels,Q4_TEST_N)
Xtr,ytr=get_flat_images(train_ds,q4_tr_idx)
Xv,yv=get_flat_images(val_ds,q4_va_idx)
Xt,yt=get_flat_images(test_dataset,q4_te_idx)
scaler_q4=StandardScaler().fit(Xtr)
Xtr_s=scaler_q4.transform(Xtr); Xv_s=scaler_q4.transform(Xv); Xt_s=scaler_q4.transform(Xt)
print('Q4 subset sizes:',len(ytr),len(yv),len(yt))

kernel_configs=[
    {'kernel':'linear','C':0.1},
    {'kernel':'linear','C':1},
    {'kernel':'linear','C':10},
    {'kernel':'rbf','C':1,'gamma':'scale'},
    {'kernel':'rbf','C':10,'gamma':'scale'},
    {'kernel':'rbf','C':10,'gamma':0.0001},
    {'kernel':'poly','C':1,'degree':2},
    {'kernel':'poly','C':1,'degree':3}
]
svm_sweep=[]; fitted_svm={}
for cfg in kernel_configs:
    print('Fitting',cfg)
    svm=SVC(**cfg,decision_function_shape='ovr')
    svm.fit(Xtr_s,ytr)
    predv=svm.predict(Xv_s); va=accuracy_score(yv,predv)
    support_fraction=len(svm.support_)/len(ytr)
    row={**cfg,'val_accuracy':float(va),'support_vectors':int(len(svm.support_)),'support_fraction':float(support_fraction)}
    svm_sweep.append(row); fitted_svm[str(cfg)]=svm
    print(row)
sweep_df=pd.DataFrame(svm_sweep).sort_values('val_accuracy',ascending=False)
sweep_df.to_csv(f'{OUT}/q4_svm_sweep.csv',index=False)
sweep_df

fig,ax=plt.subplots(figsize=(10,4))
labels=[f'{r["kernel"]}\n{r.get("C")}\n{r.get("degree",r.get("gamma",""))}' for r in svm_sweep]
ax.bar(labels,[r['val_accuracy'] for r in svm_sweep]); ax.set_ylabel('Validation accuracy'); ax.set_ylim(0,1)
ax.set_title('SVM validation kernel sweep'); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(f'{OUT}/q4_kernel_val_accuracy.png',dpi=160); plt.show()
fig,ax=plt.subplots(figsize=(10,4)); ax.bar(labels,[r['support_fraction'] for r in svm_sweep])
ax.set_ylabel('Fraction of training samples that are support vectors'); ax.set_title('Support-vector fraction'); plt.tight_layout()
plt.savefig(f'{OUT}/q4_support_fraction.png',dpi=160); plt.show()

best_row=sweep_df.iloc[0].to_dict()
best_cfg={k:best_row[k] for k in ['kernel','C'] if k in best_row}
if best_row['kernel']=='rbf': best_cfg['gamma']=best_row['gamma']
if best_row['kernel']=='poly': best_cfg['degree']=int(best_row['degree'])
selected_svm=SVC(**best_cfg,decision_function_shape='ovr').fit(Xtr_s,ytr)
test_pred=selected_svm.predict(Xt_s)
test_acc=accuracy_score(yt,test_pred)
print('Selected settings:',best_cfg)
print('Test accuracy on Q4 subset:',test_acc)
print(classification_report(yt,test_pred,target_names=CLASS_NAMES,zero_division=0))
with open(f'{OUT}/q4_selected_svm.json','w') as f: json.dump({'selected':best_cfg,'val_accuracy':float(best_row['val_accuracy']),'test_accuracy':float(test_acc),'train_subset':Q4_TRAIN_N,'val_subset':Q4_VAL_N,'test_subset':Q4_TEST_N,'support_vectors':int(len(selected_svm.support_)),'support_fraction':float(len(selected_svm.support_)/len(ytr))},f,indent=2)
cm=confusion_matrix(yt,test_pred)
plt.figure(figsize=(8,7)); plt.imshow(cm); plt.colorbar(); plt.xticks(range(10),CLASS_NAMES,rotation=45,ha='right'); plt.yticks(range(10),CLASS_NAMES)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Selected SVM test confusion matrix'); plt.tight_layout()
plt.savefig(f'{OUT}/q4_svm_confusion.png',dpi=160); plt.show()

cm=confusion_matrix(yt,test_pred)
pair_counts=[]
for i in range(10):
    for j in range(i+1,10):
        count=cm[i,j]+cm[j,i]
        if count>0: pair_counts.append((count,i,j))
pair_counts=sorted(pair_counts,reverse=True)[:3]
print('Selected pairs:',[(CLASS_NAMES[i],CLASS_NAMES[j],n) for n,i,j in pair_counts])

def subset_images(dataset, indices, max_items=None):
    ims=[]; labs=[]
    for idx in indices:
        im,lab=dataset[int(idx)]; ims.append(im); labs.append(lab)
        if max_items and len(ims)>=max_items: break
    return torch.stack(ims),np.array(labs)
q4_test_images,q4_test_labels=subset_images(test_dataset,q4_te_idx)
q4_train_images,q4_train_labels=subset_images(train_ds,q4_tr_idx)
def show_tensor_images(images,indices,title):
    indices=list(indices)
    if not indices: print('No images to show'); return
    n=len(indices); fig,axs=plt.subplots(1,n,figsize=(2*n,2.5))
    if n==1: axs=[axs]
    for ax,k in zip(axs,indices):
        ax.imshow(images[k].permute(1,2,0)); ax.axis('off')
    plt.suptitle(title); plt.tight_layout(); plt.show()

pair_analysis=[]
for _,a,b in pair_counts:
    mask_tr=np.isin(ytr,[a,b]); mask_te=np.isin(yt,[a,b])
    Xpair=Xtr_s[mask_tr]; ypair=ytr[mask_tr]
    pair_svm=SVC(kernel=best_cfg['kernel'],C=best_cfg['C'],
                  **({'gamma':best_cfg['gamma']} if best_cfg['kernel']=='rbf' else {}),
                  **({'degree':best_cfg['degree']} if best_cfg['kernel']=='poly' else {}))
    pair_svm.fit(Xpair,ypair)
    pairX=Xt_s[mask_te]; pairY=yt[mask_te]
    pair_scores=pair_svm.decision_function(pairX)
    pair_preds=pair_svm.predict(pairX)
    pairacc=accuracy_score(pairY,pair_preds)
    print(f'\nPair {CLASS_NAMES[a]} vs {CLASS_NAMES[b]} accuracy:',pairacc)
                                     
    near=np.argsort(np.abs(pair_scores))[:min(8,len(pair_scores))]
    global_test_indices=np.where(mask_te)[0][near]
    show_tensor_images(q4_test_images,global_test_indices,f'{CLASS_NAMES[a]} vs {CLASS_NAMES[b]}: test examples near boundary')
                     
    err=np.where(pair_preds!=pairY)[0]
    if len(err):
        err_global=np.where(mask_te)[0][err[:8]]
        show_tensor_images(q4_test_images,err_global,f'{CLASS_NAMES[a]} vs {CLASS_NAMES[b]}: errors')
                                   
    sv_local=pair_svm.support_
    pair_train_global=np.where(mask_tr)[0][sv_local[:8]]
    show_tensor_images(q4_train_images,pair_train_global,f'{CLASS_NAMES[a]} vs {CLASS_NAMES[b]}: support vectors')
    pair_analysis.append({'pair':[CLASS_NAMES[a],CLASS_NAMES[b]],'pair_test_accuracy':float(pairacc),'pair_test_count':int(len(pairY)),'support_vectors':int(len(pair_svm.support_))})
pd.DataFrame(pair_analysis).to_csv(f'{OUT}/q4_pair_analysis.csv',index=False)
pd.DataFrame(pair_analysis)

K=10
kmeans=KMeans(n_clusters=K,n_init=10,random_state=SEED)
cluster_tr=kmeans.fit_predict(Xtr_s)
cluster_test=kmeans.predict(Xt_s)
ari=adjusted_rand_score(yt,cluster_test)
nmi=normalized_mutual_info_score(yt,cluster_test)
svm_test_acc=accuracy_score(yt,test_pred)
cluster_result={'method':'KMeans','k':K,'train_n':len(ytr),'test_n':len(yt),'ARI':float(ari),'NMI':float(nmi),'SVM_test_accuracy':float(svm_test_acc)}
print(cluster_result)
with open(f'{OUT}/q4_clustering_metrics.json','w') as f: json.dump(cluster_result,f,indent=2)

perclass=[]
for c in range(10):
    mask=yt==c
    if mask.sum():
        acc=float(accuracy_score(yt[mask],test_pred[mask]))
        dominant_cluster=int(np.bincount(cluster_test[mask],minlength=K).argmax())
        cluster_purity=float(np.bincount(cluster_test[mask],minlength=K).max()/mask.sum())
        perclass.append({'class':CLASS_NAMES[c],'n_test':int(mask.sum()),'SVM_accuracy':acc,'dominant_cluster':dominant_cluster,'dominant_cluster_fraction':cluster_purity})
pd.DataFrame(perclass).to_csv(f'{OUT}/q4_per_class_comparison.csv',index=False)
pd.DataFrame(perclass)
