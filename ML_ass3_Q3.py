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
