# PACKAGING NOTE -- this deployment copy differs from the training original in ONE place:
# the DINOv2 backbone is built WITHOUT its pretrained weights by default. Our checkpoint is
# a FULL state_dict, frozen backbone included (which is why it is 1.2 GB), so those weights
# are overwritten by load_state_dict immediately. Fetching them first would cost 1.1 GB and
# an internet connection for a tensor that is discarded. SEG_PRETRAINED=1 restores it.
"""Shared-encoder segmenter: FROZEN DINOv2 backbone (same encoder used to calibrate the generator)
+ a light DPT-style decoder on mid-layer patch tokens. Trained on the calibrated synthetic set."""
import os, glob, random, argparse, math, sys, json
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F

DEV="cuda"; DATA=os.environ.get("SEG_DATA","/home/prusek/mt_enc_exp/train_synth"); LAYERS=[5,11,17,23]; DIM=1024
BACKBONE=os.environ.get("SEG_BACKBONE","dinov2")   # dinov2 | phikon
SEG_INPUT=os.environ.get("SEG_INPUT","raw")        # raw | residual (residual = SAME space as calibration)
SEG_MODE=os.environ.get("SEG_MODE","fg")           # fg (1ch foreground) | ori (K-ch orientation overpass)
K_ORI=6; OUTCH=K_ORI if SEG_MODE=="ori" else 1
SEG_ARCH=os.environ.get("SEG_ARCH","base")         # base | aspp (Phase-2: ASPP multi-scale fusion + deep supervision)
SCALE_MIN=float(os.environ.get("SCALE_MIN","1.0")) # Phase-2 scale-aug: crop a c/scale window then resize to c
SCALE_MAX=float(os.environ.get("SCALE_MAX","1.0")) #   (scale<1 => downsample = smaller MTs; >1 => upsample)
AUX_W=float(os.environ.get("AUX_W","0.4"))         # deep-supervision weight on the DINO-semantic aux head

def resid_norm(a):
    """Background-subtracted residual normalized to [0,1] — the calibration input domain."""
    from scipy.ndimage import gaussian_filter
    r=a/(gaussian_filter(a.astype(np.float32),40)+1e-6)-1.0
    lo,hi=np.percentile(r,(2,98)); return np.clip((r-lo)/(hi-lo+1e-6),0,1).astype(np.float32)

class DirBlock(nn.Module):
    # directional asymmetric convs (1x5 + 5x1) — capture thin curvilinear filaments better than square kernels (CS2-Net)
    def __init__(s,c):
        super().__init__(); s.h=nn.Conv2d(c,c,(1,5),padding=(0,2)); s.v=nn.Conv2d(c,c,(5,1),padding=(2,0)); s.n=nn.GroupNorm(8,c)
    def forward(s,x): return F.relu(s.n(x+s.h(x)+s.v(x)))

class ASPP(nn.Module):
    # atrous spatial pyramid: parallel dilated convs give a MULTI-SCALE receptive field at ONE resolution
    # (DINOv2 blocks are all /14, so multi-scale must come from dilation, not a block-FPN). + global context.
    def __init__(s,c,dils=(1,2,4,8)):
        super().__init__()
        s.branches=nn.ModuleList([nn.Sequential(nn.Conv2d(c,c,3,padding=d,dilation=d),nn.GroupNorm(8,c),nn.ReLU(True)) for d in dils])
        s.pool=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,c,1),nn.ReLU(True))
        s.proj=nn.Sequential(nn.Conv2d(c*(len(dils)+1),c,1),nn.GroupNorm(8,c),nn.ReLU(True))
    def forward(s,x):
        H,W=x.shape[-2:]
        gp=F.interpolate(s.pool(x),size=(H,W),mode="bilinear",align_corners=False)
        return s.proj(torch.cat([b(x) for b in s.branches]+[gp],1))

class DinoSeg(nn.Module):
    def __init__(s, dec=128):
        super().__init__()
        s.bk=BACKBONE
        if s.bk=="phikon":
            from transformers import AutoModel
            s.dino=AutoModel.from_pretrained("owkin/phikon-v2")     # pathology DINOv2-L/14, more MT-sensitive
        else:
            s.dino=torch.hub.load('facebookresearch/dinov2','dinov2_vitl14',
                                  pretrained=(os.environ.get("SEG_PRETRAINED","0")=="1"),
                                  trust_repo=True)
        for p in s.dino.parameters(): p.requires_grad=False
        s.dino.eval()
        s.arch=SEG_ARCH
        s.proj=nn.ModuleList([nn.Sequential(nn.Conv2d(DIM,dec,1),nn.GroupNorm(8,dec),nn.ReLU(True)) for _ in LAYERS])
        if s.arch=="aspp":
            s.reduce=nn.Sequential(nn.Conv2d(dec*len(LAYERS),dec,1),nn.GroupNorm(8,dec),nn.ReLU(True))
            s.aspp=ASPP(dec)
            s.aux_head=nn.Conv2d(dec,OUTCH,1)                    # deep supervision on the DINO-semantic pathway
        else:
            s.fuse=nn.Sequential(nn.Conv2d(dec*len(LAYERS),dec,3,padding=1),nn.GroupNorm(8,dec),nn.ReLU(True),
                                 nn.Conv2d(dec,dec,3,padding=1),nn.GroupNorm(8,dec),nn.ReLU(True))
        # HIGH-RES detail branch on the raw image (fixes DINOv2's coarse /14 patch resolution for thin MTs)
        s.hr=nn.Sequential(nn.Conv2d(3,48,3,padding=1),nn.GroupNorm(8,48),nn.ReLU(True),DirBlock(48),
                           nn.Conv2d(48,dec,3,padding=1),nn.GroupNorm(8,dec),nn.ReLU(True),DirBlock(dec),DirBlock(dec))
        s.merge=nn.Sequential(nn.Conv2d(dec*2,dec,3,padding=1),nn.GroupNorm(8,dec),nn.ReLU(True),DirBlock(dec),
                              nn.Conv2d(dec,dec,3,padding=1),nn.GroupNorm(8,dec),nn.ReLU(True),DirBlock(dec))
        s.head=nn.Conv2d(dec,OUTCH,1)
    def _feats(s,x):
        if s.bk=="phikon":
            hs=s.dino(pixel_values=x,output_hidden_states=True,interpolate_pos_encoding=True).hidden_states
            hh=ww=x.shape[-1]//16                               # Phikon-v2 is patch16
            out=[]
            for L in LAYERS:
                t=hs[L+1]; t=t[:,t.shape[1]-hh*ww:,:]           # drop cls prefix
                out.append(t.transpose(1,2).reshape(t.shape[0],t.shape[2],hh,ww))
            return out
        return s.dino.get_intermediate_layers(x,n=LAYERS,reshape=True,norm=True)
    def forward(s,x):
        H,W=x.shape[-2:]
        with torch.no_grad():
            feats=s._feats(x)   # list (B,DIM,h,w)
        p=torch.cat([s.proj[i](feats[i]) for i in range(len(LAYERS))],1)
        f=s.aspp(s.reduce(p)) if s.arch=="aspp" else s.fuse(p)
        f=F.interpolate(f,size=(H,W),mode="bilinear",align_corners=False)   # DINO semantic guidance @ full res
        g=s.hr(x)                                                            # fine detail @ full res
        main=s.head(s.merge(torch.cat([f,g],1)))
        if s.arch=="aspp" and s.training:
            return main, s.aux_head(f)                                       # (main, aux) for deep supervision
        return main

IMA_M=torch.tensor([0.485,0.456,0.406]).view(3,1,1); IMA_S=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
class DS(torch.utils.data.Dataset):
    def __init__(s,crop=518): s.imgs=sorted(glob.glob(DATA+"/images/*.png")); s.crop=crop
    def __len__(s): return len(s.imgs)
    def __getitem__(s,i):
        im=np.asarray(Image.open(s.imgs[i]).convert("L"),np.float32)/255.
        if random.random()<.5: im=1.0-im   # POLARITY augmentation: whole-frame inversion (moved out of the generator)
        if SEG_MODE=="ori":
            g=np.load(s.imgs[i].replace("images","ori").replace(".png",".npy")).astype(np.float32)  # (K,H,W)
        else:
            g=np.asarray(Image.open(s.imgs[i].replace("images","masks")).convert("L"),np.float32)[None]/255.  # (1,H,W)
        if SEG_INPUT=="residual": im=resid_norm(im)
        H,W=im.shape; c=s.crop; y=random.randint(0,H-c); x=random.randint(0,W-c)
        im=im[y:y+c,x:x+c]; g=g[:,y:y+c,x:x+c]
        # geometric flips: for ORI, a flip both mirrors space AND reverses the orientation-bin order
        # (a left-right or up-down flip maps tangent theta -> pi-theta -> bin b -> K-1-b).
        if random.random()<.5: im=im[:,::-1].copy(); g=(g[::-1,:,::-1] if SEG_MODE=="ori" else g[:,:,::-1]).copy()
        if random.random()<.5: im=im[::-1].copy();   g=(g[::-1,::-1,:] if SEG_MODE=="ori" else g[:,::-1,:]).copy()
        t=torch.from_numpy(im)[None].repeat(3,1,1); t=(t-IMA_M)/IMA_S
        return t, torch.from_numpy(g)

class OnlineDS(torch.utils.data.Dataset):
    """ON-THE-FLY parallel generation: every __getitem__ makes a FRESH synthetic frame (infinite data, no disk,
    no overfit to a fixed set). Same preprocessing/aug as DS. gen_fn=generate_frame, ori_fn=ori_channels."""
    def __init__(s, cfg, bg_paths, gen_fn, ori_fn, crop=518, gen=640, epoch_len=5000):
        s.cfg=cfg; s.bg=bg_paths; s.gen_fn=gen_fn; s.ori_fn=ori_fn; s.crop=crop; s.gen=gen; s.epoch_len=epoch_len
    def __len__(s): return s.epoch_len
    def __getitem__(s, i):
        wi=torch.utils.data.get_worker_info(); wid=wi.id if wi else 0
        rng=np.random.default_rng((((wid+1)*1000003)*(i+1)) ^ random.getrandbits(48))   # fresh every call
        g=s.gen; bg=None
        for _ in range(8):
            b=np.asarray(Image.open(s.bg[int(rng.integers(len(s.bg)))]).convert("F"),np.float32)
            while b.ndim>2: b=b[...,0]
            if b.shape[0]>=g and b.shape[1]>=g: bg=b; break
        if bg is None:
            bg=np.asarray(Image.fromarray(b).resize((max(g,b.shape[1]),max(g,b.shape[0]))),np.float32)
        H,W=bg.shape; y0=int(rng.integers(0,H-g+1)); x0=int(rng.integers(0,W-g+1)); bg=bg[y0:y0+g,x0:x0+g]
        img,inst,_=s.gen_fn(bg,rng,s.cfg)
        lo,hi=np.percentile(img,[1,99]); im=np.clip((img-lo)/(hi-lo+1e-6),0,1).astype(np.float32)
        if random.random()<.5: im=1.0-im
        if SEG_MODE=="ori": gt=s.ori_fn(inst,(g,g)).astype(np.float32)
        else:
            fg=np.zeros((g,g),bool)
            for ins in inst: fg|=ins["mask"]
            gt=fg[None].astype(np.float32)
        c=s.crop
        # SCALE AUGMENTATION (Phase-2): crop a c/scale window then resize to c -> the SAME MTs at a different
        # apparent resolution -> scale-robust foreground (targets cross-scope mislocalization, e.g. HTW).
        cs=int(round(c/random.uniform(SCALE_MIN,SCALE_MAX))) if SCALE_MIN<SCALE_MAX else c
        cs=max(96,min(cs,g))
        y=random.randint(0,g-cs); x=random.randint(0,g-cs); im=im[y:y+cs,x:x+cs]; gt=gt[:,y:y+cs,x:x+cs]
        if cs!=c:
            im=np.asarray(Image.fromarray(im).resize((c,c),Image.BILINEAR),np.float32)
            gt=np.stack([np.asarray(Image.fromarray(gt[k]).resize((c,c),Image.BILINEAR),np.float32) for k in range(gt.shape[0])])
        if random.random()<.5: im=im[:,::-1].copy(); gt=(gt[::-1,:,::-1] if SEG_MODE=="ori" else gt[:,:,::-1]).copy()
        if random.random()<.5: im=im[::-1].copy();   gt=(gt[::-1,::-1,:] if SEG_MODE=="ori" else gt[:,::-1,:]).copy()
        t=torch.from_numpy(im)[None].repeat(3,1,1); t=(t-IMA_M)/IMA_S
        return t, torch.from_numpy(gt)

def soft_erode(p): return torch.min(-F.max_pool2d(-p,(3,1),1,(1,0)),-F.max_pool2d(-p,(1,3),1,(0,1)))
def soft_skel(p,iters=8):
    p1=F.max_pool2d(soft_erode(p),3,1,1); skel=F.relu(p-p1)
    for _ in range(iters):
        p=soft_erode(p); p1=F.max_pool2d(soft_erode(p),3,1,1); d=F.relu(p-p1); skel=skel+F.relu(d-skel*d)
    return skel
def soft_cldice(pl,tl,e=1.):
    p=torch.sigmoid(pl).amax(1,keepdim=True); t=tl.amax(1,keepdim=True); sp=soft_skel(p); st=soft_skel(t)
    tprec=(( sp*t).sum((2,3))+e)/(sp.sum((2,3))+e); tsens=((st*p).sum((2,3))+e)/(st.sum((2,3))+e)
    return (1-2.0*tprec*tsens/(tprec+tsens)).mean()
def dice(p,t,e=1.): p=torch.sigmoid(p); i=(p*t).sum((2,3)); return (1-(2*i+e)/(p.sum((2,3))+t.sum((2,3))+e)).mean()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--epochs",type=int,default=30); a=ap.parse_args()
    if os.environ.get("ONLINE"):                                          # on-the-fly PARALLEL generation
        sys.path.insert(0,"/home/prusek/mt_enc_exp/synth"); sys.path.insert(0,"/home/prusek/mt_enc_exp/scripts")
        from mt_generator import generate_frame as _gf
        from gen_train import ori_channels as _oc, build_cfg as _bc
        _p=json.load(open(os.environ["CALIB"]))["best_params"]; _mhw=float(os.environ.get("MASK_HW","1.0"))
        if os.environ.get("DR")=="1":
            from dr_cfg import build_cfg_dr as _bcdr; _cfg=_bcdr(_p, mask_hw=_mhw); print("DR=1 domain-randomization cfg",flush=True)
        else:
            _cfg=_bc(_p, mask_hw=_mhw)
        _bg=sorted(glob.glob("/home/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2/*.tif"))
        ds=OnlineDS(_cfg,_bg,_gf,_oc,epoch_len=int(os.environ.get("EPOCH_LEN","5000")))
        dl=torch.utils.data.DataLoader(ds,batch_size=6,num_workers=int(os.environ.get("NWORKERS","10")),
                                       drop_last=True,persistent_workers=True,prefetch_factor=3)
        print(f"ONLINE gen: {os.environ.get('NWORKERS','10')} workers epoch_len={os.environ.get('EPOCH_LEN','5000')}",flush=True)
    else:
        dl=torch.utils.data.DataLoader(DS(),batch_size=6,shuffle=True,num_workers=4,drop_last=True)
    m=DinoSeg().to(DEV)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],1e-3)
    for ep in range(a.epochs):
        m.train(); m.dino.eval(); tot=0
        pw=torch.tensor(float(os.environ.get("POS_W","1")),device=DEV); cw=float(os.environ.get("CLDICE_W","0.5"))
        def sloss(o): return F.binary_cross_entropy_with_logits(o,mk,pos_weight=pw)+dice(o,mk)+cw*soft_cldice(o,mk)
        for im,mk in dl:
            im,mk=im.to(DEV),mk.to(DEV); out=m(im); aux=None
            if isinstance(out,(tuple,list)): out,aux=out
            loss=sloss(out)+(AUX_W*sloss(aux) if aux is not None else 0.0)   # deep supervision
            opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
        print(f"epoch {ep+1}/{a.epochs} loss={tot/len(dl):.4f}",flush=True)
    torch.save(m.state_dict(),"/home/prusek/mt_enc_exp/dino_seg.pth"); print("saved",flush=True)
if __name__=="__main__": main()
