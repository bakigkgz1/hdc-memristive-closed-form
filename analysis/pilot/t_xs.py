import sys; sys.path.insert(0,'/home/claude/hdc')
import numpy as np, torch, time
from hdc import *
import crosssim_bridge as xb
torch.set_num_threads(8)
rows=[]
for ds in ["synth","digits"]:
  Xtr,ytr,Xte,yte,C=load_dataset(ds,"./data")
  for enc_k,D in [("nonlinear",512),("rp",2048)]:
    enc=make_encoder(enc_k,Xtr.shape[1],D,0,Xtr,center=True); Htr,Hte=enc(Xtr),enc(Xte)
    for meth in ["single","largemargin"]:
      M=train_prototypes(Htr,ytr,C,0 if meth=="single" else 20,seed=0,lr=0.05,margin=(0.09 if meth=="largemargin" else 0))
      for prec in ["fp","binary"]:
        Mq=quantize(M,prec).numpy(); H=Hte.numpy().astype(np.float64); dig=accuracy_digital(torch.tensor(Mq),Hte,yte)
        R=xb.calibrate_adc_bound(Mq,Htr[:5000].numpy().astype(np.float64))
        for dev in xb.DEVICES:
          for per in ["ideal","realistic"]:
            ts=xb.T_GRID[dev]; t0=time.time()
            a=xb.accuracy_crosssim(Mq,H,yte,dev,per,ts,n_chips=4,adc_bound=R).mean(0)
            p=[xb.predict_accuracy(Mq,H,yte,dev,per,t,R,return_clip=True) for t in ts]
            print(f"{ds:7s}{enc_k:10s}{D:5d} {meth:11s}{prec:7s}{dev:10s}{per:10s} dig={dig:.3f} sim={np.round(a,3)} pred={[round(x[0],3) for x in p]} clip={[round(x[1],4) for x in p]} {time.time()-t0:.1f}s",flush=True)
