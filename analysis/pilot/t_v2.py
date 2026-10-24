import sys; sys.path.insert(0,'/home/claude/hdc')
import torch, numpy as np, time
from hdc import *
from analytic import predict_accuracy_analytic, predict_accuracy_analytic2, dac_scale
torch.set_num_threads(8)
for ds in ["synth","digits"]:
  Xtr,ytr,Xte,yte,C=load_dataset(ds,"./data")
  for enc_k,D in [("nonlinear",512),("rp",512),("nonlinear",2048)]:
    enc=make_encoder(enc_k,Xtr.shape[1],D,0,Xtr,center=True); Htr,Hte=enc(Xtr),enc(Xte)
    for meth in ["single","largemargin"]:
      M=train_prototypes(Htr,ytr,C,0 if meth=="single" else 20,seed=0,lr=0.05,margin=(0.09 if meth=="largemargin" else 0))
      for prec in ["fp","binary"]:
        Mq=quantize(M,prec); dig=accuracy_digital(Mq,Hte,yte); s=calibrate_input_scale(Mq,Htr[:5000])
        for dev in ["reram_wan","pcm","reram_cmo"]:
          cfg=make_inference_config(dev,"realistic"); times=[1.0,86400.0]
          a=accuracy_crossbar(Mq,Hte,yte,cfg,times,n_chips=4,in_scale=s).mean(0)
          v1=[predict_accuracy_analytic(Mq,Hte,yte,dev,t,s) for t in times]
          v2=[predict_accuracy_analytic2(Mq,Hte,yte,dev,t,s,return_parts=True) for t in times]
          print(f"{ds:7s}{enc_k:10s}{D:5d} {meth:11s}{prec:7s}{dev:10s} s={s:.4f} seff={dac_scale(s):.4f} dig={dig:.3f} sim={np.round(a,3)} v1={np.round(v1,3)} v2={[round(x[0],3) for x in v2]} clip={[round(x[1],4) for x in v2]}",flush=True)
