import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from crossids.flare import evaluate as ev
from crossids.flare.data import load_split, target_features
T='cicids2017'
ctx=ev.EvalContext(runs_dir=Path('runs/flare'),supports_dir=Path('runs/flare'),device='cuda',threads=4,al_pool=0,ks=(16,),n_draws=10,benchmark_dir=Path('results'))
split=load_split(T,0,'data',features=target_features(T))
st=ev.SeedState(T,0,split,ctx,sources_fn=lambda:[])
sup=ev.load_or_draw_supports(Path('runs/flare')/f'supports_{T}_s0.json',T,0,split.y_train,ctx.ks,ctx.n_draws)
y=split.y_test
rows=[]
for name in ['flare2-noperm','flare2-m']:
    m=st.model(name)
    for k in (16,):
        for d in range(6):
            idx=sup[k][d]; Xs=split.X_train[idx]; ys=split.y_train[idx]; Xt=split.X_test
            a=roc_auc_score(y,m.predict_proba(Xs,ys,Xt,seed=d))
            b=roc_auc_score(y,1-m.predict_proba(Xs,1-ys,Xt,seed=d))   # class ids swapped
            p=np.random.default_rng(d).permutation(len(ys))
            c=roc_auc_score(y,m.predict_proba(Xs[p],ys[p],Xt,seed=d)) # support row order shuffled
            rows.append((name,k,d,a,b,c))
r=pd.DataFrame(rows,columns=['model','k','draw','auc','auc_swapped_ids','auc_shuffled_rows'])
print(r.groupby(['model','k'])[['auc','auc_swapped_ids','auc_shuffled_rows']].mean().round(3))
print(r[(r.model=='flare2-noperm')&(r.k==16)].round(3).to_string(index=False))
