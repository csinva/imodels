"""Build docs/img/gpgam.gif, the animation at the top of the GPGam post: for three California housing features, the
points are cut into the model's quantile bins, each bin becomes its average, and the exact GP curve with its band
is drawn through those averages; the closing card adds the curves up. Fit on 1500 rows so the bands are visible.
Run from anywhere: uv run python docs/pages/gpgam_hero_gif.py"""
import numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import fetch_california_housing
from imodels import GPGamRegressor
import pathlib; HERE=pathlib.Path(__file__).resolve().parent
d=fetch_california_housing(); names=list(d.feature_names)
rng=np.random.RandomState(3); sub=rng.choice(len(d.target),1500,replace=False)
X,y=d.data[sub],d.target[sub]
m=GPGamRegressor(n_pairs=0).fit(X,y)
assert not m.log_target_
yn=(y-m.y_mean_)/m.y_std_
def contrib(j):                       # per-row contribution of feature j on the standardized scale
    u=m.units_.index(j); f=m.main_values_[m.main_offsets_[u]:m.main_offsets_[u+1]]
    return f[np.searchsorted(m.edges_[j],X[:,j],side='right')], f
total=sum(contrib(j)[0] for j in m.units_)+m.bias_
FEATS=['MedInc','HouseAge','Latitude']
BLUE, ORANGE, GREY = '#005588', '#a8541f', '#9a9a9a'
def prep(name):
    j=names.index(name); x=X[:,j]; c,f=contrib(j)
    r=yn-(total-c)                    # partial residual: what is left for this feature to explain
    idx=np.searchsorted(m.edges_[j],x,side='right'); cnt=np.bincount(idx,minlength=len(f))
    bm=np.bincount(idx,weights=r,minlength=len(f))/np.maximum(cnt,1)
    g,mu,sd=m.shape_function(j,return_std=True); mu=mu/m.y_std_; sd=sd/m.y_std_
    lo,hi=np.percentile(x,[0.5,99.5]); keep=(g>=lo)&(g<=hi)
    ok=(x>=lo)&(x<=hi)
    kb=keep&(cnt>0)
    return dict(name=name,xs=x[ok],ys=r[ok],edges=m.edges_[j],centers=g[kb],bmeans=bm[kb],g=g[keep],mu=mu[keep],sd=sd[keep],lo=lo,hi=hi)
P=[prep(n) for n in FEATS]
print({p['name']:(len(p['edges'])+1, float(p['sd'].mean()), float(np.ptp(p['mu']))) for p in P})
FPF=46
fig,ax=plt.subplots(figsize=(8,3.6),dpi=110); plt.subplots_adjust(left=.075,right=.98,top=.86,bottom=.16)
def draw(frame):
    ax.clear(); k=min(frame//FPF,len(P)); t=frame%FPF
    if k==len(P):                     # closing card: the additive sum
        ax.set_axis_off(); ax.set_xlim(-.1,5.6); ax.set_ylim(-1.3,1.3)  # shared vertical scale across features
        for i,p in enumerate(P):
            xx=np.linspace(0,1,len(p['g'])); off=i*1.3; s=0.9/max(np.ptp(q['mu']) for q in P)
            mu=(p['mu']-p['mu'].mean())*s; sd=p['sd']*s
            ax.fill_between(xx+off,mu-2*sd,mu+2*sd,color=BLUE,alpha=.18,linewidth=0)
            ax.plot(xx+off,mu,color=BLUE,lw=2.2)
            ax.text(off+.5,-0.95,f"f({p['name']})",ha='center',fontsize=11,color=BLUE)
            ax.text(off+1.15,0,'+',ha='center',va='center',fontsize=20,color=GREY)
        ax.text(3.95,0,'…  =  ŷ',ha='left',va='center',fontsize=17,color=ORANGE,fontweight='bold')
        ax.text(3.95,-0.45,'plus a few pairwise terms',ha='left',va='center',fontsize=9.5,color=GREY)
        ax.set_title('GPGam: one exact Gaussian-process curve per feature, smoothness chosen by marginal likelihood',fontsize=10.5,color='#333',loc='left')
        return
    p=P[k]; ax.set_xlim(p['lo'],p['hi']); ax.set_ylim(-2.6,2.6)
    ax.set_xlabel(p['name'],fontsize=11); ax.set_ylabel('partial residual (standardized)',fontsize=9.5)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    ax.scatter(p['xs'],p['ys'],s=7,color=GREY,alpha=.35 if t>=8 else .8,linewidths=0)
    phase=['1. the data','2. quantile bins, and the average inside each','2. quantile bins, and the average inside each','3. a GP curve and its uncertainty band, fit to those averages'][min(t//11,3)]
    ax.set_title(phase,fontsize=12,color=ORANGE if t>=22 else '#333',loc='left')
    if t>=8:
        a=min(1,(t-8)/6)
        for e in p['edges']: ax.axvline(e,color=BLUE,alpha=.12*a,lw=.8)
    if t>=14:
        n_show=int(min(1,(t-14)/8)*len(p['centers']))
        ax.scatter(p['centers'][:n_show],p['bmeans'][:n_show],s=24,color=ORANGE,zorder=3,edgecolor='white',linewidths=.6)
    if t>=22:
        q=int(min(1,(t-22)/14)*len(p['g']))
        if q>1:
            ax.fill_between(p['g'][:q],p['mu'][:q]-2*p['sd'][:q],p['mu'][:q]+2*p['sd'][:q],color=BLUE,alpha=.22,linewidth=0)
            ax.plot(p['g'][:q],p['mu'][:q],color=BLUE,lw=2.4,zorder=4)
    if t>=36: ax.text(.99,.04,'the marginal likelihood picks how smooth, exactly, in closed form',transform=ax.transAxes,ha='right',fontsize=9,color=BLUE)
frames=FPF*len(P)+34
anim=FuncAnimation(fig,draw,frames=frames,interval=90)
out=str(HERE.parent/'img'/'gpgam.gif')
anim.save(out,writer=PillowWriter(fps=11))
print("frames",frames,"size MB",round(os.path.getsize(out)/1e6,2))
