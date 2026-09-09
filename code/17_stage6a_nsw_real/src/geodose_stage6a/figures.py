from __future__ import annotations
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROLE_COLORS={'nuisance_training':'#1f77b4','support_audit':'#2ca02c','calibration':'#ff7f0e','test_target':'#d62728','buffer_excluded':'#bdbdbd','secondary_support_stratum':'#9467bd','baseline_ineligible':'#eeeeee'}
METHOD_COLORS={'M1':'#4c78a8','M3':'#f58518','M5':'#54a24b'}
MINE_SHORT={'Mt Arthur Coal':'Mt Arthur','Hunter Valley Operations':'HVO','Bulga Complex':'Bulga'}

def _save(fig,out,name):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f'{name}.png',dpi=320,bbox_inches='tight',facecolor='white')
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)
def _clean(ax):
    ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=9)

def fig01_spatial_design(s90,out):
    mines=['Mt Arthur Coal','Hunter Valley Operations','Bulga Complex'];fig,axs=plt.subplots(1,3,figsize=(12.2,4.2),constrained_layout=True)
    role_order=['nuisance_training','support_audit','calibration','test_target','buffer_excluded','secondary_support_stratum','baseline_ineligible']
    for ax,mn in zip(axs,mines):
        g=s90[s90.MineN==mn]
        for role in role_order:
            z=g[g.benchmark_role==role]
            if len(z):ax.scatter(z.centroid_x/1000.0,z.centroid_y/1000.0,s=2.3,c=ROLE_COLORS[role],alpha=.78,linewidths=0,rasterized=True)
        ax.set_title(MINE_SHORT[mn],fontsize=11,fontweight='bold');ax.set_aspect('equal');ax.set_xlabel('Easting (km)',fontsize=9)
        if ax is axs[0]:ax.set_ylabel('Northing (km)',fontsize=9)
        _clean(ax)
    handles=[Line2D([0],[0],marker='o',linestyle='',markersize=6,color=ROLE_COLORS[r],label=r.replace('_',' ').title()) for r in role_order]
    fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False,fontsize=8)
    fig.suptitle('Frozen buffered spatial design for the real NSW demonstration (90 m)',fontsize=13,fontweight='bold')
    _save(fig,out,'FIG01_NSW_SPATIAL_DESIGN')

def fig02_quality_exposure(s90,out):
    g=s90[s90.real_baseline_eligible.astype(bool)].copy();fig,axs=plt.subplots(1,2,figsize=(10.8,4.2),constrained_layout=True)
    mines=['Mt Arthur Coal','Hunter Valley Operations','Bulga Complex']
    vals=[g.loc[g.MineN==m,'mapped_rehabilitation_fraction'].to_numpy(float) for m in mines]
    axs[0].boxplot(vals,tick_labels=[MINE_SHORT[m] for m in mines],showfliers=False)
    axs[0].set_ylabel('Mapped rehabilitation fraction');axs[0].set_title('Descriptive 2026 snapshot exposure\n(not a treatment or predictor)',fontweight='bold',fontsize=10);axs[0].set_ylim(-.02,1.02);_clean(axs[0])
    ret=[g.loc[g.MineN==m,'strict_quality_ue20'].mean() for m in mines]
    axs[1].bar([MINE_SHORT[m] for m in mines],ret,color='#4c78a8');axs[1].set_ylim(0,1.02);axs[1].set_ylabel('Retained fraction');axs[1].set_title('Strict UE20 product-quality sensitivity',fontweight='bold',fontsize=10)
    for i,v in enumerate(ret):axs[1].text(i,v+.015,f'{100*v:.1f}%',ha='center',fontsize=9)
    _clean(axs[1]);fig.suptitle('EO quality and quarantined snapshot context',fontsize=13,fontweight='bold');_save(fig,out,'FIG02_EO_QUALITY_AND_SNAPSHOT_EXPOSURE')

def fig03_coverage_width(pooled,out):
    d=pooled[(pooled.scale=='90m')&(pooled.track=='RF_PRIMARY')&pooled.method.isin(['M1','M3','M5'])].set_index('method').reindex(['M1','M3','M5'])
    fig,axs=plt.subplots(1,2,figsize=(10.8,4.15),constrained_layout=True)
    x=np.arange(3);cols=[METHOD_COLORS[m] for m in ['M1','M3','M5']]
    cov=d.selective_observed_product_coverage.to_numpy(float);axs[0].bar(x,cov,color=cols);axs[0].axhline(.90,color='black',ls='--',lw=1.2,label='Nominal 0.90')
    axs[0].set_xticks(x,['M1','M3','M5']);axs[0].set_ylim(max(0,min(cov)-.08),1.01);axs[0].set_ylabel('Held-out observed-product coverage');axs[0].legend(frameon=False,fontsize=8);axs[0].set_title('Predictive coverage',fontweight='bold');_clean(axs[0])
    wid=d.mean_interval_width.to_numpy(float);axs[1].bar(x,wid,color=cols);axs[1].set_xticks(x,['M1','M3','M5']);axs[1].set_ylabel('Mean interval width');axs[1].set_title('Interval width\n(M3: frozen pre-outcome inversion subset)',fontweight='bold',fontsize=10);_clean(axs[1])
    fig.suptitle('Real NSW held-out satellite-product uncertainty (90 m, RF)',fontsize=13,fontweight='bold');_save(fig,out,'FIG03_OBSERVED_PRODUCT_COVERAGE_WIDTH')

def fig04_support_map(s90,q,out):
    d=q[(q.scale=='90m')&(q.track=='RF_PRIMARY')&(q.method=='M3')][['block_id','returned','pvalue','graph_safe_count']].copy()
    g=s90[s90.benchmark_role=='test_target'][['block_id','MineN','centroid_x','centroid_y']].merge(d,on='block_id',how='left',validate='1:1')
    mines=['Mt Arthur Coal','Hunter Valley Operations','Bulga Complex'];fig,axs=plt.subplots(1,3,figsize=(12.2,4.1),constrained_layout=True)
    sc=None
    for ax,mn in zip(axs,mines):
        z=g[g.MineN==mn];ok=z.returned.astype(bool)
        sc=ax.scatter(z.loc[ok,'centroid_x']/1000.0,z.loc[ok,'centroid_y']/1000.0,c=z.loc[ok,'pvalue'],s=6,cmap='viridis',vmin=0,vmax=1,linewidths=0,rasterized=True)
        if (~ok).any():ax.scatter(z.loc[~ok,'centroid_x']/1000.0,z.loc[~ok,'centroid_y']/1000.0,marker='x',s=14,c='#d62728',lw=.8,label='Refused')
        ax.set_title(MINE_SHORT[mn],fontweight='bold',fontsize=10);ax.set_aspect('equal');ax.set_xlabel('Easting (km)',fontsize=8)
        if ax is axs[0]:ax.set_ylabel('Northing (km)',fontsize=8)
        _clean(ax)
    if sc is not None:fig.colorbar(sc,ax=axs,shrink=.78,label='M3 spatial conformal p-value')
    fig.suptitle('Support-aware spatial warning map on frozen 90 m test targets',fontsize=13,fontweight='bold');_save(fig,out,'FIG04_SUPPORT_REFUSAL_AND_INTERVAL_MAP')

def fig05_maup(maup,out):
    d=maup[(maup.track=='RF_PRIMARY')&maup.method.isin(['M1','M3','M5'])].copy()
    fig,axs=plt.subplots(1,2,figsize=(10.8,4.15),constrained_layout=True);methods=['M1','M3','M5'];x=np.arange(2);width=.23
    for j,m in enumerate(methods):
        z=d[d.method==m].set_index('scale').reindex(['90m','180m']);axs[0].bar(x+(j-1)*width,z.selective_observed_product_coverage,width,label=m,color=METHOD_COLORS[m]);axs[1].bar(x+(j-1)*width,z.return_rate,width,label=m,color=METHOD_COLORS[m])
    axs[0].axhline(.9,color='black',ls='--',lw=1);axs[0].set_ylabel('Held-out observed-product coverage');axs[0].set_ylim(.75,1.01);axs[0].set_title('Coverage sensitivity',fontweight='bold')
    axs[1].set_ylabel('Return rate');axs[1].set_ylim(0,1.02);axs[1].set_title('Support/refusal sensitivity',fontweight='bold')
    for ax in axs:ax.set_xticks(x,['90 m','180 m']);_clean(ax)
    axs[0].legend(frameon=False,ncol=3,fontsize=8);fig.suptitle('MAUP sensitivity under anchored 90 m → 180 m aggregation',fontsize=13,fontweight='bold');_save(fig,out,'FIG05_MAUP_90M_180M')

def fig06_leakage(leak,out):
    d=leak.copy();labels=[]
    for r in d.itertuples():
        if r.split=='random_block':labels.append('Random')
        elif r.split=='buffered_spatial_block':labels.append('Buffered')
        else:labels.append('LOMO\n'+MINE_SHORT.get(r.held_out_mine,r.held_out_mine))
    x=np.arange(len(d));fig,axs=plt.subplots(1,2,figsize=(11.2,4.2),constrained_layout=True)
    axs[0].bar(x,d.coverage,color='#4c78a8');axs[0].axhline(.9,color='black',ls='--',lw=1);axs[0].set_ylabel('Held-out observed-product coverage');axs[0].set_ylim(0,1.02);axs[0].set_title('Coverage',fontweight='bold')
    axs[1].bar(x,d.rmse,color='#f58518');axs[1].set_ylabel('PV-change RMSE');axs[1].set_title('Predictive error',fontweight='bold')
    for ax in axs:ax.set_xticks(x,labels,rotation=0);_clean(ax)
    fig.suptitle('Compact spatial-leakage diagnostic (RF; predictive, not causal)',fontsize=13,fontweight='bold');_save(fig,out,'FIG06_SPATIAL_LEAKAGE_DIAGNOSTIC')

def fig07_quality(qs,out):
    d=qs[(qs.scale=='90m')&(qs.track=='RF_PRIMARY')&qs.method.isin(['M1','M3','M5'])];methods=['M1','M3','M5'];frames=['all_frozen_eligible','strict_ue20'];x=np.arange(3);width=.34
    fig,axs=plt.subplots(1,2,figsize=(10.8,4.15),constrained_layout=True)
    for j,f in enumerate(frames):
        z=d[d.quality_frame==f].set_index('method').reindex(methods);lab='Frozen base quality' if j==0 else 'Strict UE20'
        axs[0].bar(x+(j-.5)*width,z.selective_observed_product_coverage,width,label=lab);axs[1].bar(x+(j-.5)*width,z.return_rate,width,label=lab)
    axs[0].axhline(.9,color='black',ls='--',lw=1);axs[0].set_ylim(.75,1.01);axs[0].set_ylabel('Held-out observed-product coverage');axs[0].set_title('Coverage',fontweight='bold')
    axs[1].set_ylim(0,1.02);axs[1].set_ylabel('Return rate');axs[1].set_title('Support/refusal',fontweight='bold')
    for ax in axs:ax.set_xticks(x,methods);_clean(ax)
    axs[0].legend(frameon=False,fontsize=8);fig.suptitle('Product-quality sensitivity without post-hoc threshold tuning',fontsize=13,fontweight='bold');_save(fig,out,'FIG07_PRODUCT_QUALITY_SENSITIVITY')

def make_all(s90,q,pooled,maup,leak,qs,out):
    fig01_spatial_design(s90,out);fig02_quality_exposure(s90,out);fig03_coverage_width(pooled,out);fig04_support_map(s90,q,out);fig05_maup(maup,out);fig06_leakage(leak,out);fig07_quality(qs,out)
