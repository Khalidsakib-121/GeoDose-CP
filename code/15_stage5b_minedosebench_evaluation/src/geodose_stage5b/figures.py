from __future__ import annotations
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=['M2','M3','M4','M5','M6']
COLORS={'M2':'#0072B2','M3':'#009E73','M4':'#D55E00','M5':'#CC79A7','M6':'#000000'}
MARKERS={'M2':'o','M3':'s','M4':'^','M5':'D','M6':'P'}

def _save(fig,out,name):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);fig.savefig(out/f'{name}.png',dpi=320,bbox_inches='tight',metadata={'Software':'GeoDose-CP Stage5B v1.0.0'});fig.savefig(out/f'{name}.pdf',bbox_inches='tight',metadata={'Creator':'GeoDose-CP Stage5B v1.0.0','Producer':'Matplotlib 3.10.8','CreationDate':None,'ModDate':None});plt.close(fig)

def hero(hero,out):
    if hero.empty:return
    fig,axs=plt.subplots(1,2,figsize=(10.6,4.15),sharex=True)
    for m in METHODS:
        g=hero[hero.method==m].sort_values('spatial_rho');
        if g.empty:continue
        axs[0].plot(g.spatial_rho,g.selective_coverage,marker=MARKERS[m],label=m,color=COLORS[m],linewidth=1.8,markersize=5)
        if {'coverage_ci_low','coverage_ci_high'}.issubset(g.columns):
            lo=np.maximum(0,g.selective_coverage.to_numpy(float)-g.coverage_ci_low.to_numpy(float));hi=np.maximum(0,g.coverage_ci_high.to_numpy(float)-g.selective_coverage.to_numpy(float));axs[0].errorbar(g.spatial_rho,g.selective_coverage,yerr=np.vstack([lo,hi]),fmt='none',ecolor=COLORS[m],elinewidth=.9,capsize=2,alpha=.65)
        if g.mean_width.notna().any():
            axs[1].plot(g.spatial_rho,g.mean_width,marker=MARKERS[m],label=m,color=COLORS[m],linewidth=1.8,markersize=5)
            if {'width_ci_low','width_ci_high'}.issubset(g.columns):
                lo=np.maximum(0,g.mean_width.to_numpy(float)-g.width_ci_low.to_numpy(float));hi=np.maximum(0,g.width_ci_high.to_numpy(float)-g.mean_width.to_numpy(float));axs[1].errorbar(g.spatial_rho,g.mean_width,yerr=np.vstack([lo,hi]),fmt='none',ecolor=COLORS[m],elinewidth=.9,capsize=2,alpha=.65)
    axs[0].axhline(.90,color='0.45',linestyle='--',linewidth=1.2);axs[0].set_ylabel('Selective coverage');axs[0].set_ylim(0,1.02);axs[0].set_title('(a) Validity under combined shift + dependence')
    axs[1].set_ylabel('Mean interval / hull width');axs[1].set_title('(b) Finite-domain efficiency subset')
    for ax in axs:ax.set_xlabel(r'Spatial dependence $\rho$');ax.grid(alpha=.18,linewidth=.6);ax.tick_params(direction='out')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='lower center',bbox_to_anchor=(.5,-.035),frameon=False);fig.subplots_adjust(bottom=.20,wspace=.28);_save(fig,out,'FIG01_S4_HERO_COVERAGE_WIDTH')

def refusal_heatmap(query,out):
    g=query[(query.track=='RF_ET_FG_PRIMARY')&(query.truth_scale=='observed')].groupby(['case_id','method']).operational_return.mean().unstack('method').reindex(columns=METHODS);r=1-g
    fig,ax=plt.subplots(figsize=(7.2,8.3));im=ax.imshow(r.to_numpy(float),aspect='auto',vmin=0,vmax=1,cmap='Greys');ax.set_xticks(range(len(METHODS)),METHODS);ax.set_yticks(range(len(r)),[x.replace('MDB_','') for x in r.index],fontsize=7.5);ax.set_xlabel('Method');ax.set_ylabel('Registered MineDoseBench case');ax.set_title('Operational refusal rate (RF primary track)');c=fig.colorbar(im,ax=ax,pad=.02);c.set_label('Refusal rate');fig.tight_layout();_save(fig,out,'FIG02_REFUSAL_HEATMAP')

def s9(s9df,out):
    if s9df.empty:return
    # Collapse over case variants for a compact method x truth-scale view, with case points lightly shown.
    agg=s9df.groupby(['method','truth_scale']).selective_coverage.mean().reset_index();fig,ax=plt.subplots(figsize=(7.5,4.2));x=np.arange(len(METHODS));w=.34
    for j,scale in enumerate(['observed','latent']):
        vals=[float(agg[(agg.method==m)&(agg.truth_scale==scale)].selective_coverage.iloc[0]) if len(agg[(agg.method==m)&(agg.truth_scale==scale)]) else np.nan for m in METHODS];ax.bar(x+(j-.5)*w,vals,width=w,label='Observed product' if scale=='observed' else 'Latent ecological truth',alpha=.88)
    ax.axhline(.9,color='0.4',linestyle='--',linewidth=1.2);ax.set_xticks(x,METHODS);ax.set_ylim(0,1.02);ax.set_ylabel('Mean selective coverage across S9 variants');ax.set_xlabel('Method');ax.set_title('EO measurement-error sensitivity: observed vs latent target');ax.legend(frameon=False,ncol=2);ax.grid(axis='y',alpha=.18);fig.tight_layout();_save(fig,out,'FIG03_S9_MEASUREMENT_ERROR')

def s10(s10df,out):
    if s10df.empty:return
    agg=s10df.groupby(['case_id','method']).selective_coverage.mean().reset_index();fig,ax=plt.subplots(figsize=(7.5,4.2));x=np.arange(len(METHODS));w=.34
    for j,(case,label) in enumerate([('MDB_S10_90M','90 m'),('MDB_S10_180M','180 m')]):
        vals=[float(agg[(agg.case_id==case)&(agg.method==m)].selective_coverage.iloc[0]) if len(agg[(agg.case_id==case)&(agg.method==m)]) else np.nan for m in METHODS];ax.bar(x+(j-.5)*w,vals,width=w,label=label,alpha=.88)
    ax.axhline(.9,color='0.4',linestyle='--',linewidth=1.2);ax.set_xticks(x,METHODS);ax.set_ylim(0,1.02);ax.set_ylabel('Selective coverage');ax.set_xlabel('Method');ax.set_title('Prespecified MAUP sensitivity');ax.legend(frameon=False);ax.grid(axis='y',alpha=.18);fig.tight_layout();_save(fig,out,'FIG04_S10_MAUP')

def diagnostic(diag,out):
    if diag.empty:return
    g=diag[(diag.track=='RF_ET_FG_PRIMARY')&diag.empirical_undercoverage.notna()&diag.mean_sparse_tv_diagnostic.notna()]
    if g.empty:return
    fig,ax=plt.subplots(figsize=(5.4,4.6));ax.scatter(g.mean_sparse_tv_diagnostic,g.empirical_undercoverage,s=34,alpha=.75);mx=max(float(g.mean_sparse_tv_diagnostic.max()),float(g.empirical_undercoverage.max()),.02);ax.plot([0,mx],[0,mx],linestyle='--',linewidth=1.1,color='0.4');ax.set_xlim(0,mx*1.04);ax.set_ylim(0,mx*1.04);ax.set_xlabel('Mean sparse-law TV diagnostic');ax.set_ylabel('Empirical undercoverage max(0, 0.90−coverage)');ax.set_title('Coverage-loss diagnostic calibration (M6)');ax.grid(alpha=.18);fig.tight_layout();_save(fig,out,'FIG05_DIAGNOSTIC_VS_UNDERCOVERAGE')

def exact_audit(exact,out):
    if exact.empty:return
    g=exact[exact.reference=='sparse_m64'].groupby('method').agg(pvalue_abs_diff=('pvalue_absolute_difference','mean'),pinsker=('full_vs_sparse_pinsker_tv','mean')).reindex(METHODS).dropna(how='all');fig,ax=plt.subplots(figsize=(5.7,4.2));x=np.arange(len(g));ax.bar(x,g.pvalue_abs_diff.to_numpy(float));ax.set_xticks(x,g.index);ax.set_ylabel('Mean |p_sparse − p_full|');ax.set_xlabel('Method');ax.set_title('Exact full-graph vs sparse-m64 audit');ax.grid(axis='y',alpha=.18);fig.tight_layout();_save(fig,out,'FIG06_EXACT_SPARSE_AUDIT')

def build_all(summary,query,exact,out):
    hero(summary['hero'],out);refusal_heatmap(query,out);s9(summary['s9'],out);s10(summary['s10'],out);diagnostic(summary['diagnostic'],out);exact_audit(exact,out)
