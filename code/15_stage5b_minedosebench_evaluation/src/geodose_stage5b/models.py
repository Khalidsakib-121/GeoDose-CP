from __future__ import annotations
import hashlib,json,math
import numpy as np,pandas as pd
from scipy.special import betaln,expit,logit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.preprocessing import StandardScaler
from .common import require,Stage5BError
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor=None

def _hash_frame(df,cols):
    q=df[['block_id']+cols].copy().sort_values('block_id',kind='mergesort'); raw=q.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode(); return hashlib.sha256(raw).hexdigest()

def robust_scale(x):
    x=np.asarray(x,float); med=float(np.median(x)); mad=float(np.median(np.abs(x-med))); s=1.4826*mad
    if not np.isfinite(s) or s<1e-6: s=float(np.std(x,ddof=1)) if len(x)>1 else 1.0
    return float(np.clip(s,1e-6,.5))

class OutcomeModel:
    def __init__(self,kind,features,seed,contract):
        self.kind=str(kind);self.features=list(features);self.seed=int(seed);self.contract=contract;self.fit_id='';self.training_hash=''
        if kind=='rf':
            hp=dict(contract['rf_hyperparameters']); hp['random_state']=self.seed; self.hp=hp; self.model=RandomForestRegressor(**hp)
        elif kind=='xgb':
            require(XGBRegressor is not None,'xgboost not installed'); hp=dict(contract['xgb_hyperparameters']);hp['random_state']=self.seed;hp['objective']='reg:squarederror';hp['verbosity']=0;self.hp=hp;self.model=XGBRegressor(**hp)
        else: raise Stage5BError(f'Unknown outcome model {kind}')
        self.scale_=np.nan
    def design(self,df,a=None):
        aa=df.A.to_numpy(float) if a is None else np.asarray(a,float); return np.column_stack([aa,df[self.features].to_numpy(float)])
    def fit(self,train):
        require(len(train)>=20,'Insufficient nuisance-training rows'); self.training_hash=_hash_frame(train,['A']+self.features+['Y_observed_at_A']); self.model.fit(self.design(train),train.Y_observed_at_A.to_numpy(float)); resid=train.Y_observed_at_A.to_numpy(float)-self.model.predict(self.design(train));self.scale_=robust_scale(resid)
        payload={'kind':self.kind,'features':self.features,'hp':self.hp,'training_hash':self.training_hash,'scale':self.scale_};self.fit_id=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24];return self
    def predict(self,df,a=None): return np.asarray(self.model.predict(self.design(df,a)),float)

class MixedPropensity:
    """Estimated mixed measure: multinomial atom/interior mass + beta interior density.
    This is an operational nuisance fit; Stage5B does not claim a finite-sample nuisance theorem for it.
    """
    def __init__(self,features,seed):
        self.features=list(features);self.seed=int(seed);self.scaler=StandardScaler();self.cat=None;self.const_cat=None;self.mean_model=None;self.kappa=8.;self.fit_id='';self.training_hash=''
    def fit(self,train):
        self.training_hash=_hash_frame(train,['A']+self.features+['treatment_category'])
        X=self.scaler.fit_transform(train[self.features].to_numpy(float)); rawlab=train.treatment_category
        if pd.api.types.is_numeric_dtype(rawlab): labels=pd.to_numeric(rawlab,errors='raise').to_numpy(int)
        else: labels=rawlab.astype(str).map({'atom_0':0,'atom_1':1,'interior':2,'0':0,'1':1,'2':2}).to_numpy(float); require(np.isfinite(labels).all(),'Unknown treatment_category label'); labels=labels.astype(int)
        require(set(np.unique(labels)).issubset({0,1,2}),'Invalid treatment_category code'); classes=np.unique(labels)
        if len(classes)==1:
            self.const_cat=np.zeros(3);self.const_cat[classes[0]]=1.
        else:
            self.cat=LogisticRegression(C=10.0,solver='lbfgs',max_iter=1200,random_state=self.seed).fit(X,labels)
        interior=train[(train.A>0)&(train.A<1)].copy()
        if len(interior)>=20:
            Xi=self.scaler.transform(interior[self.features].to_numpy(float)); a=np.clip(interior.A.to_numpy(float),1e-6,1-1e-6); self.mean_model=Ridge(alpha=.10).fit(Xi,logit(a)); mu=np.clip(expit(self.mean_model.predict(Xi)),.01,.99); var=float(np.mean((a-mu)**2)); base=float(np.mean(mu*(1-mu))); self.kappa=float(np.clip(base/max(var,1e-5)-1,2,100))
        else: self.kappa=6.
        payload={'features':self.features,'training_hash':self.training_hash,'kappa':self.kappa,'classes':classes.tolist()};self.fit_id=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24];return self
    def category_probs(self,df):
        n=len(df)
        if self.const_cat is not None: return np.repeat(self.const_cat[None,:],n,axis=0)
        X=self.scaler.transform(df[self.features].to_numpy(float)); raw=self.cat.predict_proba(X); out=np.full((n,3),1e-12,float)
        for j,c in enumerate(self.cat.classes_.astype(int)):out[:,c]=raw[:,j]
        out=np.clip(out,1e-12,1);return out/out.sum(axis=1,keepdims=True)
    def beta_params(self,df):
        if self.mean_model is None: mu=np.full(len(df),.5)
        else: mu=np.clip(expit(self.mean_model.predict(self.scaler.transform(df[self.features].to_numpy(float)))),.01,.99)
        return np.maximum(mu*self.kappa,1e-4),np.maximum((1-mu)*self.kappa,1e-4)
    def density(self,df,a=None):
        aa=df.A.to_numpy(float) if a is None else np.asarray(a,float); probs=self.category_probs(df); alpha,beta=self.beta_params(df); out=np.zeros(len(df),float); z0=aa==0.;z1=aa==1.;inter=(aa>0)&(aa<1);out[z0]=probs[z0,0];out[z1]=probs[z1,1]
        if inter.any():
            x=np.clip(aa[inter],1e-12,1-1e-12); logpdf=(alpha[inter]-1)*np.log(x)+(beta[inter]-1)*np.log1p(-x)-betaln(alpha[inter],beta[inter]);out[inter]=probs[inter,2]*np.exp(np.clip(logpdf,-745,700))
        return out

def oracle_density(df,a=None):
    aa=df.A.to_numpy(float) if a is None else np.asarray(a,float); p0=df.g_pi0.to_numpy(float);p1=df.g_pi1.to_numpy(float);pi=df.g_pii.to_numpy(float);al=df.g_alpha.to_numpy(float);be=df.g_beta.to_numpy(float);out=np.zeros(len(df));z0=aa==0.;z1=aa==1.;inter=(aa>0)&(aa<1);out[z0]=p0[z0];out[z1]=p1[z1]
    if inter.any():
        x=np.clip(aa[inter],1e-12,1-1e-12); lp=(al[inter]-1)*np.log(x)+(be[inter]-1)*np.log1p(-x)-betaln(al[inter],be[inter]);out[inter]=pi[inter]*np.exp(np.clip(lp,-745,700))
    return out

def oracle_mean(case,df,a):
    # outcome_baseline_truth and x_pv_truth are generated oracle fields.
    from minedosebench.math import conditional_mean
    return conditional_mean(case,df.outcome_baseline_truth.to_numpy(float),df.x_pv_truth.to_numpy(float),np.asarray(a,float))
