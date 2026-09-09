from __future__ import annotations
import hashlib,json
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from .common import require

class RealOutcomeModel:
    def __init__(self,kind,features,seed,contract):
        self.kind=str(kind);self.features=list(features);self.seed=int(seed);self.contract=contract;self.model=None;self.fit_id=None;self.training_hash=None
        self.hp=dict(contract['rf_hyperparameters'] if self.kind=='rf' else contract['xgb_hyperparameters'])
    def _X(self,df):
        X=df[self.features].to_numpy(float)
        require(np.isfinite(X).all(),'Nonfinite predictors')
        return X
    def fit(self,df,outcome):
        X=self._X(df);y=df[outcome].to_numpy(float);require(np.isfinite(y).all(),'Nonfinite training outcome')
        if self.kind=='rf':
            self.model=RandomForestRegressor(random_state=self.seed,**self.hp)
        elif self.kind=='xgb':
            self.model=XGBRegressor(random_state=self.seed,objective='reg:squarederror',verbosity=0,**self.hp)
        else: raise ValueError(self.kind)
        self.model.fit(X,y)
        ids='|'.join(df.block_id.astype(str));self.training_hash=hashlib.sha256(ids.encode()).hexdigest()
        self.fit_id=hashlib.sha256((self.kind+'|'+str(self.seed)+'|'+self.training_hash+'|'+json.dumps(self.hp,sort_keys=True)).encode()).hexdigest()
        return self
    def predict(self,df):
        X=self._X(df);p=np.asarray(self.model.predict(X),float);require(np.isfinite(p).all(),'Nonfinite predictions');return p
    def feature_importance(self):
        v=np.asarray(getattr(self.model,'feature_importances_',np.full(len(self.features),np.nan)),float)
        return pd.DataFrame({'feature':self.features,'importance':v})
