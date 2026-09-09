import sys,importlib
req={'numpy':'2.2.6','pandas':'2.3.3','scipy':'1.15.3','geopandas':'1.1.4','shapely':'2.1.2','pyogrio':'0.13.0','pyproj':'3.7.1','yaml':'6.0.3','sklearn':'1.7.2','xgboost':'3.0.5','joblib':'1.5.2','matplotlib':'3.10.8','packaging':'26.3'}
if sys.version_info[:3]!=(3,10,0):raise SystemExit(f'Exact Python 3.10.0 required; got {sys.version}')
got={'python':sys.version_info[:3]}
for n,v in req.items():
    m=importlib.import_module(n);g=getattr(m,'__version__',None)
    if g!=v:raise SystemExit(f'Runtime version mismatch {n}: {g} != {v}')
    got[n]=g
print('FROZEN STAGE6A RUNTIME VERIFIED:',got)
