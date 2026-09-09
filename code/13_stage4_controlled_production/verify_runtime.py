import platform,sys
req={'numpy':'2.2.6','pandas':'2.3.3','scipy':'1.15.3','sklearn':'1.7.2','xgboost':'3.0.5','joblib':'1.5.2','yaml':'6.0.3'}
if sys.version_info[:2]!=(3,10):raise SystemExit(f'Python 3.10 required, got {platform.python_version()}')
mods={}
for n,v in req.items():
 m=__import__(n);got=getattr(m,'__version__',None);mods[n]=got
 if got!=v:raise SystemExit(f'{n}=={v} required, got {got}')
print('FROZEN RUNTIME VERIFIED:',{'python_major_minor':(3,10),**mods})
