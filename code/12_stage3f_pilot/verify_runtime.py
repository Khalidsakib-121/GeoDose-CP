import sys
req={'numpy':'2.2.6','pandas':'2.3.3','scipy':'1.15.3','yaml':'6.0.3','sklearn':'1.7.2','joblib':'1.5.2','xgboost':'3.0.5'}
if sys.version_info[:2]!=(3,10): raise SystemExit(f'Python 3.10 required, got {sys.version}')
mods={}
for name,v in req.items():
 m=__import__(name); got=getattr(m,'__version__',None); mods[name]=got
 if got!=v: raise SystemExit(f'{name} expected {v}, got {got}')
print('FROZEN RUNTIME VERIFIED:',{'python_major_minor':sys.version_info[:2],**mods})
