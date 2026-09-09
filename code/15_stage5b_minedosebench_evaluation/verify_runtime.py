import sys
from packaging.version import Version
if sys.version_info[:3]!=(3,10,0):raise SystemExit(f'REFUSE: exact Python 3.10.0 required; found {sys.version.split()[0]}')
import numpy,pandas,scipy,geopandas,shapely,pyogrio,pyproj,yaml,sklearn,xgboost,joblib,matplotlib,packaging
expected={'numpy':'2.2.6','pandas':'2.3.3','scipy':'1.15.3','geopandas':'1.1.4','shapely':'2.1.2','pyogrio':'0.13.0','pyproj':'3.7.1','yaml':'6.0.3','sklearn':'1.7.2','xgboost':'3.0.5','joblib':'1.5.2','matplotlib':'3.10.8','packaging':'26.3'}
mods={'numpy':numpy,'pandas':pandas,'scipy':scipy,'geopandas':geopandas,'shapely':shapely,'pyogrio':pyogrio,'pyproj':pyproj,'yaml':yaml,'sklearn':sklearn,'xgboost':xgboost,'joblib':joblib,'matplotlib':matplotlib,'packaging':packaging}
found={}
for k,m in mods.items():
    v=str(m.__version__);found[k]=v
    if Version(v)!=Version(expected[k]):raise SystemExit(f'REFUSE: {k} expected {expected[k]}, found {v}')
print('FROZEN STAGE5B RUNTIME VERIFIED:',{'python':sys.version_info[:3],**found})
