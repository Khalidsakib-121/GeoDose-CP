from __future__ import annotations
import sys
req={'numpy':'2.2.6','pandas':'2.3.3','scipy':'1.15.3','yaml':'6.0.3'}
if sys.version_info[:2]!=(3,10): raise SystemExit(f'Python 3.10 required, found {sys.version}')
import numpy,pandas,scipy,yaml
obs={'numpy':numpy.__version__,'pandas':pandas.__version__,'scipy':scipy.__version__,'yaml':yaml.__version__}
bad={k:(req[k],obs[k]) for k in req if req[k]!=obs[k]}
if bad: raise SystemExit(f'Frozen runtime mismatch: {bad}')
print('FROZEN RUNTIME VERIFIED:',{'python_major_minor':(3,10),**obs})
