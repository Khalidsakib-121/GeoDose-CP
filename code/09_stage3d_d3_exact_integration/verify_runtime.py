import sys
import numpy, pandas, scipy, yaml
expected={'python_major_minor':(3,10),'numpy':'2.2.6','pandas':'2.3.3','scipy':'1.15.3','pyyaml':'6.0.3'}
obs={'python_major_minor':sys.version_info[:2],'numpy':numpy.__version__,'pandas':pandas.__version__,'scipy':scipy.__version__,'pyyaml':yaml.__version__}
errors=[]
if tuple(obs['python_major_minor'])!=expected['python_major_minor']: errors.append(f"Python {obs['python_major_minor']} != (3,10)")
for k in ['numpy','pandas','scipy','pyyaml']:
    if obs[k]!=expected[k]: errors.append(f"{k} {obs[k]} != {expected[k]}")
if errors: raise SystemExit('FROZEN RUNTIME MISMATCH: '+'; '.join(errors))
print('FROZEN RUNTIME VERIFIED:',obs)
