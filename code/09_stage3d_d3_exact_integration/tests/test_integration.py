from pathlib import Path
import hashlib
import sys
import tempfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
for p in [ROOT/'vendor',ROOT/'src']:
    sys.path.insert(0,str(p))

from geodose_stage3c.methods import build_interval
from geodose_stage3d_d3_m4.m4_oracle import evaluate_m4
from geodose_stage3d_d3_m4.orbit import numeric_key
from geodose_stage3d_d3_integration.d2_adapter import AcceptedD2ExactAdapter
from geodose_stage3d_d3_integration.io import (
    IntegrationError, ast_api_inventory, safe_reset_output_dir, verify_source_tree,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    passed=0

    # 1. Signed-zero duplicate-key hardening remains active.
    assert numeric_key(-0.0)==numeric_key(+0.0)
    passed+=1

    # 2. M4 remains exactly one normalized dose x spatial product in the
    # constant-spatial reduction.
    s=np.ones(6)/6
    d=np.array([.25,.5,1.5,2,4,3.])
    e=np.array([-1,-.5,-.2,.3,.7,.4])
    r=evaluate_m4(s,e,np.log(d),5)
    assert np.max(np.abs(r.q4-d/d.sum()))<1e-14
    assert r.normalization_count==1
    passed+=1

    # 3. Frozen Stage3C interval code is callable without changing identity.
    scores=np.array([.1,.2,.4,.7,1.1])
    a=build_interval(0,scores,np.zeros(5),0,.1)
    b=build_interval(0,scores,np.zeros(5),0,.1)
    assert a.interval_status==b.interval_status
    assert (np.isinf(a.quantile) and np.isinf(b.quantile)) or a.quantile==b.quantile
    passed+=1

    # 4. The D2 exact-reference adapter accepts only archived candidate hexes
    # and fails closed for new candidates.
    df=pd.DataFrame({
        'fixture_id':['F'],'candidate_y':[0.0],'candidate_hex':[float(0).hex()],
        'conservative_p':[1.0],'conservative_accept':[True],
        'singular_conservative_inclusion':[False],'distinct_states':[6],
        'probability_sum':[1.0]
    })
    ad=AcceptedD2ExactAdapter(df,'F',.1)
    assert ad.evaluate_registered(0.0,float(0).hex()).included
    try:
        ad.evaluate_registered(1.0,float(1).hex())
        raise AssertionError('must refuse unregistered D2 candidate')
    except IntegrationError:
        pass
    passed+=1

    # 5. Frozen-source verifier accepts exact bytes and rejects a changed byte.
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        fp=root/'src'/'pkg'/'a.py'; fp.parent.mkdir(parents=True)
        fp.write_text('def f(x):\n    return x + 1\n',encoding='utf-8')
        rec={'files':{'src/pkg/a.py':_sha(fp)}}
        out=verify_source_tree(root,rec)
        assert out['pass'] and out['matched_file_count']==1
        fp.write_text('def f(x):\n    return x + 2\n',encoding='utf-8')
        try:
            verify_source_tree(root,rec)
            raise AssertionError('changed frozen source must fail')
        except IntegrationError:
            pass
    passed+=1

    # 6. D2 API inventory is AST-only and captures callable signatures/classes.
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        fp=root/'src'/'pkg'/'api.py'; fp.parent.mkdir(parents=True)
        fp.write_text('def f(a, b):\n    return a+b\n\nclass C:\n    def m(self, x):\n        return x\n',encoding='utf-8')
        rec={'files':{'src/pkg/api.py':_sha(fp)}}
        inv=ast_api_inventory(root,rec)
        mod=inv['modules']['src/pkg/api.py']
        assert mod['functions'][0]['name']=='f'
        assert mod['functions'][0]['args']==['a','b']
        assert mod['classes'][0]['name']=='C' and 'm' in mod['classes'][0]['methods']
        assert inv['source_only_metadata'] is True
    passed+=1

    # 7. Output reset is path-guarded and cannot target package source roots.
    try:
        safe_reset_output_dir(ROOT/'src',ROOT,True)
        raise AssertionError('source directory reset must be forbidden')
    except IntegrationError:
        pass
    passed+=1


    # 8. Windows configuration supports both normal double-nested Explorer
    # extraction and single-nested M3/M4 output layouts.
    import json
    cfg=json.loads((ROOT/'configs'/'local_paths_windows.json').read_text(encoding='utf-8'))
    for key, project in [
        ('stage3d_d3_m3_output_zip','GeoDose_Stage3D_D3_M3_Oracle_v1_0'),
        ('stage3d_d3_m4_output_zip','GeoDose_Stage3D_D3_M4_Oracle_Structural_Gate_v1_0'),
    ]:
        vals=cfg[key]
        assert len(vals) >= 2
        assert any(f'\\{project}\\{project}\\' in x for x in vals)
        assert any(x.count(project)==1 for x in vals)
    passed+=1

    assert passed==8
    print('STAGE3D D3 EXACT INTEGRATION UNIT TESTS PASSED (8/8)')


if __name__=='__main__':
    run()
