from __future__ import annotations
from .io import require

def build_main_registry(contract,seed_frame):
    rows=[]
    for a in contract['main_production_allocation']:
        for rep in range(int(a['seed_rep_start']),int(a['seed_rep_end'])+1):
            rows.append({'scope':'main_production','case_id':str(a['case_id']),'scenario_id':str(a['scenario_id']),'seed_replication':rep,'case_local_index':rep-int(a['seed_rep_start'])+1,'f06_level':'','target_dose_override':None,'method_evaluation':True})
    require(len(rows)==int(contract['production_main_rows_expected']),'MAIN_PRODUCTION_ROW_COUNT_NOT_3000')
    prod=seed_frame[seed_frame.phase.astype(str)=='production_provisional'].copy();keys={(str(r.scenario_id),int(r.replication)) for r in prod.itertuples()};used={(r['scenario_id'],r['seed_replication']) for r in rows}
    require(len(keys)==3000 and used==keys,'MAIN_REGISTRY_DOES_NOT_USE_EACH_STAGE3A_PRODUCTION_SEED_EXACTLY_ONCE')
    return rows

def build_f06_registry(contract):
    """Prospectively frozen F06 extensions using the matching main-case seed window.

    This matters for S4_RHO060: its main-production rows are Stage3A S4 seed
    replications 301..400, not 1..100.  Reusing the corresponding main-case
    window avoids silently changing the replication mapping when n changes.
    Small/large still receive distinct derived DGP substreams through the frozen
    F06 common_random_group labels, so this is paired by master replication
    identity rather than an unsupported nested-CRN claim.
    """
    rows=[]
    n=int(contract['f06_freeze']['replications_per_nonprimary_level_per_case'])
    alloc={str(a['case_id']):a for a in contract['main_production_allocation']}
    for cid in contract['f06_freeze']['representative_case_ids']:
        cid=str(cid); require(cid in alloc,f'F06_REPRESENTATIVE_CASE_NOT_IN_MAIN_ALLOCATION:{cid}')
        a=alloc[cid]; start=int(a['seed_rep_start']); end=int(a['seed_rep_end'])
        require(end-start+1>=n,f'F06_MAIN_CASE_HAS_TOO_FEW_SEED_REPS:{cid}')
        for level in ['small','large']:
            for j,rep in enumerate(range(start,start+n),1):
                rows.append({'scope':'f06_extension','case_id':cid,'scenario_id':str(a['scenario_id']),'seed_replication':rep,'case_local_index':j,'f06_level':level,'target_dose_override':None,'method_evaluation':True})
    require(len(rows)==len(contract['f06_freeze']['representative_case_ids'])*2*n,'F06_EXTENSION_ROW_COUNT_DRIFT')
    return rows

def build_extension_registry(contract):
    rows=[]
    for e in contract['extensions']:
        for rep in range(int(e['seed_rep_start']),int(e['seed_rep_end'])+1):
            rows.append({'scope':'registered_stress_extension','extension_id':str(e['extension_id']),'case_id':str(e['case_id']),'scenario_id':None,'seed_replication':rep,'case_local_index':rep-int(e['seed_rep_start'])+1,'f06_level':'','target_dose_override':e.get('target_dose_override'),'method_evaluation':bool(e['method_evaluation']),'expected_structural_refusal':e.get('expected_structural_refusal',''),'theorem_status':str(e['theorem_status'])})
    return rows

def registry_summary(main,f06,ext):
    return {'main_rows':len(main),'f06_extension_rows':len(f06),'stress_extension_rows':len(ext),'method_evaluable_rows':sum(r['method_evaluation'] for r in main+f06+ext),'structural_only_rows':sum(not r['method_evaluation'] for r in ext),'total_case_replication_rows':len(main)+len(f06)+len(ext)}
