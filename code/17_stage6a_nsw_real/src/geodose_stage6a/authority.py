from pathlib import Path
import json
from .common import require,sha256_file,read_json
def verify_authorities(root:Path,contract):
    exp=read_json(root/'configs/expected_authorities.json');checks=[]
    for fn,h in exp['input_sha256'].items():
        p=root/'inputs'/fn;require(p.exists(),f'Missing frozen input {fn}');require(sha256_file(p)==h,f'Frozen input hash mismatch {fn}');checks.append('input:'+fn)
    for fn,h in exp['authority_sha256'].items():
        p=root/'authorities'/fn;require(p.exists(),f'Missing authority {fn}');require(sha256_file(p)==h,f'Authority hash mismatch {fn}');checks.append('authority:'+fn)
    for rel,h in exp['vendor_sha256'].items():
        p=root/rel;require(p.exists(),f'Missing vendor source {rel}');require(sha256_file(p)==h,f'Vendor hash mismatch {rel}');checks.append('vendor:'+rel)
    st1=(root/'authorities/STAGE1_CLEANING_DECISION.md').read_text(encoding='utf-8')
    require('Annual longitudinal treatment from the selected public snapshot: NOT CONSTRUCTIBLE.' in st1,'Stage1 treatment infeasibility authority absent')
    require('Use controlled simulation and MineDoseBench—not this snapshot—to establish causal potential-outcome coverage.' in st1,'Stage1 causal boundary absent')
    th=read_json(root/'authorities/STAGE3F_OPERATIONAL_THRESHOLDS_FROZEN.json')
    s=contract['support_thresholds']
    require(float(th['minimum_ess'])==float(s['minimum_ess']),'ESS threshold changed')
    require(float(th['maximum_normalized_weight'])==float(s['maximum_normalized_weight']),'Max-weight threshold changed')
    require(int(th['minimum_graph_safe_count'])==int(s['minimum_graph_safe_count']),'Graph-safe threshold changed')
    cb=contract['real_application_claim_boundary']
    require(cb['authentic_longitudinal_continuous_treatment_available'] is False,'Real treatment gate changed')
    require(cb['mapped_rehabilitation_fraction_is_causal_treatment'] is False,'Snapshot exposure relabeled as treatment')
    require(cb['M6_treatment_route_operational'] is False,'M6 real treatment route improperly enabled')
    require(cb['real_intervals_are_counterfactual_potential_outcome_intervals'] is False,'Real intervals overclaimed')
    return checks
