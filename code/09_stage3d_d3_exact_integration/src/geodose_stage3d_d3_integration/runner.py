from __future__ import annotations

import hashlib
import inspect
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import yaml

from geodose_stage3d_d3_m4.io import load_stage3b_oracle, load_d2_reference, load_m3_reference

from . import __version__
from .exact_grid import build_s4_common_trace
from .io import (
    IntegrationError, ast_api_inventory, read_zip_csv, read_zip_json, safe_reset_output_dir,
    sha256_file, verify_artifact_hashes, verify_source_tree, write_csv_gz, write_json,
)
from .reductions import reduction_table
from .stage3c_replay import replay_calibration_audit


def _load_json(path: Path) -> dict[str,Any]: return json.loads(path.read_text(encoding="utf-8"))
def _load_yaml(path: Path) -> dict[str,Any]: return yaml.safe_load(path.read_text(encoding="utf-8"))


def _vendor_hash_audit(package_root: Path, stage3c_hashes: dict, m4_hashes: dict) -> dict[str,Any]:
    checks=[]
    # Exact Stage3C bytes used by integration.
    s3map=stage3c_hashes["files"]
    for name in ["__init__.py","methods.py","weights.py","graph.py"]:
        rel=f"src/geodose_stage3c/{name}"; fp=package_root/"vendor"/"geodose_stage3c"/name
        checks.append({"component":"stage3c", "path":rel, "expected":s3map[rel], "observed":sha256_file(fp), "pass":sha256_file(fp)==s3map[rel]})
    # Exact accepted M4/M3 engine bytes used by integration.
    m4map={x["path"]:x["sha256"] for x in m4_hashes["files"]}
    for name in ["__init__.py","graph_law.py","io.py","m3_oracle.py","m4_oracle.py","m6_extractor.py","orbit.py","factorization_fixture.py","runner.py"]:
        rel=f"src/geodose_stage3d_d3_m4/{name}"; fp=package_root/"vendor"/"geodose_stage3d_d3_m4"/name
        checks.append({"component":"m3_m4", "path":rel, "expected":m4map[rel], "observed":sha256_file(fp), "pass":sha256_file(fp)==m4map[rel]})
    if not all(x["pass"] for x in checks): raise IntegrationError("Vendored frozen source bytes changed")
    return {"checks":checks,"all_pass":True}


def _m3_m4_regression(stage3b, m3ref, m4_zip: Path, tol: float) -> dict[str,Any]:
    # M3 accepted output identity is already hash-locked; verify its own status plus accepted M4 regression summary.
    m4ver=read_zip_json(m4_zip,"STAGE3D_D3_M4_VERIFICATION.json")
    m4reg=read_zip_json(m4_zip,"stage3d_d3_m4_m3_regression.json")
    s4=read_zip_csv(m4_zip,"stage3d_d3_m4_s4_structural_summary.csv")
    fac=read_zip_csv(m4_zip,"stage3d_d3_m4_factorizing_reduction.csv")
    if m4ver.get("status")!="verified_complete" or int(m4ver.get("failed_checks",-1))!=0:
        raise IntegrationError("Accepted M4 output not verified_complete")
    if not bool(m4reg.get("pass")):
        raise IntegrationError("Accepted M4 did not preserve accepted M3")
    if not bool(s4["nonfactorization_detected"].all()):
        raise IntegrationError("Accepted M4 structural nonfactorization record changed")
    fac_err=float(np.nanmax(fac[["q3_q4_max_abs","q3_q6_max_abs","p_max_abs"]].to_numpy(float)))
    if fac_err > tol:
        raise IntegrationError("Accepted M4 factorizing reduction no longer matches M3=M4=M6")
    return {
        "m3_status":m3ref.verification.get("status"),"m4_status":m4ver.get("status"),
        "m3_regression_max_abs_pvalue_difference":m4reg.get("max_abs_pvalue_difference"),
        "m3_regression_max_abs_source_marginal_difference":m4reg.get("max_abs_source_marginal_difference"),
        "factorizing_candidate_count":len(fac),"factorizing_max_abs_error":fac_err,
        "s4_structural_candidates":len(s4),"s4_all_nonfactorization_detected":bool(s4["nonfactorization_detected"].all()),
        "s4_max_delta_fact":float(s4["delta_fact_logosc"].max()),
        "s4_max_tv_q4_q6":float(s4["tv_q4_q6"].max()),
    }


def _d2_regression(d2, tol: float) -> dict[str,Any]:
    tr=d2.candidate_trace.copy()
    prob_err=float(np.max(np.abs(tr["probability_sum"].astype(float)-1.0)))
    rule=((tr["conservative_p"].astype(float)>0.1) | tr["singular_conservative_inclusion"].astype(bool))
    mismatch=int(np.sum(rule.to_numpy(bool)!=tr["conservative_accept"].to_numpy(bool)))
    rnd_subset=int(np.sum(tr["randomized_accept"].astype(bool) & ~tr["conservative_accept"].astype(bool)))
    if prob_err>tol or mismatch or rnd_subset:
        raise IntegrationError("Accepted D2 regression failed")
    return {"candidate_rows":len(tr),"fixtures":sorted(tr.fixture_id.astype(str).unique().tolist()),"max_probability_sum_abs_error":prob_err,"conservative_rule_mismatch_count":mismatch,"randomized_not_subset_count":rnd_subset,"registered_domain":d2.contract["candidate_contract"]["candidate_domain"]}


def run(package_root: Path, paths: dict[str,Path], output_dir: Path, overwrite: bool) -> dict[str,Any]:
    t0=time.perf_counter(); cfg=package_root/"configs"
    expected=_load_json(cfg/"expected_upstream_hashes.json"); contract=_load_yaml(cfg/"exact_integration_contract.yaml"); tolerances=_load_yaml(cfg/"tolerances.yaml")
    safe_reset_output_dir(output_dir,package_root,overwrite)

    input_audit=verify_artifact_hashes(paths,expected)
    stage3c_hashes=read_zip_json(paths["stage3c_output_zip"],"stage3c_source_hashes.json")
    d2_hashes=read_zip_json(paths["stage3d_d2_output_zip"],"stage3d_d2_source_hashes.json")
    m4_hashes=read_zip_json(paths["stage3d_d3_m4_output_zip"],"stage3d_d3_m4_source_hashes.json")
    input_audit["stage3c_source_tree"]=verify_source_tree(paths["stage3c_source_root"],stage3c_hashes)
    input_audit["d2_source_tree"]=verify_source_tree(paths["stage3d_d2_source_root"],d2_hashes)
    input_audit["d2_api_inventory_generated"]=True
    write_json(output_dir/"stage3d_d3_d2_api_inventory.json",ast_api_inventory(paths["stage3d_d2_source_root"],d2_hashes))
    input_audit["vendor_frozen_source_audit"]=_vendor_hash_audit(package_root,stage3c_hashes,m4_hashes)
    write_json(output_dir/"stage3d_d3_integration_input_audit.json",input_audit)

    stage3c_ver=read_zip_json(paths["stage3c_output_zip"],"STAGE3C_REFERENCE_VERIFICATION.json")
    if stage3c_ver.get("status")!="verified_complete": raise IntegrationError("Stage3C reference output not verified_complete")
    calibration_audit=read_zip_csv(paths["stage3c_output_zip"],"stage3c_calibration_weight_audit.csv.gz")
    replay,replay_summary=replay_calibration_audit(calibration_audit,float(tolerances["baseline_quantile_abs"]))
    write_csv_gz(replay,output_dir/"stage3d_d3_stage3c_baseline_replay.csv.gz")
    write_json(output_dir/"stage3d_d3_stage3c_baseline_replay_summary.json",replay_summary)

    stage3b=load_stage3b_oracle(paths["stage3b_output_zip"]); d2=load_d2_reference(paths["stage3d_d2_output_zip"]); m3ref=load_m3_reference(paths["stage3d_d3_m3_output_zip"])
    m3m4=_m3_m4_regression(stage3b,m3ref,paths["stage3d_d3_m4_output_zip"],float(tolerances["pvalue_abs"]))
    d2reg=_d2_regression(d2,float(tolerances["probability_sum_abs"]))
    write_json(output_dir/"stage3d_d3_m3_m4_regression.json",m3m4); write_json(output_dir/"stage3d_d3_m6_d2_regression.json",d2reg)

    common,summary,meta=build_s4_common_trace(stage3b,d2.candidate_trace,float(contract["alpha"]),float(tolerances["probability_sum_abs"]),int(tolerances["minimum_s4_candidate_count"]))
    write_csv_gz(common,output_dir/"stage3d_d3_s4_common_candidate_trace.csv.gz"); summary.to_csv(output_dir/"stage3d_d3_s4_method_summary.csv",index=False,lineterminator="\n"); write_json(output_dir/"stage3d_d3_s4_common_grid_metadata.json",meta)

    reductions=reduction_table(m3ref.source_marginals,d2.contract,float(tolerances["reduction_abs"])); reductions.to_csv(output_dir/"stage3d_d3_reduction_status.csv",index=False,lineterminator="\n")
    if (reductions[reductions.reduction_id.isin(["R1","R2","R4"])].status!="PASS").any(): raise IntegrationError("In-scope reduction check failed")
    if reductions.loc[reductions.reduction_id=="R3","status"].iloc[0]!="DEFERRED_STAGE3E": raise IntegrationError("R3 must remain deferred to Stage3E")

    method_registry=pd.DataFrame([
        ["M1","standard_split_conformal",True,"frozen_stage3c","interval"],
        ["M2","dose_only_weighted_conformal",True,"frozen_stage3c","interval"],
        ["M3","spatial_only_graph_residual_conformal",True,"accepted_d3_m3","candidate_pvalue"],
        ["M4","naive_dose_times_spatial_marginal",True,"accepted_d3_m4","candidate_pvalue"],
        ["M5","graph_safe_weighted_conformal",True,"frozen_stage3c","interval"],
        ["M6","GeoDose_CP_G1_G2_G3_exact",True,"accepted_d2_exact_reference","candidate_pvalue"],
        ["H1","geographic_distance_heuristic",False,"frozen_stage3c_ablation","not_principal"],
        ["H4","M2_times_geographic_heuristic",False,"frozen_stage3c_ablation","not_principal"],
    ],columns=["method","definition","principal","source","representation"])
    method_registry.to_csv(output_dir/"stage3d_d3_method_registry.csv",index=False,lineterminator="\n")
    readiness=_load_yaml(cfg/"nuisance_readiness.yaml")
    ready_rows=[]
    for m,vals in readiness["methods"].items():
        for v,status in vals.items(): ready_rows.append({"method":m,"nuisance_variant":v,"status":status})
    pd.DataFrame(ready_rows).to_csv(output_dir/"stage3d_d3_nuisance_readiness.csv",index=False,lineterminator="\n")

    claim={
        "exact_common_interface_integrated":True,"principal_methods":["M1","M2","M3","M4","M5","M6"],
        "H1_H4_remain_ablations":True,"M6_adapter_mode":"accepted_D2_output_backed_exact_reference",
        "M6_new_data_pilot_engine_ready":False,"estimated_M6_exact_coverage_claimed":False,
        "R3_N2_reduction_completed":False,"R3_owner":"Stage3E","pilot_run":False,"production_run":False,
        "publication_evidence":False,"D2_full_real_line_claim":False,"D2_registered_domain":[-8.0,8.0],
        "next_gate":"source_level_M6_new_data_adapter_plus_estimated_nuisance_variants_before_20_rep_pilot"
    }
    write_json(output_dir/"stage3d_d3_claim_boundary.json",claim)
    pd.DataFrame(columns=["case_id","method","candidate_y","refusal_code","detail"]).to_csv(output_dir/"stage3d_d3_refusal_log.csv",index=False,lineterminator="\n")
    env={"python":platform.python_version(),"implementation":platform.python_implementation(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"pyyaml":yaml.__version__,"platform":platform.platform(),"script_version":__version__}
    write_json(output_dir/"stage3d_d3_environment_inventory.json",env)
    runtime={"seconds":time.perf_counter()-t0,"script_version":__version__,"candidate_count":int(meta["candidate_count"]),"common_trace_rows":len(common)}; write_json(output_dir/"stage3d_d3_runtime.json",runtime)
    return {"input_audit":input_audit,"replay_summary":replay_summary,"m3m4":m3m4,"d2reg":d2reg,"meta":meta,"reductions":reductions.to_dict(orient="records"),"claim":claim,"runtime":runtime}
