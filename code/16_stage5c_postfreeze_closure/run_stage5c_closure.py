from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "INPUTS"
CONFIG = ROOT / "config" / "closure_contract.json"
OUT = ROOT / "STAGE5C_CLOSURE_OUTPUTS"
WORK = ROOT / "_STAGE5C_WORK"
STATE = WORK / "checkpoint.json"

FILES = {
    "stage5b": INPUTS / "STAGE5B_PRIMARY_RESULTS.zip",
    "exact": INPUTS / "EXACT_SPARSE_AUDIT_VERIFIED_v2.zip",
    "g": INPUTS / "ESTIMATED_G_SENSITIVITY_RESULTS.zip",
    "external": INPUTS / "EXTERNAL_BASELINE_RESULTS.zip",
}

STAGES = [
    "01_VERIFY_INPUTS",
    "02_REGISTERED_PRIMARY",
    "03_EXACT_SPARSE",
    "04_ESTIMATED_G",
    "05_EXTERNAL_BASELINES",
    "06_MANUSCRIPT_EXPORTS",
    "07_CLOSURE_REPORT",
    "08_FINAL_PACKAGE",
]

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"version": "1.0.0", "completed": {}, "created_utc": now_iso()}

def save_state(state):
    WORK.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE)

def mark_done(state, stage, outputs=None):
    state["completed"][stage] = {
        "utc": now_iso(),
        "outputs": outputs or {},
    }
    save_state(state)

def stage_done(state, stage):
    rec = state.get("completed", {}).get(stage)
    if not rec:
        return False
    for rel, expected in rec.get("outputs", {}).items():
        p = ROOT / rel
        if not p.exists() or sha256_file(p) != expected:
            return False
    return True

def clean_outputs_if_new():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

def read_csv(zf: zipfile.ZipFile, name: str, **kwargs):
    return pd.read_csv(io.BytesIO(zf.read(name)), **kwargs)

def read_json(zf: zipfile.ZipFile, name: str):
    return json.loads(zf.read(name).decode("utf-8"))

def verify_zip_crc(path: Path):
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"CRC failure in {path.name}: {bad}")

def verify_manifest(zf: zipfile.ZipFile, manifest_name: str, base_prefix: str = ""):
    obj = read_json(zf, manifest_name)
    failures = []
    for row in obj.get("files", []):
        rel = row["file"]
        member = f"{base_prefix}{rel}"
        if member not in zf.namelist():
            failures.append(f"missing:{member}")
            continue
        b = zf.read(member)
        if len(b) != int(row["bytes"]):
            failures.append(f"size:{member}")
        if sha256_bytes(b) != row["sha256"]:
            failures.append(f"sha256:{member}")
    if failures:
        raise RuntimeError(f"Manifest verification failed for {manifest_name}: {failures[:10]}")
    return {"files_verified": len(obj.get("files", [])), "declared_aggregate_sha256": obj.get("aggregate_sha256")}

def output_hashes(paths):
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}

def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.12g")

def assert_close(a, b, tol=1e-9, label="value"):
    if not np.isfinite(a) or not np.isfinite(b) or abs(float(a)-float(b)) > tol:
        raise AssertionError(f"{label}: observed={a}, expected={b}, tol={tol}")

def main(force=False):
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    state = load_state()
    clean_outputs_if_new()

    verification = {
        "stage": contract["stage"],
        "version": contract["version"],
        "started_utc": now_iso(),
        "status": "RUNNING",
        "checks": {},
        "evidence_classification": contract["evidence_blocks"],
    }

    # ------------------------------------------------------
    # 01 INPUT INTEGRITY
    # ------------------------------------------------------
    if force or not stage_done(state, "01_VERIFY_INPUTS"):
        observed_hashes = {}
        for key, path in FILES.items():
            if not path.exists():
                raise FileNotFoundError(path)
            verify_zip_crc(path)
            observed_hashes[path.name] = sha256_file(path)

        for name, exp in contract["expected_input_sha256"].items():
            obs = observed_hashes[name]
            if obs != exp:
                raise RuntimeError(f"Input SHA mismatch for {name}: {obs} != {exp}")

        # Verify internal manifests and accepted nested Stage5B production release.
        with zipfile.ZipFile(FILES["stage5b"]) as z:
            m5 = verify_manifest(
                z,
                "STAGE5B_PRODUCTION_OUTPUTS/OUTPUT_FILE_MANIFEST.json",
                "STAGE5B_PRODUCTION_OUTPUTS/",
            )
            nested = z.read("STAGE5B_PRODUCTION_OUTPUTS/GeoDose_Stage5B_MineDoseBench_PRODUCTION_RESULTS.zip")
            nested_hash = sha256_bytes(nested)
            if nested_hash != contract["accepted_nested_stage5b_release_sha256"]:
                raise RuntimeError("Accepted nested Stage5B release SHA mismatch")
            v5 = read_json(z, "STAGE5B_PRODUCTION_OUTPUTS/STAGE5B_VERIFICATION.json")
            if v5.get("checks_failed") != 0 or v5.get("query_rows") != 187500:
                raise RuntimeError("Stage5B verification gate failed")

        with zipfile.ZipFile(FILES["g"]) as z:
            mg = verify_manifest(z, "OUTPUT_FILE_MANIFEST.json", "")
            vg = read_json(z, "ESTIMATED_G_VERIFICATION.json")
            if vg.get("status") != "PASS" or vg.get("query_rows") != 10500 or vg.get("computational_failures") != 0:
                raise RuntimeError("Estimated-g verification gate failed")

        with zipfile.ZipFile(FILES["external"]) as z:
            me = verify_manifest(
                z,
                "EXTERNAL_BASELINE_OUTPUTS/OUTPUT_FILE_MANIFEST.json",
                "EXTERNAL_BASELINE_OUTPUTS/",
            )
            ve = read_json(z, "EXTERNAL_BASELINE_OUTPUTS/EXTERNAL_BASELINE_VERIFICATION.json")
            if ve.get("status") != "PASS" or ve.get("combined_query_rows") != 14000 or ve.get("computational_failures") != 0:
                raise RuntimeError("External-baseline verification gate failed")

        verification["checks"]["input_integrity"] = {
            "status": "PASS",
            "outer_input_sha256": observed_hashes,
            "stage5b_manifest": m5,
            "estimated_g_manifest": mg,
            "external_manifest": me,
            "accepted_nested_stage5b_release_sha256": nested_hash,
        }
        p = OUT / "INPUT_INTEGRITY.json"
        p.write_text(json.dumps(verification["checks"]["input_integrity"], indent=2), encoding="utf-8")
        mark_done(state, "01_VERIFY_INPUTS", output_hashes([p]))
    else:
        verification["checks"]["input_integrity"] = json.loads((OUT/"INPUT_INTEGRITY.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------
    # 02 REGISTERED PRIMARY M2-M6
    # ------------------------------------------------------
    primary_path = OUT / "tables" / "TABLE_PRIMARY_M2_M6.csv"
    if force or not stage_done(state, "02_REGISTERED_PRIMARY"):
        with zipfile.ZipFile(FILES["stage5b"]) as z:
            cov = read_csv(z, "STAGE5B_PRODUCTION_OUTPUTS/SUMMARY_COVERAGE.csv")
            loc = read_csv(z, "STAGE5B_PRODUCTION_OUTPUTS/SUMMARY_FIFTH_PERCENTILE_COVERAGE.csv")
            report = read_json(z, "STAGE5B_PRODUCTION_OUTPUTS/STAGE5B_PRODUCTION_REPORT.json")
            claim = read_json(z, "STAGE5B_PRODUCTION_OUTPUTS/STAGE5B_CLAIM_BOUNDARY.json")

        pf = cov[
            (cov["track"]=="RF_ET_FG_PRIMARY") &
            (cov["truth_scale"]=="observed") &
            (cov["method"].isin(["M2","M3","M4","M5","M6"]))
        ].copy()
        pl = loc[
            (loc["track"]=="RF_ET_FG_PRIMARY") &
            (loc["truth_scale"]=="observed") &
            (loc["method"].isin(["M2","M3","M4","M5","M6"]))
        ].copy()
        if len(pf) != 135 or pf["case_id"].nunique() != 27 or len(pl) != 135:
            raise RuntimeError("Registered primary dimensions do not equal 27 cases x 5 methods")

        rows=[]
        for m in ["M2","M3","M4","M5","M6"]:
            a=pf[pf.method==m]
            b=pl[pl.method==m]
            rows.append({
                "method":m,
                "mean_selective_coverage":a.selective_coverage.mean(),
                "minimum_selective_coverage":a.selective_coverage.min(),
                "cases_at_or_above_0_90":int((a.selective_coverage>=0.90).sum()),
                "n_registered_cases":27,
                "mean_local_q0_05":b.fifth_percentile_local_coverage.mean(),
                "minimum_local_q0_05":b.fifth_percentile_local_coverage.min(),
                "mean_return_rate":(a.n_returned/a.n_queries).mean(),
                "evidence_class":"REGISTERED_PRIMARY",
                "scope":"RF-primary observed-scale MineDoseBench across 27 registered configurations",
            })
        primary = pd.DataFrame(rows)
        write_csv(primary, primary_path)

        expected = {
            "M2": (0.8854576564868963,0.8022222222222221,8,0.8264928404489095,0.7078787878787878,0.95),
            "M3": (0.9855555555555555,0.836,26,0.9724847222464943,0.7464390243902439,1.0),
            "M4": (0.9804622048949534,0.942222222222222,27,0.9573896938220355,0.8803603603603603,0.95),
            "M5": (0.8988662401953549,0.836,12,0.8385030996385359,0.7694117647058824,0.8945185185185184),
            "M6": (0.9692280701754385,0.914,27,0.9488295293434471,0.8950619469026548,0.95),
        }
        for _,r in primary.iterrows():
            e=expected[r.method]
            assert_close(r.mean_selective_coverage,e[0],1e-12,f"{r.method} mean coverage")
            assert_close(r.minimum_selective_coverage,e[1],1e-12,f"{r.method} min coverage")
            if int(r.cases_at_or_above_0_90)!=e[2]: raise AssertionError("case count mismatch")
            assert_close(r.mean_local_q0_05,e[3],1e-12,f"{r.method} mean q05")
            assert_close(r.minimum_local_q0_05,e[4],1e-12,f"{r.method} min q05")
            assert_close(r.mean_return_rate,e[5],1e-12,f"{r.method} return")

        if report.get("thresholds_retuned") or report.get("hyperparameters_retuned_from_production") or report.get("stage5a_modified"):
            raise RuntimeError("Stage5B freeze discipline violated")
        if claim.get("width_superiority_without_matched_coverage") is not False:
            raise RuntimeError("Stage5B width claim boundary mismatch")

        verification["checks"]["registered_primary"] = {
            "status":"PASS","rows":len(primary),"registered_cases":27,
            "stage5a_modified":report.get("stage5a_modified"),
            "thresholds_retuned":report.get("thresholds_retuned"),
            "hyperparameters_retuned":report.get("hyperparameters_retuned_from_production"),
        }
        mark_done(state, "02_REGISTERED_PRIMARY", output_hashes([primary_path]))
    else:
        primary = pd.read_csv(primary_path)
        verification["checks"]["registered_primary"]={"status":"PASS_REUSED","rows":len(primary)}

    # ------------------------------------------------------
    # 03 EXACT-SPARSE DEEP AUDIT
    # ------------------------------------------------------
    exact_path = OUT / "tables" / "TABLE_EXACT_SPARSE_AUDIT.csv"
    exact_s4_path = OUT / "tables" / "TABLE_EXACT_SPARSE_S4_RHO.csv"
    if force or not stage_done(state, "03_EXACT_SPARSE"):
        with zipfile.ZipFile(FILES["exact"]) as z:
            summ = read_csv(z,"EXACT_SPARSE_M6_SUMMARY_VERIFIED.csv")
            q = read_csv(z,"EXACT_SPARSE_M6_QUERY_LEVEL_VERIFIED.csv")
            fig = z.read("EXACT_SPARSE_M6_ECDF_VERIFIED.png")
        overall = summ[summ.record_type=="OVERALL_M6"].copy()
        if len(overall)!=1 or len(q)!=2700:
            raise RuntimeError("Exact-sparse audit dimensions invalid")
        r=overall.iloc[0]
        checks={
            "mean_abs_delta_p":0.005566498125545724,
            "median_abs_delta_p":2.220446049250313e-16,
            "q90_abs_delta_p":0.016068109414595556,
            "q95_abs_delta_p":0.03466683991151527,
            "q99_abs_delta_p":0.09738635179564382,
            "max_abs_delta_p":0.42944562999658564,
            "fraction_abs_delta_gt_0_01":352/2700,
            "fraction_abs_delta_gt_0_05":76/2700,
            "fraction_abs_delta_gt_0_10":23/2700,
            "decision_disagreement_rate_alpha_0_10":9/2700,
            "exact_coverage":0.9885185185185185,
            "sparse_coverage":0.9874074074074074,
        }
        for k,v in checks.items(): assert_close(float(r[k]),v,1e-12,k)

        # Independent query-level recomputation.
        absd=(q.p_exact-q.p_sparse).abs()
        assert_close(absd.mean(),checks["mean_abs_delta_p"],1e-12,"query recomputed mean |dp|")
        if int((q.decision_disagreement_alpha010==True).sum())!=9:
            raise AssertionError("Decision disagreement count != 9")

        out = pd.DataFrame([{
            "n_target_identities":2700,
            "mean_abs_delta_p":float(r.mean_abs_delta_p),
            "median_abs_delta_p":float(r.median_abs_delta_p),
            "q90_abs_delta_p":float(r.q90_abs_delta_p),
            "q95_abs_delta_p":float(r.q95_abs_delta_p),
            "q99_abs_delta_p":float(r.q99_abs_delta_p),
            "max_abs_delta_p":float(r.max_abs_delta_p),
            "n_abs_delta_gt_0_01":int(r.n_abs_delta_gt_0_01),
            "n_abs_delta_gt_0_05":int(r.n_abs_delta_gt_0_05),
            "n_abs_delta_gt_0_10":int(r.n_abs_delta_gt_0_10),
            "decision_disagreement_count_alpha_0_10":int(r.decision_disagreement_count_alpha_0_10),
            "decision_disagreement_rate_alpha_0_10":float(r.decision_disagreement_rate_alpha_0_10),
            "exact_coverage":float(r.exact_coverage),
            "sparse_coverage":float(r.sparse_coverage),
            "target_specific_ess_matches":int(r.target_specific_diagnostic_matches),
            "evidence_class":"POSTFREEZE_EXACT_SPARSE_AUDIT",
            "claim_boundary":"Strong aggregate agreement; no uniform query-level equivalence claim.",
        }])
        write_csv(out,exact_path)

        s4=summ[summ.record_type=="S4_RHO"][[
            "rho","n_pairs","mean_abs_delta_p","q90_abs_delta_p","q95_abs_delta_p",
            "q99_abs_delta_p","max_abs_delta_p","decision_disagreement_count_alpha_0_10",
            "exact_coverage","sparse_coverage"
        ]].sort_values("rho")
        write_csv(s4,exact_s4_path)
        (OUT/"figures"/"FIG_EXACT_SPARSE_ECDF.png").write_bytes(fig)

        verification["checks"]["exact_sparse"]={
            "status":"PASS","pairs":2700,"decision_disagreements":9,
            "target_specific_diagnostic_matches":int(r.target_specific_diagnostic_matches),
        }
        mark_done(state,"03_EXACT_SPARSE",output_hashes([
            exact_path,exact_s4_path,OUT/"figures"/"FIG_EXACT_SPARSE_ECDF.png"
        ]))
    else:
        verification["checks"]["exact_sparse"]={"status":"PASS_REUSED"}

    # ------------------------------------------------------
    # 04 ESTIMATED-g SENSITIVITY
    # ------------------------------------------------------
    g_overall_path = OUT/"tables"/"TABLE_G_SENSITIVITY.csv"
    g_cases_path = OUT/"tables"/"TABLE_G_SENSITIVITY_CASES.csv"
    g_vs_path = OUT/"tables"/"TABLE_G_SENSITIVITY_VS_ORACLE.csv"
    if force or not stage_done(state,"04_ESTIMATED_G"):
        with zipfile.ZipFile(FILES["g"]) as z:
            overall=read_csv(z,"SUMMARY_ESTIMATED_G_OVERALL.csv")
            cases=read_csv(z,"SUMMARY_ESTIMATED_G_CASES.csv")
            vs=read_csv(z,"SUMMARY_ESTIMATED_G_VS_ORACLE.csv")
            ver=read_json(z,"ESTIMATED_G_VERIFICATION.json")
            rep=read_json(z,"ESTIMATED_G_PRODUCTION_REPORT.json")
            fig1=z.read("figures/FIG01_G_SENSITIVITY_COVERAGE_LOCAL_REFUSAL.png")
            fig2=z.read("figures/FIG02_G_SENSITIVITY_WIDTH_SUBSET.png")
        if set(overall.g_track)!={"G_ORACLE","G_ESTIMATED","G_MISSPECIFIED"} or len(cases)!=21:
            raise RuntimeError("Estimated-g track/case dimensions invalid")
        if ver.get("query_rows")!=10500 or ver.get("inversion_rows")!=210 or ver.get("computational_failures")!=0:
            raise RuntimeError("Estimated-g verification counts invalid")
        if rep.get("stage5a_modified") or rep.get("thresholds_retuned") or rep.get("outcome_model_retuned") or rep.get("spatial_model_retuned") or rep.get("registered_stage5b_primary_modified"):
            raise RuntimeError("Estimated-g freeze/isolation discipline violated")

        # Verify summary is exactly mean/min of frozen seven case summaries.
        for _,r in overall.iterrows():
            g=cases[cases.g_track==r.g_track]
            assert_close(g.selective_coverage.mean(),r.mean_case_selective_coverage,1e-12,f"{r.g_track} mean coverage")
            assert_close(g.selective_coverage.min(),r.minimum_case_selective_coverage,1e-12,f"{r.g_track} min coverage")
            assert_close(g.local_q0_05.mean(),r.mean_case_local_q0_05,1e-12,f"{r.g_track} mean q05")
            assert_close(g.local_q0_05.min(),r.minimum_case_local_q0_05,1e-12,f"{r.g_track} min q05")
            assert_close(g.return_rate.mean(),r.mean_case_return_rate,1e-12,f"{r.g_track} return")

        gout=overall.copy()
        gout["evidence_class"]="POSTFREEZE_ESTIMATED_G_SENSITIVITY"
        gout["claim_boundary"]="Sensitivity analysis only; registered 27-case Stage5B remains primary evidence."
        write_csv(gout,g_overall_path)
        write_csv(cases,g_cases_path)
        write_csv(vs,g_vs_path)
        (OUT/"figures"/"FIG_G_SENSITIVITY_COVERAGE_LOCAL_REFUSAL.png").write_bytes(fig1)
        (OUT/"figures"/"FIG_G_SENSITIVITY_WIDTH_SUBSET.png").write_bytes(fig2)

        verification["checks"]["estimated_g"]={
            "status":"PASS","query_rows":10500,"inversion_rows":210,"cases":7,
            "tracks":["G_ORACLE","G_ESTIMATED","G_MISSPECIFIED"],
            "primary_modified":False,"retuned":False,
        }
        mark_done(state,"04_ESTIMATED_G",output_hashes([
            g_overall_path,g_cases_path,g_vs_path,
            OUT/"figures"/"FIG_G_SENSITIVITY_COVERAGE_LOCAL_REFUSAL.png",
            OUT/"figures"/"FIG_G_SENSITIVITY_WIDTH_SUBSET.png",
        ]))
    else:
        verification["checks"]["estimated_g"]={"status":"PASS_REUSED"}

    # ------------------------------------------------------
    # 05 EXTERNAL BASELINES
    # ------------------------------------------------------
    ext_overall_path=OUT/"tables"/"TABLE_EXTERNAL_BASELINES.csv"
    ext_cases_path=OUT/"tables"/"TABLE_EXTERNAL_BASELINE_CASES.csv"
    ext_cap_path=OUT/"tables"/"TABLE_EXTERNAL_CAPABILITY.csv"
    ext_width_path=OUT/"tables"/"TABLE_EXTERNAL_COVERAGE_MATCHED_WIDTH.csv"
    ext_vs_path=OUT/"tables"/"TABLE_EXTERNAL_VS_M6.csv"
    if force or not stage_done(state,"05_EXTERNAL_BASELINES"):
        with zipfile.ZipFile(FILES["external"]) as z:
            prefix="EXTERNAL_BASELINE_OUTPUTS/"
            overall=read_csv(z,prefix+"SUMMARY_EXTERNAL_BASELINE_OVERALL.csv")
            cases=read_csv(z,prefix+"SUMMARY_EXTERNAL_BASELINE_CASES.csv")
            cap=read_csv(z,prefix+"EXTERNAL_BASELINE_CAPABILITY_TABLE.csv")
            width=read_csv(z,prefix+"SUMMARY_EXTERNAL_COVERAGE_MATCHED_WIDTH.csv")
            vs=read_csv(z,prefix+"SUMMARY_EXTERNAL_VS_M6.csv")
            ver=read_json(z,prefix+"EXTERNAL_BASELINE_VERIFICATION.json")
            rep=read_json(z,prefix+"EXTERNAL_BASELINE_PRODUCTION_REPORT.json")
            fig1=z.read(prefix+"figures/FIG01_EXTERNAL_BASELINE_COVERAGE_LOCAL_RETURN.png")
            fig2=z.read(prefix+"figures/FIG02_EXTERNAL_BASELINE_WIDTH_SUBSET.png")
        expected_methods={"E1_DRWCP_LOCAL","E2_WCP_JOINT_SHIFT","E3_SLSCP_SPLIT","M6"}
        if set(overall.method)!=expected_methods:
            raise RuntimeError("External method set invalid")
        if ver.get("external_query_rows")!=10500 or ver.get("M6_reference_rows")!=3500 or ver.get("combined_query_rows")!=14000 or ver.get("computational_failures")!=0:
            raise RuntimeError("External verification counts invalid")
        if rep.get("stage5a_modified") or rep.get("thresholds_retuned") or rep.get("outcome_model_retuned") or rep.get("registered_stage5b_primary_modified"):
            raise RuntimeError("External-baseline freeze discipline violated")

        # M6 in external subset must equal the G_ESTIMATED sensitivity M6 reference summary.
        with zipfile.ZipFile(FILES["g"]) as zg:
            gsum=read_csv(zg,"SUMMARY_ESTIMATED_G_OVERALL.csv")
        gm=gsum[gsum.g_track=="G_ESTIMATED"].iloc[0]
        em=overall[overall.method=="M6"].iloc[0]
        assert_close(em.mean_case_selective_coverage,gm.mean_case_selective_coverage,1e-12,"external M6 vs g-estimated mean coverage")
        assert_close(em.minimum_case_selective_coverage,gm.minimum_case_selective_coverage,1e-12,"external M6 vs g-estimated min coverage")
        assert_close(em.mean_case_local_q0_05,gm.mean_case_local_q0_05,1e-12,"external M6 vs g-estimated mean q05")
        assert_close(em.minimum_case_local_q0_05,gm.minimum_case_local_q0_05,1e-12,"external M6 vs g-estimated min q05")
        assert_close(em.mean_case_native_return_rate,gm.mean_case_return_rate,1e-12,"external M6 vs g-estimated return")
        assert_close(em.mean_width_on_frozen_70_identity_subset,gm.mean_width_on_frozen_subset,1e-12,"external M6 vs g-estimated width")

        eout=overall.copy()
        eout["evidence_class"]="POSTFREEZE_EXTERNAL_BASELINES"
        eout["claim_boundary"]="External families are adapted published comparator families; M6 validity/local-robustness advantage does not imply universal efficiency dominance."
        write_csv(eout,ext_overall_path)
        write_csv(cases,ext_cases_path)
        write_csv(cap,ext_cap_path)
        write_csv(width,ext_width_path)
        write_csv(vs,ext_vs_path)
        (OUT/"figures"/"FIG_EXTERNAL_BASELINE_COVERAGE_LOCAL_RETURN.png").write_bytes(fig1)
        (OUT/"figures"/"FIG_EXTERNAL_BASELINE_WIDTH_SUBSET.png").write_bytes(fig2)

        verification["checks"]["external_baselines"]={
            "status":"PASS","external_query_rows":10500,"M6_reference_rows":3500,
            "combined_query_rows":14000,"methods":sorted(expected_methods),
            "M6_reference_aligned_to_estimated_g":True,
        }
        mark_done(state,"05_EXTERNAL_BASELINES",output_hashes([
            ext_overall_path,ext_cases_path,ext_cap_path,ext_width_path,ext_vs_path,
            OUT/"figures"/"FIG_EXTERNAL_BASELINE_COVERAGE_LOCAL_RETURN.png",
            OUT/"figures"/"FIG_EXTERNAL_BASELINE_WIDTH_SUBSET.png",
        ]))
    else:
        verification["checks"]["external_baselines"]={"status":"PASS_REUSED"}

    # ------------------------------------------------------
    # 06 EVIDENCE INDEX + CLAIM BOUNDARIES + compact summary figure
    # ------------------------------------------------------
    evidence_path=OUT/"FINAL_EVIDENCE_INDEX.csv"
    claims_path=OUT/"FINAL_CLAIM_BOUNDARIES.csv"
    if force or not stage_done(state,"06_MANUSCRIPT_EXPORTS"):
        evidence=pd.DataFrame([
            {
                "evidence_id":"EVIDENCE_01_STAGE5B_PRIMARY",
                "class":"REGISTERED_PRIMARY",
                "source":"Stage5B MineDoseBench production",
                "scope":"27 registered configurations; RF-primary observed scale; M2-M6",
                "manuscript_role":"Primary known-truth benchmark",
                "replaces_primary":False,
            },
            {
                "evidence_id":"EVIDENCE_02_EXACT_SPARSE",
                "class":"POSTFREEZE_EXACT_SPARSE_AUDIT",
                "source":"Verified v2 exact-sparse audit",
                "scope":"2700 M6 target identities; exact vs sparse-m64",
                "manuscript_role":"Approximation-depth sensitivity/audit",
                "replaces_primary":False,
            },
            {
                "evidence_id":"EVIDENCE_03_ESTIMATED_G",
                "class":"POSTFREEZE_ESTIMATED_G_SENSITIVITY",
                "source":"Estimated-g sensitivity v1.0.0",
                "scope":"7 frozen cases x 20 replications x 25 targets; oracle/estimated/misspecified g",
                "manuscript_role":"Treatment-density practical-pipeline sensitivity",
                "replaces_primary":False,
            },
            {
                "evidence_id":"EVIDENCE_04_EXTERNAL_BASELINES",
                "class":"POSTFREEZE_EXTERNAL_BASELINES",
                "source":"External-baseline comparison v1.0.0",
                "scope":"7 frozen cases; E1/E2/E3 plus identical M6 reference",
                "manuscript_role":"External comparator evidence",
                "replaces_primary":False,
            },
            {
                "evidence_id":"EVIDENCE_05_STAGE6A_BOUNDARY",
                "class":"SEPARATELY_FROZEN_NOT_REPROCESSED_HERE",
                "source":"Stage6A NSW real EO demonstration",
                "scope":"Real NSW observed-product applicability; full M6 treatment route nonoperational",
                "manuscript_role":"Real-data applicability/fail-closed boundary",
                "replaces_primary":False,
            },
        ])
        write_csv(evidence,evidence_path)

        claims=pd.DataFrame([
            ["Stage5B primary","M6 mean selective coverage 0.9692; all 27 configuration point estimates >=0.90; strongest minimum local q0.05 among M2-M6.","Registered primary evidence; fitted nuisance implementation is empirical, not automatically theorem-certified."],
            ["Exact-sparse","Mean |Δp| ≈0.00557; q99≈0.09739; max≈0.42945; 9/2700 alpha=0.10 decisions differ.","Aggregate agreement only; no pointwise-equivalence or uniform-closeness claim."],
            ["Estimated g","G_ESTIMATED closely tracks G_ORACLE in mean coverage/local q0.05; misspecification can degrade local q0.05 and distort support diagnostics.","Post-freeze treatment-model sensitivity only; do not replace registered 27-case benchmark."],
            ["External baselines","M6 has strongest selective coverage and lower-tail local robustness in the frozen external-comparison subset.","E3 is substantially narrower in the restricted width audit; no universal efficiency or numerical dominance claim."],
            ["NSW","Treatment-dependent M2/M4/M6 remain scientifically nonoperational without authentic longitudinal rehabilitation treatment.","Observed-product applicability/fail-closed evidence, not full-M6 causal validation."],
        ],columns=["topic","defensible_claim","mandatory_boundary"])
        write_csv(claims,claims_path)

        # Compact manuscript-prep summary figure (not a replacement for original figures).
        ptab=pd.read_csv(primary_path)
        gtab=pd.read_csv(g_overall_path)
        etab=pd.read_csv(ext_overall_path)
        fig, ax = plt.subplots(figsize=(10,6))
        labels=["Stage5B M6\n27 cases","G oracle\n7 cases","G estimated\n7 cases","G misspecified\n7 cases","E1\n6 defined","E2\n7 cases","E3\n7 cases","External M6\n7 cases"]
        vals=[
            float(ptab[ptab.method=="M6"].mean_selective_coverage.iloc[0]),
            float(gtab[gtab.g_track=="G_ORACLE"].mean_case_selective_coverage.iloc[0]),
            float(gtab[gtab.g_track=="G_ESTIMATED"].mean_case_selective_coverage.iloc[0]),
            float(gtab[gtab.g_track=="G_MISSPECIFIED"].mean_case_selective_coverage.iloc[0]),
            float(etab[etab.method=="E1_DRWCP_LOCAL"].mean_case_selective_coverage.iloc[0]),
            float(etab[etab.method=="E2_WCP_JOINT_SHIFT"].mean_case_selective_coverage.iloc[0]),
            float(etab[etab.method=="E3_SLSCP_SPLIT"].mean_case_selective_coverage.iloc[0]),
            float(etab[etab.method=="M6"].mean_case_selective_coverage.iloc[0]),
        ]
        ax.bar(np.arange(len(vals)),vals)
        ax.axhline(0.90,linestyle="--",linewidth=1)
        ax.set_ylim(0.80,1.01)
        ax.set_ylabel("Mean case selective coverage")
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels,rotation=25,ha="right")
        ax.set_title("GeoDose-CP evidence closure: primary and post-freeze sensitivity blocks")
        ax.grid(axis="y",alpha=0.25)
        fig.tight_layout()
        figpath=OUT/"figures"/"FIG_STAGE5C_EVIDENCE_CLOSURE_SUMMARY.png"
        fig.savefig(figpath,dpi=300,bbox_inches="tight")
        plt.close(fig)

        verification["checks"]["manuscript_exports"]={"status":"PASS","evidence_rows":len(evidence),"claim_rows":len(claims)}
        mark_done(state,"06_MANUSCRIPT_EXPORTS",output_hashes([evidence_path,claims_path,figpath]))
    else:
        verification["checks"]["manuscript_exports"]={"status":"PASS_REUSED"}

    # ------------------------------------------------------
    # 07 CLOSURE REPORT
    # ------------------------------------------------------
    report_path=OUT/"STAGE5C_CLOSURE_REPORT.md"
    brief_path=OUT/"MANUSCRIPT_EVIDENCE_BRIEF.md"
    verification_path=OUT/"STAGE5C_VERIFICATION.json"
    if force or not stage_done(state,"07_CLOSURE_REPORT"):
        ptab=pd.read_csv(primary_path)
        xtab=pd.read_csv(exact_path).iloc[0]
        gtab=pd.read_csv(g_overall_path)
        etab=pd.read_csv(ext_overall_path)

        m6p=ptab[ptab.method=="M6"].iloc[0]
        gor=gtab[gtab.g_track=="G_ORACLE"].iloc[0]
        gest=gtab[gtab.g_track=="G_ESTIMATED"].iloc[0]
        gmis=gtab[gtab.g_track=="G_MISSPECIFIED"].iloc[0]
        e3=etab[etab.method=="E3_SLSCP_SPLIT"].iloc[0]
        m6e=etab[etab.method=="M6"].iloc[0]

        report=f"""# GeoDose-CP Stage5C Post-Freeze Computational Closure

## Decision

**PASS / FREEZE MANUSCRIPT-READY COMPUTATIONAL EVIDENCE PACKAGE.**

This closure does not modify the registered Stage5B primary benchmark. It verifies and classifies the registered primary evidence plus three supervisor-requested post-freeze analyses.

## 1. Input integrity and freeze discipline

- All four embedded result ZIPs pass CRC and exact outer SHA-256 checks.
- Stage5B output-manifest files verify against their declared hashes.
- The nested accepted Stage5B production release matches `{contract['accepted_nested_stage5b_release_sha256']}`.
- Estimated-g and external-baseline output-manifest files verify against their declared hashes.
- No Stage5A modification, outcome-model retuning, threshold retuning, or registered-primary modification is accepted by this closure.

## 2. Registered Stage5B primary evidence

M6 across the 27 registered RF-primary observed-scale configurations:
- mean selective coverage: **{m6p.mean_selective_coverage:.6f}**
- minimum selective coverage: **{m6p.minimum_selective_coverage:.6f}**
- cases at/above 0.90: **{int(m6p.cases_at_or_above_0_90)}/27**
- mean local q0.05: **{m6p.mean_local_q0_05:.6f}**
- minimum local q0.05: **{m6p.minimum_local_q0_05:.6f}**
- mean return rate: **{m6p.mean_return_rate:.6f}**

This remains the governing primary empirical evidence.

## 3. Exact-sparse deep audit

- identities: **2700**
- mean |Δp|: **{xtab.mean_abs_delta_p:.9f}**
- median |Δp|: approximately **0**
- q90/q95/q99: **{xtab.q90_abs_delta_p:.6f} / {xtab.q95_abs_delta_p:.6f} / {xtab.q99_abs_delta_p:.6f}**
- maximum |Δp|: **{xtab.max_abs_delta_p:.6f}**
- |Δp| > 0.01: **{int(xtab.n_abs_delta_gt_0_01)}/2700**
- |Δp| > 0.05: **{int(xtab.n_abs_delta_gt_0_05)}/2700**
- |Δp| > 0.10: **{int(xtab.n_abs_delta_gt_0_10)}/2700**
- alpha=0.10 decision disagreements: **{int(xtab.decision_disagreement_count_alpha_0_10)}/2700**
- exact/sparse coverage: **{xtab.exact_coverage:.6f}/{xtab.sparse_coverage:.6f}**

**Boundary:** strong aggregate agreement does not imply pointwise equivalence.

## 4. Estimated g(A|U) sensitivity

Seven prospectively frozen sensitivity cases:
- G_ORACLE mean coverage/local q0.05/return: **{gor.mean_case_selective_coverage:.6f} / {gor.mean_case_local_q0_05:.6f} / {gor.mean_case_return_rate:.6f}**
- G_ESTIMATED: **{gest.mean_case_selective_coverage:.6f} / {gest.mean_case_local_q0_05:.6f} / {gest.mean_case_return_rate:.6f}**
- G_MISSPECIFIED: **{gmis.mean_case_selective_coverage:.6f} / {gmis.mean_case_local_q0_05:.6f} / {gmis.mean_case_return_rate:.6f}**
- G_MISSPECIFIED minimum local q0.05: **{gmis.minimum_case_local_q0_05:.6f}**

Interpretation: fitted g closely tracks oracle aggregate validity, whereas deliberate misspecification can weaken local robustness and can make support diagnostics look artificially favorable. This is secondary sensitivity evidence, not a replacement for Stage5B.

## 5. External baselines

Strongest external comparator E3 versus M6 on the frozen external-comparison subset:
- E3 mean selective coverage: **{e3.mean_case_selective_coverage:.6f}**
- M6 mean selective coverage: **{m6e.mean_case_selective_coverage:.6f}**
- E3 minimum selective coverage: **{e3.minimum_case_selective_coverage:.6f}**
- M6 minimum selective coverage: **{m6e.minimum_case_selective_coverage:.6f}**
- E3 mean/min local q0.05: **{e3.mean_case_local_q0_05:.6f}/{e3.minimum_case_local_q0_05:.6f}**
- M6 mean/min local q0.05: **{m6e.mean_case_local_q0_05:.6f}/{m6e.minimum_case_local_q0_05:.6f}**
- E3 mean restricted-subset width: **{e3.mean_width_on_frozen_70_identity_subset:.6f}**
- M6 mean restricted-subset width: **{m6e.mean_width_on_frozen_70_identity_subset:.6f}**

Interpretation: M6 is strongest for supported-regime selective coverage and lower-tail local robustness, while E3 is substantially sharper on the restricted width audit. **No universal efficiency dominance claim is permitted.**

## 6. Evidence hierarchy

1. Registered Stage5B primary benchmark — governing primary known-truth evidence.
2. Exact-sparse deep audit — post-freeze approximation-depth evidence.
3. Estimated-g sensitivity — post-freeze treatment-density sensitivity.
4. External-baseline comparison — post-freeze comparator evidence.
5. Stage6A NSW — remains separately frozen; not reprocessed here; real observed-product applicability/fail-closed evidence only.

## Final closure decision

**PASS. No additional training or retuning is required before manuscript integration.**
"""
        report_path.write_text(report,encoding="utf-8")

        brief=f"""# Manuscript Evidence Brief

Use these points when revising the paper:

- **Primary benchmark remains unchanged:** Stage5B M6 mean selective coverage {m6p.mean_selective_coverage:.4f}, minimum {m6p.minimum_selective_coverage:.4f}, all 27/27 point estimates >=0.90, minimum local q0.05 {m6p.minimum_local_q0_05:.4f}.
- **Exact-sparse:** mean |Δp| {xtab.mean_abs_delta_p:.5f}; maximum {xtab.max_abs_delta_p:.5f}; only {int(xtab.decision_disagreement_count_alpha_0_10)}/2700 alpha=0.10 decisions differ. State explicitly that aggregate agreement does not imply pointwise equivalence.
- **Estimated treatment density:** G_ESTIMATED closely tracks G_ORACLE in aggregate coverage/local robustness; G_MISSPECIFIED minimum local q0.05 falls to {gmis.minimum_case_local_q0_05:.4f}. Treat this as post-freeze sensitivity evidence.
- **External methods:** M6 mean coverage {m6e.mean_case_selective_coverage:.4f} vs E3 {e3.mean_case_selective_coverage:.4f}; M6 minimum local q0.05 {m6e.minimum_case_local_q0_05:.4f} vs E3 {e3.minimum_case_local_q0_05:.4f}. E3 is much narrower on the restricted width subset, so do not claim universal efficiency dominance.
- **Real NSW:** retain the existing treatment-authenticity/fail-closed boundary. Do not construct a proxy treatment to force M6 operation.
"""
        brief_path.write_text(brief,encoding="utf-8")

        verification["status"]="PASS"
        verification["completed_utc"]=now_iso()
        verification["final_decision"]="PASS_FREEZE_MANUSCRIPT_READY_COMPUTATIONAL_EVIDENCE"
        verification["no_additional_training_required"]=True
        verification["registered_stage5b_primary_modified"]=False
        verification["postfreeze_evidence_replaces_primary"]=False
        verification["stage6a_reprocessed"]=False
        verification_path.write_text(json.dumps(verification,indent=2),encoding="utf-8")

        mark_done(state,"07_CLOSURE_REPORT",output_hashes([report_path,brief_path,verification_path]))
    else:
        verification=json.loads(verification_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------
    # 08 FINAL OUTPUT MANIFEST + results ZIP
    # ------------------------------------------------------
    manifest_path=OUT/"OUTPUT_MANIFEST.json"
    results_zip=OUT/"GeoDose_Stage5C_PostFreeze_Closure_RESULTS.zip"
    # Always rebuild the final archive so it reflects all current verified outputs.
    files=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in {"OUTPUT_MANIFEST.json","GeoDose_Stage5C_PostFreeze_Closure_RESULTS.zip"}:
            files.append({
                "file":str(p.relative_to(OUT)).replace("\\","/"),
                "bytes":p.stat().st_size,
                "sha256":sha256_file(p),
            })
    aggregate=hashlib.sha256(
        "\n".join(f"{r['file']}|{r['bytes']}|{r['sha256']}" for r in files).encode("utf-8")
    ).hexdigest()
    manifest={"stage":"Stage5C","version":"1.0.0","status":"PASS","aggregate_sha256":aggregate,"files":files}
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    with zipfile.ZipFile(results_zip,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and p.name!="GeoDose_Stage5C_PostFreeze_Closure_RESULTS.zip":
                z.write(p,arcname=str(p.relative_to(OUT)).replace("\\","/"))
    verify_zip_crc(results_zip)

    # hash manifest/result for resume record
    mark_done(state,"08_FINAL_PACKAGE",output_hashes([manifest_path,results_zip]))

    print("")
    print("="*72)
    print("STAGE5C VERIFIED COMPLETE")
    print("="*72)
    print(f"Results directory : {OUT}")
    print(f"Final results ZIP : {results_zip}")
    print(f"Results ZIP SHA256: {sha256_file(results_zip)}")
    print("Decision          : PASS / FREEZE MANUSCRIPT-READY EVIDENCE")
    print("No additional training or retuning is required.")
    print("="*72)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--force",action="store_true",help="Recompute all closure stages instead of reusing verified checkpoints.")
    args=ap.parse_args()
    try:
        main(force=args.force)
    except Exception as exc:
        print("\nSTAGE5C CLOSURE FAILED:",repr(exc),file=sys.stderr)
        print("Fix the reported issue and rerun. Previously completed verified stages are retained.",file=sys.stderr)
        raise
