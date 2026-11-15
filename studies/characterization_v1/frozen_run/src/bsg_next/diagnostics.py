"""Bounded CPU development checks, permanently excluded from paper promotion."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

from .io import atomic_json, prepare_run
from .statistics import binomial_interval
from .synthetic import simulate_seed


def diagnose_context(config, output, *, replicates=12, max_seconds=180., family_sizes=(32,48,64),
                     simulator=None, clock=None):
    """Paired independent development seeds; never authorize confirmation.

    The soft wall budget is checked between fits; one fit is not interrupted.
    A production calibration must use independent seeds and full registered grids.
    """
    if not 1<=int(replicates)<=200:raise ValueError("Development replicates must be 1..200")
    if not 0<float(max_seconds)<=3600:raise ValueError("Development wall budget must be 0..3600 seconds")
    if tuple(family_sizes)!=(32,48,64):raise ValueError("Keep the full paired development grid 32/48/64")
    output=Path(output)
    if output.exists():raise ValueError("Development diagnosis requires a fresh output directory")
    local=deepcopy(config)
    local.update({"technical_test":True,"paper_eligible":False,
                  "execution_purpose":"bounded_context_power_development"})
    plan={"replicates":int(replicates),"family_sizes":list(family_sizes),
          "development_seed_start":2026101900,"max_seconds":float(max_seconds),
          "cell":["Y4","C7",.1,1.],"paired_seeds_across_sizes":True,
          "independent_validation_required":True,"paper_eligible":False}
    prepare_run(output,local,"context_development",provenance=plan)
    fit=simulator or simulate_seed;now=clock or time.monotonic
    started=now();records={n:[] for n in family_sizes};timed_out=False
    for rep in range(int(replicates)):
        for n in family_sizes:
            if now()-started>=max_seconds:
                timed_out=True;break
            candidate=deepcopy(local);candidate.setdefault("synthetic",{})["test_families"]=n
            record=fit(candidate,("Y4","C7",.1,1.),rep,2026101900+rep)
            record.update({"technical_test":True,"paper_eligible":False,"test_families":n,
                           "development_only":True})
            records[n].append(record)
            atomic_json(output/"seeds"/f"families_{n}_rep_{rep:04d}.json",record)
            print(f"DEVELOPMENT families={n} replicate={rep+1}/{replicates}",flush=True)
        if timed_out:break
    summary=[]
    for n,rs in records.items():
        count=sum(r["decisions"]["both_groups_both_endpoints_superior"] for r in rs)
        summary.append({"test_families":n,"completed":len(rs),
                        "joint_detection":binomial_interval(count,len(rs)),
                        "individual_successes":{key:sum(r["decisions"]["tests"][key]["reject"] for r in rs)
                            for key in ("C7.pred.superiority","C7.ctx.superiority","D4.pred.superiority","D4.ctx.superiority")}})
    result={**plan,"technical_test":True,"status":"budget_exhausted" if timed_out else "development_complete",
            "wall_seconds":now()-started,"planned_fits":len(family_sizes)*int(replicates),
            "completed_fits":sum(map(len,records.values())),"candidates":summary,
            "full_calibration_passed":False,"confirmation_authorized":False,
            "interpretation":"Small development screen only; no candidate has established 90% joint detection."}
    atomic_json(output/"summary.json",result)
    return result
