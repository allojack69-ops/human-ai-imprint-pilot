#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

def family(name):
    n=(name or "").strip().lower()
    if "chatgpt" in n or "gpt" in n or "openai" in n:return "openai"
    if "claude" in n or "anthropic" in n:return "anthropic"
    if "gemini" in n or "google" in n:return "google"
    if "grok" in n or "xai" in n:return "xai"
    if "deepseek" in n:return "deepseek"
    if "llama" in n or "meta" in n:return "meta"
    return n or "unknown"

def cosine(a,b):
    ks=sorted(set(a)&set(b))
    if not ks:return np.nan
    x=np.array([a[k] for k in ks],float);y=np.array([b[k] for k in ks],float)
    nx=np.linalg.norm(x);ny=np.linalg.norm(y)
    return float(x@y/(nx*ny)) if nx and ny else np.nan

def sign_agreement(a,b,eps=.15):
    ks=sorted(set(a)&set(b))
    if not ks:return np.nan
    s=lambda v:1 if v>eps else(-1 if v<-eps else 0)
    return float(np.mean([s(a[k])==s(b[k]) for k in ks]))

def load(db):
    con=sqlite3.connect(db);con.row_factory=sqlite3.Row
    h={r["participant_id"]:json.loads(r["vector_json"]) for r in con.execute("select * from fingerprints")}
    a={r["participant_id"]:{
        "name":r["model_name"],"family":family(r["model_name"]),"vec":json.loads(r["vector_json"])
    } for r in con.execute("select * from ai_fingerprints")}
    con.close();return h,a

def identify(h,a,within_family=False):
    ids=sorted(set(h)&set(a)); rows=[];hits=0;total=0
    for pid in ids:
        fam=a[pid]["family"]
        cand=[]
        for aid in ids:
            if within_family and a[aid]["family"]!=fam:continue
            s=cosine(h[pid],a[aid]["vec"])
            if not np.isnan(s):cand.append((aid,s))
        if within_family and len(cand)<2:continue
        cand.sort(key=lambda x:x[1],reverse=True)
        if not cand:continue
        pred=cand[0][0];total+=1;hits+=int(pred==pid)
        rows.append({"participant_id":pid,"predicted_ai_owner":pred,"correct":pred==pid,
                     "best_similarity":cand[0][1],"model_family":fam})
    return (hits/total if total else np.nan),pd.DataFrame(rows),total

def stratified_permutation_p(h,a,observed,n=5000,seed=42):
    if np.isnan(observed):return np.nan
    ids=sorted(set(h)&set(a)); groups={}
    for pid in ids:groups.setdefault(a[pid]["family"],[]).append(pid)
    eligible={f:g for f,g in groups.items() if len(g)>=2}
    eligible_ids=[x for g in eligible.values() for x in g]
    if not eligible_ids:return np.nan
    rng=random.Random(seed); vals=[]
    for _ in range(n):
        perm_map={}
        for fam,g in eligible.items():
            shuffled=g[:];rng.shuffle(shuffled)
            perm_map.update(dict(zip(g,shuffled)))
        fake={}
        for pid in eligible_ids:
            source=perm_map[pid]
            fake[pid]=a[source]
        hh={pid:h[pid] for pid in eligible_ids}
        acc,_,_=identify(hh,fake,within_family=True)
        if not np.isnan(acc):vals.append(acc)
    return (sum(v>=observed for v in vals)+1)/(len(vals)+1) if vals else np.nan

def bootstrap_gap(matched,mismatch,n=5000,seed=42):
    if not matched or not mismatch:return np.nan,np.nan,np.nan
    rng=np.random.default_rng(seed);gap=float(np.mean(matched)-np.mean(mismatch));boots=[]
    for _ in range(n):
        boots.append(np.mean(rng.choice(matched,len(matched),replace=True))-
                     np.mean(rng.choice(mismatch,len(mismatch),replace=True)))
    lo,hi=np.quantile(boots,[.025,.975]);return gap,float(lo),float(hi)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("db");ap.add_argument("--out",default="analysis_results")
    ap.add_argument("--permutations",type=int,default=5000)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    h,a=load(args.db);ids=sorted(set(h)&set(a))

    pairs=[];matched=[];mismatch=[];within=[]
    for hid in ids:
        for aid in ids:
            s=cosine(h[hid],a[aid]["vec"])
            same=hid==aid;same_family=a[hid]["family"]==a[aid]["family"]
            pairs.append({"human_id":hid,"ai_id":aid,"matched":same,"same_model_family":same_family,
                          "human_pair_family":a[hid]["family"],"ai_model_family":a[aid]["family"],"cosine":s})
            if np.isnan(s):continue
            if same:matched.append(s)
            else:
                mismatch.append(s)
                if same_family:within.append(s)
    pd.DataFrame(pairs).to_csv(out/"pair_matrix.csv",index=False)

    acc,id_df,nid=identify(h,a,False);id_df.to_csv(out/"identification_all.csv",index=False)
    wacc,wdf,wn=identify(h,a,True);wdf.to_csv(out/"identification_within_family.csv",index=False)
    wp=stratified_permutation_p(h,a,wacc,args.permutations)

    gap,lo,hi=bootstrap_gap(matched,mismatch)
    wgap,wlo,whi=bootstrap_gap(matched,within)

    per=[]
    for pid in ids:
        per.append({"participant_id":pid,"model_name":a[pid]["name"],"model_family":a[pid]["family"],
                    "matched_cosine":cosine(h[pid],a[pid]["vec"]),
                    "sign_agreement":sign_agreement(h[pid],a[pid]["vec"]),
                    "shared_dimensions":len(set(h[pid])&set(a[pid]["vec"]))})
    pd.DataFrame(per).to_csv(out/"participant_pairs.csv",index=False)

    report=f"""# Human–AI Imprint Pilot v1.2 — Analysis

Paired participants: {len(ids)}

## All-model comparison
Mean matched cosine: {np.mean(matched) if matched else np.nan:.4f}
Mean mismatched cosine: {np.mean(mismatch) if mismatch else np.nan:.4f}
Matched − mismatched gap: {gap:.4f}
Bootstrap 95% CI: {lo:.4f} .. {hi:.4f}
Top-1 identification: {acc:.4f} (n={nid})
Nominal chance: {(1/len(ids)) if ids else np.nan:.4f}

## Model-family controlled comparison
Mean same-family mismatched cosine: {np.mean(within) if within else np.nan:.4f}
Matched − same-family mismatch gap: {wgap:.4f}
Bootstrap 95% CI: {wlo:.4f} .. {whi:.4f}
Within-family top-1 identification: {wacc:.4f} (eligible n={wn})
Stratified permutation p-value: {wp:.6f}

## Strong evidence gate
The human-imprint hypothesis is materially stronger only if:
1. matched similarity exceeds mismatched similarity;
2. matched similarity also exceeds mismatched pairs within the SAME model family;
3. within-family identification beats its shuffled baseline;
4. results reproduce on held-out participants;
5. repeat-session reliability is acceptable;
6. a later free-text layer survives style normalization.
"""
    (out/"REPORT.md").write_text(report,encoding="utf-8");print(report)

if __name__=="__main__":main()
