#!/usr/bin/env python3
"""
Data Health Scan — intern starter scaffold.

Architecture (build the spine once, then each check is a small plug-in):
    load_tables()  ->  profile_table()  ->  [ Check.run() for each check ]  ->  score()  ->  ranked report

Every check returns a list of Finding objects. Most checks only read the cheap
COLUMN PROFILE (computed once per table); a few cross-table checks also read the
raw DataFrames. Thresholds live in POLICY so they're config, not magic numbers.

Worked examples included (copy their shape for the rest):
    sparse, empty, constant, missing_key, fake_key   (profile-only, Tier 1)
    sentinel, dirty_categorical, outlier              (value analysis, Tier 2)
    orphaned_reference, key_conformity, nomenclature   (cross-table, Tier 3)
TODO stubs you fill in: validity, cross_field, near_duplicate, units,
    sensitivity, reference_standardization, undocumented_join, stale_data,
    schema_drift, hidden_structure, mixed_format, locked_table.

Run:  python dhs.py --data ./data
Test: pytest test_dhs.py
"""
import argparse, glob, json, os, re, statistics, warnings
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import pandas as pd, numpy as np

# ----------------------------- shared model -----------------------------
@dataclass
class ColumnProfile:
    table: str; name: str; dtype: str
    row_count: int; null_count: int
    distinct: int; top_values: list           # [(value, count), ...] desc
    len_min: int; len_max: int
    num_min: float; num_max: float
    sample: list
    @property
    def null_pct(self): return self.null_count / self.row_count if self.row_count else 0.0
    @property
    def cardinality_ratio(self):
        nonnull = self.row_count - self.null_count
        return self.distinct / nonnull if nonnull else 0.0

@dataclass
class TableProfile:
    name: str; row_count: int
    columns: dict = field(default_factory=dict)

@dataclass
class Finding:
    check_id: str; severity: str; table: str; column: str
    evidence: dict; risk: str
SEV_WEIGHT = {"HIGH": 12, "MED": 6, "LOW": 2}

POLICY = {
    "sparse_null_pct": 0.30, "sentinel_top_share": 0.40,
    "fuzzy_sim": 0.82, "outlier_z": 3.5, "orphan_pct_flag": 0.01,
    "nomenclature_names_per_key": 1,
    "hidden_structure_min_parseable": 2,
    "hidden_structure_min_share": 0.50,
    "cross_field_as_of": "2026-06-12",
    "declared_primary_keys": {},
    "sensitivity_evidence_limit": 5,
    "sensitivity_free_text_hints": ("note", "comment", "description", "remarks", "text"),
    "sensitivity_patterns": {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "dod_id": r"\b\d{10}\b",
        "cui_marking": r"\bCUI//[A-Z0-9/-]+\b",
    },
}
SENTINELS = {"", "na", "n/a", "null", "none", "unknown", "tbd", "xxx", "-1",
             "9999", "999999", "0000-00-00", "1900-01-01", "9999-12-31"}

# ----------------------------- profiler -----------------------------
def load_tables(data_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        out[os.path.splitext(os.path.basename(f))[0]] = pd.read_csv(f, dtype=str, keep_default_na=False)
    return out  # read as str so we see raw values exactly as stored

def profile_table(name, df):
    tp = TableProfile(name=name, row_count=len(df))
    for col in df.columns:
        s = df[col]
        nn = s[(s != "") & (s.notna())]
        vc = nn.value_counts()
        lens = nn.str.len()
        nums = pd.to_numeric(nn, errors="coerce").dropna()
        tp.columns[col] = ColumnProfile(
            table=name, name=col, dtype=("numeric" if len(nums) == len(nn) and len(nn) else "string"),
            row_count=len(df), null_count=int((s == "").sum()),
            distinct=int(nn.nunique()), top_values=list(vc.head(8).items()),
            len_min=int(lens.min()) if len(lens) else 0, len_max=int(lens.max()) if len(lens) else 0,
            num_min=float(nums.min()) if len(nums) else float("nan"),
            num_max=float(nums.max()) if len(nums) else float("nan"),
            sample=list(nn.head(5)))
    return tp

# ----------------------------- checks: Tier 1 (profile-only) -----------------------------
def check_empty(tables, profs):
    return [Finding("empty_table","HIGH",t,"*",{"rows":p.row_count},
            "Schema implies data that isn't there") for t,p in profs.items() if p.row_count==0]

def check_sparse(tables, profs):
    f=[]
    for p in profs.values():
        for c in p.columns.values():
            if c.null_pct > POLICY["sparse_null_pct"]:
                f.append(Finding("sparse_column","MED",c.table,c.name,
                    {"null_pct":round(c.null_pct,2)},"Reports silently undercount; 'total' isn't a total"))
    return f

def check_constant(tables, profs):
    f=[]
    for p in profs.values():
        for c in p.columns.values():
            if c.distinct==1 and c.row_count>1:
                f.append(Finding("constant_column","LOW",c.table,c.name,
                    {"value":c.top_values[0][0] if c.top_values else None},
                    "Field looks meaningful but carries no signal"))
    return f

def check_missing_key(tables, profs):
    f=[]
    for p in profs.values():
        declared=set(POLICY["declared_primary_keys"].get(p.name,set()))
        for c in p.columns.values():
            if c.name in declared:
                continue
            if (c.name.lower().endswith("_id") and c.distinct>0 and
                    abs(c.cardinality_ratio-1.0)<1e-9 and c.distinct==p.row_count):
                f.append(Finding("missing_key","LOW",c.table,c.name,
                    {"cardinality_ratio":round(c.cardinality_ratio,3),
                     "distinct":c.distinct,"rows":p.row_count},
                    "Column behaves like a primary key but is not declared, so joins rely on hidden assumptions"))
    return f

def _locked_table_info(value):
    if isinstance(value,dict) and value.get("locked"):
        return value
    if isinstance(value,Exception):
        return {"locked":True,"reason":str(value)}
    return None

def check_locked_table(tables, profs):
    f=[]
    for table,value in tables.items():
        info=_locked_table_info(value)
        if info:
            # Locked inputs are represented as metadata so the scan can report incomplete coverage.
            f.append(Finding("locked_table","HIGH",table,"*",
                {"reason":info.get("reason","unreadable"),
                 "path":info.get("path")},
                "The scan is incomplete because a table could not be read"))
    return f


def _is_table_own_key(df, column):
    first_col = df.columns[0].lower() if len(df.columns) else ""
    return column.lower().endswith("_id") and column.lower() == first_col

def check_fake_key(tables, profs):
    f=[]
    for p in profs.values():
        for c in p.columns.values():
            # Only flag the table's own identifier; repeated foreign keys are expected in child tables.
            if _is_table_own_key(tables[p.name], c.name) and c.cardinality_ratio < 1.0 and p.row_count>1:
                f.append(Finding("fake_key","MED",c.table,c.name,
                    {"cardinality_ratio":round(c.cardinality_ratio,3),
                     "duplicates":p.row_count-c.distinct},
                    "Duplicate records inflate counts and joins"))
    return f

def _json_shape(value):
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return None
    if isinstance(parsed, dict):
        return "object"
    if isinstance(parsed, list):
        return "array"
    return None

def _hidden_structure_evidence(samples):
    shapes=[shape for shape in (_json_shape(v) for v in samples) if shape]
    if (len(shapes)>=POLICY["hidden_structure_min_parseable"] and
            len(shapes)/max(1,len(samples))>=POLICY["hidden_structure_min_share"]):
        return {"parseable_samples":len(shapes),"shape":max(set(shapes), key=shapes.count)}
    return None

def check_hidden_structure(tables, profs):
    f=[]
    for p in profs.values():
        for c in p.columns.values():
            if c.dtype!="string" or c.distinct<2:
                continue
            samples=[v for v in c.sample if str(v).strip()]
            evidence=_hidden_structure_evidence(samples)
            if evidence:
                f.append(Finding("hidden_structure","MED",c.table,c.name,
                    evidence,
                    "Structured data is hidden in text, so fields cannot be queried directly"))
    return f

# ----------------------------- checks: Tier 2 (value analysis) -----------------------------
def check_sentinel(tables, profs):
    f=[]
    for p in profs.values():
        for c in p.columns.values():
            if not c.top_values or c.distinct<=1: continue
            val,cnt = c.top_values[0]
            share = cnt/(c.row_count-c.null_count) if (c.row_count-c.null_count) else 0
            if share>POLICY["sentinel_top_share"] and str(val).strip().lower() in SENTINELS:
                f.append(Finding("sentinel_value","HIGH",c.table,c.name,
                    {"value":val,"share":round(share,2),"effective_nulls":int(cnt)},
                    "Looks populated, but most of it means 'unknown'"))
    return f

def _clusters(values, thr):
    vals=list(values); seen=set(); groups=[]
    for i,a in enumerate(vals):
        if a in seen: continue
        g=[a]; seen.add(a)
        for b in vals[i+1:]:
            if b in seen: continue
            if SequenceMatcher(None,a.lower(),b.lower()).ratio()>=thr:
                g.append(b); seen.add(b)
        if len(g)>1: groups.append(g)
    return groups

def check_dirty_categorical(tables, profs):
    f=[]
    for p in profs.values():
        for c in p.columns.values():
            if c.dtype!="string" or c.distinct>40 or c.distinct<2: continue
            vals=[v for v,_ in c.top_values]
            for g in _clusters(vals, POLICY["fuzzy_sim"]):
                f.append(Finding("dirty_categorical","MED",c.table,c.name,
                    {"variants":g},"Group-bys split one real category into many"))
    return f

def check_outlier(tables, profs):
    f=[]
    for name,df in tables.items():
        p=profs[name]
        for c in p.columns.values():
            if c.dtype!="numeric": continue
            x=pd.to_numeric(df[c.name],errors="coerce").dropna().values
            if len(x)<8: continue
            med=np.median(x); mad=np.median(np.abs(x-med)) or 1e-9
            z=0.6745*(x-med)/mad
            n=int((np.abs(z)>POLICY["outlier_z"]).sum())
            if n: f.append(Finding("distribution_outlier","LOW",name,c.name,
                {"outliers":n,"max":float(np.max(x))},"Unit-error or bad reading skews the answer"))
    return f

# ----------------------------- checks: Tier 3 (cross-table) -----------------------------
def _key_cols(tables):
    # naive FK discovery for the scaffold: same column name ending _id or 'nsn' in >1 table
    from collections import defaultdict
    idx=defaultdict(list)
    for t,df in tables.items():
        for c in df.columns:
            if c.lower().endswith("_id") or c.lower()=="nsn": idx[c].append(t)
    return {c:ts for c,ts in idx.items() if len(ts)>1}

def check_orphaned_reference(tables, profs):
    f=[]
    for col,ts in _key_cols(tables).items():
        # treat the table where the col is unique-ish as the parent
        parents=[t for t in ts if profs[t].columns[col].cardinality_ratio>0.99]
        children=[t for t in ts if t not in parents]
        for parent in parents:
            pvals=set(tables[parent][col])
            for child in children:
                cvals=tables[child][col]
                orphan=cvals[~cvals.isin(pvals) & (cvals!="")]
                pct=len(orphan)/len(cvals) if len(cvals) else 0
                if pct>POLICY["orphan_pct_flag"]:
                    f.append(Finding("orphaned_reference","HIGH",child,col,
                        {"parent":parent,"orphan_pct":round(pct,3),
                         "examples":list(orphan.unique()[:3])},
                        "Joins drop rows; numbers quietly go missing"))
    return f

def _sig(series):
    s=series[series!=""]
    return {"len_min":int(s.str.len().min()) if len(s) else 0,
            "len_max":int(s.str.len().max()) if len(s) else 0,
            "has_dash":bool(s.str.contains("-").any()),
            "all_digit":bool(s.str.replace("-","",regex=False).str.isdigit().all()) if len(s) else False}

def check_key_conformity(tables, profs):
    f=[]
    for col,ts in _key_cols(tables).items():
        sigs={t:_sig(tables[t][col]) for t in ts}
        base=ts[0]
        for other in ts[1:]:
            a,b=sigs[base],sigs[other]
            mismatch=[k for k in ("len_min","len_max","has_dash","all_digit") if a[k]!=b[k]]
            if mismatch:
                # estimate raw vs normalized (strip dashes, zero-pad) match
                A=set(tables[base][col]); B=set(tables[other][col])
                raw=len(A&B)/max(1,len(B))
                norm=lambda s:{x.replace("-","").zfill(max(a["len_max"],b["len_max"])) for x in s}
                nm=len(norm(A)&norm(B))/max(1,len(norm(B)))
                f.append(Finding("key_conformity","HIGH",f"{base}+{other}",col,
                    {"mismatch":mismatch,"raw_match":round(raw,2),"normalized_match":round(nm,2)},
                    "These tables can't cleanly join — rows drop silently"))
    return f

def check_nomenclature(tables, profs):
    """One key (e.g. NSN) mapped to many names — the '5 names -> 1 NSN' moat."""
    f=[]
    for t,df in tables.items():
        cols=[c.lower() for c in df.columns]
        if "nsn" in cols:
            name_col=next((c for c in df.columns if "name" in c.lower()), None)
            if name_col:
                g=df.groupby("nsn")[name_col].nunique()
                for nsn,n in g.items():
                    if n>POLICY["nomenclature_names_per_key"]:
                        names=list(df[df["nsn"]==nsn][name_col].unique())
                        f.append(Finding("nomenclature_collision","HIGH",t,"nsn↔"+name_col,
                            {"key":nsn,"distinct_names":int(n),"names":names},
                            f"{n} names for one NSN — demand is split, joins miss"))
    return f

# TODO stubs — interns implement these next (same signature, return list[Finding]):
def check_validity(tables,profs): return []
def _row_sample(row):
    for key in ("person_id","item_id","supply_id","wo_id"):
        if key in row.index and str(row[key]).strip():
            return str(row[key])
    return str(row.iloc[0]) if len(row) else ""

def _parse_date_series(series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(series, errors="coerce")

def _columns_by_lower_name(df):
    return {c.lower():c for c in df.columns}

def check_cross_field(tables,profs):
    f=[]
    as_of=pd.Timestamp(POLICY["cross_field_as_of"])
    for t,df in tables.items():
        cols=_columns_by_lower_name(df)

        if "last_exam" in cols and "next_due" in cols:
            last=_parse_date_series(df[cols["last_exam"]])
            due=_parse_date_series(df[cols["next_due"]])
            bad=df[last.notna() & due.notna() & (due < last)]
            if len(bad):
                f.append(Finding("cross_field","MED",t,"last_exam,next_due",
                    {"rule":"next_due>=last_exam","violations":int(len(bad)),
                     "samples":[_row_sample(row) for _,row in bad.head(5).iterrows()]},
                    "Due dates before exam dates make surveillance timelines impossible"))

        if "status" in cols and "next_due" in cols:
            status=df[cols["status"]].astype(str).str.strip().str.upper()
            due=_parse_date_series(df[cols["next_due"]])
            bad=df[
                ((status=="CURRENT") & due.notna() & (due < as_of)) |
                ((status=="OVERDUE") & due.notna() & (due >= as_of))
            ]
            if len(bad):
                f.append(Finding("cross_field","MED",t,"status,next_due",
                    {"rule":"status_matches_due_date","as_of":POLICY["cross_field_as_of"],
                     "violations":int(len(bad)),
                     "samples":[_row_sample(row) for _,row in bad.head(5).iterrows()]},
                    "Status labels disagree with due dates, so compliance counts are wrong"))

        if "consumed" in cols and "ordered" in cols:
            consumed=pd.to_numeric(df[cols["consumed"]], errors="coerce")
            ordered=pd.to_numeric(df[cols["ordered"]], errors="coerce")
            bad=df[consumed.notna() & ordered.notna() & (consumed > ordered)]
            if len(bad):
                f.append(Finding("cross_field","MED",t,"consumed,ordered",
                    {"rule":"consumed<=ordered","violations":int(len(bad)),
                     "samples":[_row_sample(row) for _,row in bad.head(5).iterrows()]},
                    "Consumed quantities above ordered quantities break inventory accounting"))
    return f
def check_near_duplicate(tables,profs): return []
def check_units(tables,profs): return []
def _sensitivity_candidate(column, profile):
    name=column.lower()
    if name.endswith("_id") or "date" in name:
        return False
    if profile.dtype=="numeric":
        return False
    return profile.dtype=="string" and (
        any(hint in name for hint in POLICY["sensitivity_free_text_hints"]) or profile.len_max>=20
    )

def _mask_sensitive(text):
    masked=str(text)
    masked=re.sub(POLICY["sensitivity_patterns"]["ssn"], lambda m: "***-**-"+m.group(0)[-4:], masked)
    masked=re.sub(POLICY["sensitivity_patterns"]["dod_id"], lambda m: "******"+m.group(0)[-4:], masked)
    masked=re.sub(POLICY["sensitivity_patterns"]["cui_marking"], "CUI//***", masked)
    return masked

def _sensitivity_hits(value):
    text=str(value)
    hits=[]
    for label,pattern in POLICY["sensitivity_patterns"].items():
        if re.search(pattern,text):
            hits.append(label)
    return hits

def check_sensitivity(tables,profs):
    f=[]
    limit=POLICY["sensitivity_evidence_limit"]
    for table,df in tables.items():
        if table not in profs:
            continue
        for column in df.columns:
            profile=profs[table].columns[column]
            if not _sensitivity_candidate(column,profile):
                continue
            counts={}
            samples=[]
            for value in df[column]:
                if pd.isna(value) or not str(value).strip():
                    continue
                hits=_sensitivity_hits(value)
                if not hits:
                    continue
                for hit in hits:
                    counts[hit]=counts.get(hit,0)+1
                masked=_mask_sensitive(value)
                if masked not in samples and len(samples)<limit:
                    samples.append(masked)
            if counts:
                f.append(Finding("sensitivity","HIGH",table,column,
                    {"matches":dict(sorted(counts.items())),"samples":samples},
                    "Sensitive identifiers in free text create privacy and compliance exposure"))
    return f
def check_reference_standardization(tables,profs): return []
def check_undocumented_join(tables,profs): return []
def check_stale_data(tables,profs): return []

CHECKS=[check_empty,check_sparse,check_constant,check_fake_key,check_missing_key,check_locked_table,
        check_hidden_structure,
        check_sentinel,check_dirty_categorical,check_outlier,
        check_orphaned_reference,check_key_conformity,check_nomenclature,
        check_validity,check_cross_field,check_near_duplicate,check_units,
        check_sensitivity,check_reference_standardization,check_undocumented_join,check_stale_data]

# ----------------------------- score + report -----------------------------
def score(findings):
    penalty=sum(SEV_WEIGHT.get(f.severity,0) for f in findings)
    return max(0, 100-penalty)

def run_scan(data_dir):
    tables=load_tables(data_dir)
    profs={t:profile_table(t,df) for t,df in tables.items() if isinstance(df,pd.DataFrame)}
    findings=[fd for chk in CHECKS for fd in chk(tables,profs)]
    order={"HIGH":0,"MED":1,"LOW":2}
    findings.sort(key=lambda f:order[f.severity])
    return tables,profs,findings,score(findings)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--data",default="./data"); a=ap.parse_args()
    tables,profs,findings,sc=run_scan(a.data)
    hi=sum(f.severity=="HIGH" for f in findings); me=sum(f.severity=="MED" for f in findings); lo=sum(f.severity=="LOW" for f in findings)
    print(f"\nDATA HEALTH SCORE: {sc}/100    ({hi} HIGH · {me} MED · {lo} LOW)   tables scanned: {len(tables)}\n"+"="*78)
    for f in findings:
        print(f"[{f.severity:4}] {f.check_id:22} {f.table}.{f.column}")
        print(f"        evidence: {f.evidence}")
        print(f"        risk: {f.risk}")
