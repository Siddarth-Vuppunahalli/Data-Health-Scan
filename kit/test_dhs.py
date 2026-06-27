"""
TDD harness for the Data Health Scan — the synthetic HazMat data has KNOWN seeded
issues, so each check has a ground-truth answer. Run:  pytest test_dhs.py -v
Set DHS_DATA to your data dir (defaults to ./data).

Intern workflow: pick a stub check, write/uncomment its test here first (red),
implement until it passes (green), then refactor. Test-first, like the rest of the firm.
"""
import os
import pandas as pd
import dhs
from dhs import run_scan

DATA = os.environ.get("DHS_DATA", "./data")
_, _, FINDINGS, SCORE = run_scan(DATA)

def ids(check_id):
    return [f for f in FINDINGS if f.check_id == check_id]

def _findings_for(check, tables):
    profs = {
        name: dhs.profile_table(name, df)
        for name, df in tables.items()
        if isinstance(df, pd.DataFrame)
    }
    return check(tables, profs)

# ---- seeded issues that MUST be found (pass with the starter scaffold) ----
def test_orphan_shop_join_found():
    orph = ids("orphaned_reference")
    assert any(f.table == "roster_hr" and "S99" in f.evidence.get("examples", []) for f in orph)

def test_nomenclature_five_names_one_nsn():
    nom = ids("nomenclature_collision")
    assert any(f.evidence.get("distinct_names", 0) >= 5 for f in nom)

def test_duplicate_supply_id_found():
    fk = ids("fake_key")
    assert any(f.table == "supply_transactions" and f.column == "supply_id" for f in fk)

def test_score_is_low_for_dirty_data():
    assert SCORE < 70   # this dataset is deliberately unhealthy

def test_missing_key_flags_unique_id_candidate():
    tables = {
        "assets": pd.DataFrame({
            "asset_id": ["A1", "A2", "A3"],
            "name": ["one", "two", "three"],
        })
    }

    findings = _findings_for(dhs.check_missing_key, tables)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.check_id == "missing_key"
    assert finding.severity == "LOW"
    assert finding.table == "assets"
    assert finding.column == "asset_id"
    assert finding.evidence.get("cardinality_ratio") == 1.0
    assert "primary key" in finding.risk.lower()

def test_missing_key_ignores_declared_keys_and_duplicate_ids():
    original = dhs.POLICY.get("declared_primary_keys", {}).copy()
    dhs.POLICY["declared_primary_keys"] = {"assets": {"asset_id"}}
    try:
        tables = {
            "assets": pd.DataFrame({
                "asset_id": ["A1", "A2", "A3"],
            }),
            "events": pd.DataFrame({
                "asset_id": ["A1", "A1", "A2"],
            }),
        }

        findings = _findings_for(dhs.check_missing_key, tables)
    finally:
        dhs.POLICY["declared_primary_keys"] = original

    assert findings == []

def test_locked_table_flags_unreadable_table_metadata():
    tables = {
        "readable": pd.DataFrame({"row_id": ["R1"]}),
        "secure_export": {
            "locked": True,
            "reason": "permission denied",
            "path": "secure_export.csv",
        },
    }

    findings = _findings_for(dhs.check_locked_table, tables)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.check_id == "locked_table"
    assert finding.severity == "HIGH"
    assert finding.table == "secure_export"
    assert finding.column == "*"
    assert finding.evidence.get("reason") == "permission denied"
    assert "incomplete" in finding.risk.lower()

# ---- TODO: write these RED, then implement the stub to turn them GREEN ----
# def test_state_spelled_three_ways():
#     # shops.state = California / CA / Calif  -> needs reference_standardization
#     std = ids("reference_standardization")
#     assert any(f.table == "shops" and f.column == "state" for f in std)
#
# def test_surveillance_status_vs_date_consistency():
#     # status CURRENT but last_exam stale -> needs cross_field
#     assert any(f.table == "medical_surveillance" for f in ids("cross_field"))
#
# def test_dirty_categorical_excludes_id_and_date_columns():
#     bad = [f for f in ids("dirty_categorical")
#            if f.column.endswith("_id") or "date" in f.column]
#     assert bad == []
