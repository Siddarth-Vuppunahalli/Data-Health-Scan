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
    profs = {name: dhs.profile_table(name, df) for name, df in tables.items()}
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

def test_stale_data_flags_rows_past_policy_threshold():
    tables = {
        "medical_surveillance": pd.DataFrame({
            "person_id": ["P1", "P2", "P3"],
            "last_exam": ["2025-01-01", "2026-02-01", "2024-12-15"],
            "status": ["CURRENT", "CURRENT", "CURRENT"],
        })
    }

    findings = _findings_for(dhs.check_stale_data, tables)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.check_id == "stale_data"
    assert finding.severity == "MED"
    assert finding.table == "medical_surveillance"
    assert finding.column == "last_exam"
    assert finding.evidence.get("threshold_days") == 365
    assert finding.evidence.get("stale_rows") == 2
    assert set(finding.evidence.get("samples", [])) == {"P1", "P3"}
    assert "refresh" in finding.risk.lower()

def test_stale_data_ignores_fresh_rows():
    tables = {
        "medical_surveillance": pd.DataFrame({
            "person_id": ["P1", "P2"],
            "last_exam": ["2026-01-01", "2026-04-01"],
        })
    }

    findings = _findings_for(dhs.check_stale_data, tables)

    assert findings == []

def test_stale_data_handles_unparseable_dates():
    tables = {
        "medical_surveillance": pd.DataFrame({
            "person_id": ["P1", "P2", "P3"],
            "last_exam": ["not-a-date", "", None],
        })
    }

    findings = _findings_for(dhs.check_stale_data, tables)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.evidence.get("rule") == "parseable_date"
    assert finding.evidence.get("unparseable_rows") == 1
    assert finding.evidence.get("samples") == ["P1"]

def test_stale_data_uses_per_table_recency_policy():
    tables = {
        "work_orders": pd.DataFrame({
            "wo_id": ["WO-1", "WO-2"],
            "date": ["2024-05-01", "2025-05-01"],
        })
    }

    findings = _findings_for(dhs.check_stale_data, tables)

    assert len(findings) == 1
    assert findings[0].table == "work_orders"
    assert findings[0].evidence.get("threshold_days") == 730
    assert findings[0].evidence.get("samples") == ["WO-1"]

def test_stale_data_ignores_empty_and_all_null_columns():
    tables = {
        "medical_surveillance": pd.DataFrame({
            "person_id": ["P1", "P2", "P3"],
            "last_exam": [None, pd.NA, ""],
        }),
        "exposure_logs": pd.DataFrame({
            "sample_id": pd.Series(dtype="object"),
            "sample_date": pd.Series(dtype="object"),
        }),
    }

    findings = _findings_for(dhs.check_stale_data, tables)

    assert findings == []

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
