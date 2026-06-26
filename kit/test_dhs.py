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

def test_mixed_format_detects_incompatible_code_patterns():
    tables = {
        "assets": pd.DataFrame({
            "asset_code": ["AB-1234", "AB-5678", "123-ABCD", "XY_9999"],
            "clean_code": ["ZZ-1000", "ZZ-1001", "ZZ-1002", "ZZ-1003"],
        })
    }

    findings = _findings_for(dhs.check_mixed_format, tables)

    assert any(f.table == "assets" and f.column == "asset_code" for f in findings)
    assert not any(f.table == "assets" and f.column == "clean_code" for f in findings)

def test_mixed_format_uses_value_shape_when_name_is_not_code_like():
    tables = {
        "assets": pd.DataFrame({
            "plate": ["AB-1234", "AB-5678", "123-ABCD", "XY_9999"],
            "role": ["Maintainer", "Painter", "Inspector", "Supervisor"],
        })
    }

    findings = _findings_for(dhs.check_mixed_format, tables)

    assert any(f.table == "assets" and f.column == "plate" for f in findings)
    assert not any(f.table == "assets" and f.column == "role" for f in findings)

def test_mixed_format_reports_date_split_and_dominant_fix():
    tables = {
        "exposure_logs": pd.DataFrame({
            "sample_date": [
                "2026-01-01", "2026-01-02", "2026-01-03", "01/04/2026",
                None, pd.NA, "", " ",
            ],
        })
    }

    findings = _findings_for(dhs.check_mixed_format, tables)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.check_id == "mixed_format"
    assert finding.severity == "MED"
    assert finding.table == "exposure_logs"
    assert finding.column == "sample_date"
    assert finding.evidence.get("patterns") == {
        "YYYY-MM-DD": 3,
        "MM/DD/YYYY": 1,
    }
    assert finding.evidence.get("dominant_format") == "YYYY-MM-DD"
    assert finding.evidence.get("suggested_format") == "YYYY-MM-DD"
    assert "sort" in finding.risk.lower()

def test_mixed_format_ignores_clean_dates_and_nulls():
    tables = {
        "exposure_logs": pd.DataFrame({
            "sample_date": ["2026-01-01", "2026-01-02", None, pd.NA, "", " "],
        })
    }

    findings = _findings_for(dhs.check_mixed_format, tables)

    assert findings == []

def test_mixed_format_ignores_all_null_date_column():
    tables = {
        "exposure_logs": pd.DataFrame({
            "sample_date": [None, pd.NA, "", " "],
        })
    }

    findings = _findings_for(dhs.check_mixed_format, tables)

    assert findings == []

def test_mixed_format_reports_tie_without_format_suggestion():
    tables = {
        "supply_transactions": pd.DataFrame({
            "transaction_date": [
                "2026-01-01", "2026-01-02", "01/03/2026", "01/04/2026",
            ],
        })
    }

    findings = _findings_for(dhs.check_mixed_format, tables)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.evidence.get("patterns") == {
        "YYYY-MM-DD": 2,
        "MM/DD/YYYY": 2,
    }
    assert finding.evidence.get("dominant_format") is None
    assert finding.evidence.get("suggested_format") is None

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
