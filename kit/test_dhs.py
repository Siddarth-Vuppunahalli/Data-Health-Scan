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

def _findings_for(check, tables):
    profs = {name: dhs.profile_table(name, df) for name, df in tables.items()}
    return check(tables, profs)

def test_validity_flags_malformed_configured_identifiers():
    tables = {
        "materials_hmid": pd.DataFrame({
            "hmid_id": ["HM-001", "HM-XYZ", "HM-003"],
            "nsn": ["8010-01-555-1234", "8010015551234", "8010-01-555-9999"],
        }),
        "work_orders": pd.DataFrame({
            "wo_id": ["WO-00001", "WO-123", "WO-00003"],
            "nsn": ["8010-01-555-1234", "8010-01-555-1234", "BAD-NSN"],
        }),
    }

    findings = _findings_for(dhs.check_validity, tables)
    by_column = {(f.table, f.column): f for f in findings}

    assert set(by_column) == {
        ("materials_hmid", "hmid_id"),
        ("materials_hmid", "nsn"),
        ("work_orders", "wo_id"),
        ("work_orders", "nsn"),
    }
    assert "HM-XYZ" in by_column[("materials_hmid", "hmid_id")].evidence["samples"]
    assert "8010015551234" in by_column[("materials_hmid", "nsn")].evidence["samples"]
    assert "WO-123" in by_column[("work_orders", "wo_id")].evidence["samples"]
    assert "BAD-NSN" in by_column[("work_orders", "nsn")].evidence["samples"]
    for finding in findings:
        assert finding.check_id == "validity"
        assert finding.severity == "MED"
        assert finding.evidence["rule"] == "regex_format"
        assert finding.evidence["pattern"]
        assert "join" in finding.risk.lower()

def test_validity_flags_impossible_dates_and_numeric_ranges():
    tables = {
        "surveillance": pd.DataFrame({
            "person_id": ["P1", "P2", "P3"],
            "last_exam": ["2025-01-15", "not-a-date", "3025-01-15"],
            "months_overdue": ["2", "-1", "300"],
        })
    }

    findings = _findings_for(dhs.check_validity, tables)
    by_column = {f.column: f for f in findings}

    assert by_column["last_exam"].evidence["rule"] == "date_plausibility"
    assert set(by_column["last_exam"].evidence["samples"]) == {"not-a-date", "3025-01-15"}
    assert by_column["months_overdue"].evidence["rule"] == "numeric_range"
    assert set(by_column["months_overdue"].evidence["samples"]) == {"-1", "300"}

def test_validity_flags_values_outside_allowed_set():
    tables = {
        "surveillance": pd.DataFrame({
            "person_id": ["P1", "P2", "P3"],
            "status": ["CURRENT", "LATE", "OVERDUE"],
        })
    }

    findings = _findings_for(dhs.check_validity, tables)

    assert len(findings) == 1
    assert findings[0].column == "status"
    assert findings[0].evidence == {
        "rule": "allowed_set",
        "allowed": ["CURRENT", "OVERDUE"],
        "samples": ["LATE"],
    }

def test_validity_does_not_flag_valid_values():
    tables = {
        "inventory": pd.DataFrame({
            "hmid_id": ["HM-001", "HM-002"],
            "nsn": ["8010-01-555-1234", "8010-01-555-9999"],
            "last_exam": ["2025-01-15", "2025-02-15"],
            "qty": ["10", "250"],
            "status": ["CURRENT", "OVERDUE"],
        })
    }

    findings = _findings_for(dhs.check_validity, tables)

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
