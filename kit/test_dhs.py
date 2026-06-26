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

def test_units_flags_mixed_units_with_evidence():
    tables = {
        "exposure_logs": pd.DataFrame({
            "sample_id": ["IH-1", "IH-2", "IH-3"],
            "dust_mass": ["5 mg", "0.002 g", "7 mg"],
        })
    }

    findings = _findings_for(dhs.check_units, tables)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.check_id == "units"
    assert finding.severity == "HIGH"
    assert finding.table == "exposure_logs"
    assert finding.column == "dust_mass"
    assert finding.evidence.get("units") == ["g", "mg"]
    assert "0.002 g" in finding.evidence.get("samples", [])
    assert "total" in finding.risk.lower()

def test_units_ignores_single_unit_and_unitless_numeric_columns():
    tables = {
        "exposure_logs": pd.DataFrame({
            "dust_mass": ["5 mg", "2 mg", "7 mg"],
            "cr6_ug_m3": ["2.4", "5.1", "3.2"],
        })
    }

    findings = _findings_for(dhs.check_units, tables)

    assert findings == []

def test_units_normalizes_case_and_ignores_blanks():
    tables = {
        "exposure_logs": pd.DataFrame({
            "dust_mass": ["5 MG", "0.002 g", None, pd.NA, "", " "],
        })
    }

    findings = _findings_for(dhs.check_units, tables)

    assert len(findings) == 1
    assert findings[0].evidence.get("units") == ["g", "mg"]

def test_units_ignores_free_text_and_unknown_units():
    tables = {
        "notes": pd.DataFrame({
            "comment": ["mix 5 mg into sample", "handled by shop", "urgent"],
            "rating": ["5 score", "4 score", "3 score"],
        })
    }

    findings = _findings_for(dhs.check_units, tables)

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
