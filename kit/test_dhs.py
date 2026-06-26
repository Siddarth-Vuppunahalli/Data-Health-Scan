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

def test_fake_key_ignores_foreign_key_reuse():
    fk = ids("fake_key")
    assert not any(f.column == "shop_id" for f in fk)
    assert not any(f.table == "exposure_logs" and f.column == "person_id" for f in fk)

def test_score_is_low_for_dirty_data():
    assert SCORE < 70   # this dataset is deliberately unhealthy

def test_hidden_structure_detects_json_text():
    tables = {
        "events": pd.DataFrame({
            "event_id": ["E1", "E2", "E3"],
            "payload": [
                '{"lot": "A1", "qty": 3}',
                '{"lot": "A2", "qty": 5}',
                '{"lot": "A3", "qty": 8}',
            ],
            "status": ["ok", "ok", "ok"],
            "notes": ["normal text", "plain comment", "follow up"],
        })
    }

    findings = _findings_for(dhs.check_hidden_structure, tables)

    assert any(f.table == "events" and f.column == "payload" for f in findings)
    assert not any(f.table == "events" and f.column in {"status", "notes"} for f in findings)

def test_cross_field_flags_surveillance_status_date_violations():
    tables = {
        "medical_surveillance": pd.DataFrame({
            "person_id": ["P1", "P2", "P3"],
            "last_exam": ["2025-01-15", "2025-02-15", "2025-03-15"],
            "next_due": ["2026-07-15", "2026-01-15", "2026-08-15"],
            "status": ["CURRENT", "CURRENT", "OVERDUE"],
        })
    }

    findings = _findings_for(dhs.check_cross_field, tables)

    status_findings = [
        f for f in findings
        if f.table == "medical_surveillance" and f.column == "status,next_due"
    ]
    assert status_findings
    samples = {sample for f in status_findings for sample in f.evidence.get("samples", [])}
    assert "P2" in samples
    assert "P3" in samples

def test_cross_field_flags_due_before_exam():
    tables = {
        "medical_surveillance": pd.DataFrame({
            "person_id": ["P1", "P2"],
            "last_exam": ["2025-05-15", "2025-06-15"],
            "next_due": ["2026-05-15", "2025-01-15"],
            "status": ["CURRENT", "OVERDUE"],
        })
    }

    findings = _findings_for(dhs.check_cross_field, tables)

    assert any(
        f.table == "medical_surveillance"
        and f.column == "last_exam,next_due"
        and "P2" in f.evidence.get("samples", [])
        for f in findings
    )

def test_cross_field_flags_consumed_above_ordered():
    tables = {
        "usage": pd.DataFrame({
            "item_id": ["I1", "I2", "I3"],
            "ordered": ["10", "20", "30"],
            "consumed": ["5", "25", "10"],
        })
    }

    findings = _findings_for(dhs.check_cross_field, tables)

    assert any(
        f.table == "usage"
        and f.column == "consumed,ordered"
        and "I2" in f.evidence.get("samples", [])
        for f in findings
    )

def test_cross_field_ignores_consistent_rows():
    tables = {
        "medical_surveillance": pd.DataFrame({
            "person_id": ["P1", "P2"],
            "last_exam": ["2025-01-15", "2025-02-15"],
            "next_due": ["2026-07-15", "2026-08-15"],
            "status": ["CURRENT", "CURRENT"],
        }),
        "usage": pd.DataFrame({
            "item_id": ["I1", "I2"],
            "ordered": ["10", "20"],
            "consumed": ["5", "20"],
        }),
    }

    findings = _findings_for(dhs.check_cross_field, tables)

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
