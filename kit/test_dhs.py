"""
TDD harness for the Data Health Scan — the synthetic HazMat data has KNOWN seeded
issues, so each check has a ground-truth answer. Run:  pytest test_dhs.py -v
Set DHS_DATA to your data dir (defaults to ./data).

Intern workflow: pick a stub check, write/uncomment its test here first (red),
implement until it passes (green), then refactor. Test-first, like the rest of the firm.
"""
import os
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
