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

def test_fake_key_ignores_foreign_key_reuse():
    fk = ids("fake_key")
    assert not any(f.column == "shop_id" for f in fk)
    assert not any(f.table == "exposure_logs" and f.column == "person_id" for f in fk)

def test_score_is_low_for_dirty_data():
    assert SCORE < 70   # this dataset is deliberately unhealthy

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

def test_hidden_structure_ignores_sparse_json_text():
    tables = {
        "events": pd.DataFrame({
            "event_id": ["E1", "E2", "E3", "E4"],
            "notes": ['{"lot": "A1"}', "plain text", "follow up", "normal"],
        })
    }

    findings = _findings_for(dhs.check_hidden_structure, tables)

    assert findings == []

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

def test_cross_field_ignores_blank_or_malformed_dates():
    tables = {
        "medical_surveillance": pd.DataFrame({
            "person_id": ["P1", "P2", "P3"],
            "last_exam": ["", "not-a-date", "2025-02-15"],
            "next_due": ["", "2026-08-15", "not-a-date"],
            "status": ["CURRENT", "CURRENT", "OVERDUE"],
        })
    }

    findings = _findings_for(dhs.check_cross_field, tables)

    assert findings == []

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

def test_missing_key_ignores_empty_and_all_null_id_columns():
    tables = {
        "empty": pd.DataFrame({
            "asset_id": pd.Series(dtype="object"),
        }),
        "nulls": pd.DataFrame({
            "asset_id": [None, pd.NA, "", " "],
        }),
    }

    findings = _findings_for(dhs.check_missing_key, tables)

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

def test_locked_table_accepts_exception_metadata():
    tables = {
        "secure_export": PermissionError("permission denied"),
    }

    findings = _findings_for(dhs.check_locked_table, tables)

    assert len(findings) == 1
    assert findings[0].table == "secure_export"
    assert findings[0].evidence.get("reason") == "permission denied"

def test_sensitivity_flags_ssn_dod_id_and_cui_with_masked_samples():
    tables = {
        "case_notes": pd.DataFrame({
            "note_id": ["N1", "N2", "N3"],
            "notes": [
                "Employee SSN 123-45-6789 included in note",
                "DoD ID 1234567890 copied from badge",
                "Marked CUI//SP-PRVCY before export",
            ],
        })
    }

    findings = _findings_for(dhs.check_sensitivity, tables)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.check_id == "sensitivity"
    assert finding.severity == "HIGH"
    assert finding.table == "case_notes"
    assert finding.column == "notes"
    assert finding.evidence.get("matches") == {
        "cui_marking": 1,
        "dod_id": 1,
        "ssn": 1,
    }
    assert finding.evidence.get("samples") == [
        "Employee SSN ***-**-6789 included in note",
        "DoD ID ******7890 copied from badge",
        "Marked CUI//*** before export",
    ]
    assert "exposure" in finding.risk.lower()

def test_sensitivity_ignores_clean_free_text():
    tables = {
        "case_notes": pd.DataFrame({
            "notes": [
                "Employee completed training",
                "Badge was checked without storing the number",
                "Privacy review completed",
            ],
        })
    }

    findings = _findings_for(dhs.check_sensitivity, tables)

    assert findings == []

def test_sensitivity_ignores_structured_id_columns():
    tables = {
        "badges": pd.DataFrame({
            "badge_id": ["1234567890", "2345678901"],
            "notes": ["badge checked", "badge checked"],
        })
    }

    findings = _findings_for(dhs.check_sensitivity, tables)

    assert findings == []

def test_sensitivity_ignores_empty_and_all_null_columns():
    tables = {
        "case_notes": pd.DataFrame({
            "notes": [None, pd.NA, "", " "],
        })
    }

    findings = _findings_for(dhs.check_sensitivity, tables)

    assert findings == []

def test_sensitivity_limits_masked_samples():
    tables = {
        "case_notes": pd.DataFrame({
            "notes": [
                "SSN 100-10-1000",
                "SSN 100-10-1001",
                "SSN 100-10-1002",
                "SSN 100-10-1003",
                "SSN 100-10-1004",
                "SSN 100-10-1005",
            ],
        })
    }

    findings = _findings_for(dhs.check_sensitivity, tables)

    assert len(findings) == 1
    assert findings[0].evidence.get("matches") == {"ssn": 6}
    assert len(findings[0].evidence.get("samples")) == dhs.POLICY["sensitivity_evidence_limit"]
    assert findings[0].evidence["samples"][0] == "SSN ***-**-1000"

def test_key_conformity_flags_normalizable_nsn_length_mismatch():
    tables = {
        "materials_hmid": pd.DataFrame({
            "nsn": ["8010-01-555-1234", "8010-01-555-9999"],
        }),
        "work_orders": pd.DataFrame({
            "nsn": ["8010015551234", "8010015559999"],
        }),
    }

    findings = _findings_for(dhs.check_key_conformity, tables)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.check_id == "key_conformity"
    assert finding.severity == "HIGH"
    assert finding.column == "nsn"
    assert set(finding.evidence.get("mismatch")) >= {"has_dash", "len_min", "len_max"}
    assert finding.evidence.get("raw_match") == 0.0
    assert finding.evidence.get("normalized_match") == 1.0
    assert finding.evidence.get("examples") == [
        {"base": "8010-01-555-1234", "other": "8010015551234"},
        {"base": "8010-01-555-9999", "other": "8010015559999"},
    ]
    assert "join" in finding.risk.lower()

def test_key_conformity_ignores_compatible_key_formats():
    tables = {
        "materials_hmid": pd.DataFrame({
            "nsn": ["8010-01-555-1234", "8010-01-555-9999"],
        }),
        "work_orders": pd.DataFrame({
            "nsn": ["8010-01-555-1234", "8010-01-555-9999"],
        }),
    }

    findings = _findings_for(dhs.check_key_conformity, tables)

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
