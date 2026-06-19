# Data Health Scan — Intern Build Guide (with time estimates)

You're building **one scan** that profiles a database and returns a **Data Health
Score (0–100)** plus a **ranked list of issues**. Each issue type is a small,
independent **check**. The hard architecture is already done for you in `dhs.py`; your
job is to add checks and tune them.

The gift: the synthetic HazMat dataset has **known, seeded problems**, so every check
has a right answer. That's your test oracle (`test_dhs.py`). Build test-first.

---

## How the code is organized (read `dhs.py` once, top to bottom)

```
load_tables()       reads the CSVs
profile_table()     -> TableProfile { columns: {name: ColumnProfile} }   (run once per table; cheap)
Check functions     run(tables, profiles) -> list[Finding]               (you add these)
score()             100 minus weighted penalties
run_scan()          glues it together; CLI prints the ranked report
```

Two objects are all you need to understand:
- **`ColumnProfile`** — everything already computed about one column: `null_pct`,
  `distinct`, `cardinality_ratio`, `top_values`, `len_min/len_max`, `num_min/num_max`,
  `sample`. **Most checks just read this.**
- **`Finding`** — what a check returns: `check_id, severity (HIGH/MED/LOW), table,
  column, evidence{}, risk`.

**The pattern for every check is identical:**
```python
def check_myrule(tables, profiles):
    findings = []
    for p in profiles.values():
        for c in p.columns.values():
            if <condition on c>:
                findings.append(Finding("my_rule","MED",c.table,c.name,
                    {"evidence_key": value}, "one-line business risk"))
    return findings
```
Add the function name to the `CHECKS` list and it runs. That's the whole contract.

---

## Step 0 — Setup (½ day, once, pair on it)
1. `python -m venv .venv && source .venv/bin/activate && pip install pandas numpy pytest`
2. Copy the `data/` folder from `FINAL_HAZMAT_DEMO/data`.
3. Run the starter: `python dhs.py --data ./data` — you should see SCORE 0/100 and the
   orphan-join, 5-names-1-NSN, and duplicate-supply-id findings.
4. Run the oracle: `pytest test_dhs.py -v` — four tests pass. This is your safety net.
5. Read `dhs.py` end to end and the v2 feature-set doc (Part B has the algorithm for
   every check). **Do not write a check until you've read its Part-B entry.**

## Step 1 — Understand the worked examples (½ day)
`dhs.py` ships **11 checks already implemented** as templates. Read these three first,
they cover the three difficulty tiers:
- `check_sparse` (Tier 1 — pure profile read)
- `check_sentinel` (Tier 2 — value frequency + a dictionary)
- `check_key_conformity` / `check_orphaned_reference` (Tier 3 — two tables at once)

---

## Step 2 — Build the remaining checks (the bulk of the work)

Pick a check, **write its test in `test_dhs.py` first** (red), implement, make it green,
refactor. Estimates assume an intern pace (learning the codebase as you go) and include
writing the test.

### Tier 1 — profile-only, easy wins (≈1–2 hrs each)
| Check | What to do | Est. |
|---|---|---|
| `check_missing_key` (finish stub) | flag a unique `*_id` not declared PK (needs a catalog lookup; for now flag unique-but-undeclared candidates) | 2 hrs |
| `hidden_structure` | detect JSON/encoded text in a plain column (try `json.loads` on samples) | 1.5 hrs |
| `mixed_format` | one column, several `value_pattern_signature`s (regex-mask the values, count distinct masks) | 2 hrs |
| `locked_table` | record tables/columns the connector couldn't read | 1 hr (mostly plumbing) |

### Tier 2 — value analysis (≈½–1 day each)
| Check | What to do | Est. |
|---|---|---|
| `check_validity` | per-column rules: numeric range (p1/p99), date plausibility, allowed-set; report violations | 1 day |
| `check_cross_field` | row-level predicates: `next_due >= last_exam`, `consumed <= ordered`, status-vs-date | 1 day |
| **tune `check_dirty_categorical`** | exclude `*_id` and date columns (the false positives in the starter run); keep only low-cardinality text | ½ day |
| **tune `check_fake_key`** | only flag an `*_id` that is the table's *own* key (e.g. `supply_id` in `supply_transactions`), not foreign keys like `shop_id` | ½ day |

### Tier 3 — cross-table / reference data (≈1–2 days each)
| Check | What to do | Est. |
|---|---|---|
| `check_reference_standardization` | conform values to a dictionary: ISO/FIPS states (catches California/CA/Calif that string-similarity misses), then defense masks (CAGE 5-alnum, NSN 13-digit, DoDAAC 6) | 2 days |
| `check_near_duplicate` | blocking key (`soundex(name)+shop`) + fuzzy match (`difflib` or `rapidfuzz`) to find the same person twice | 1.5 days |
| `check_units` | detect a column mixing kg/lb (bimodal distribution) or mismatched units across tables for the same measure | 1.5 days |
| `check_sensitivity` | regex/classifier for SSN, DoD-ID, CUI markings, weighted toward free-text columns | 1 day |
| `check_undocumented_join` | MinHash resemblance between columns across tables to suggest joins nobody declared | 2 days |
| `check_stale_data` | newest row per partition vs a freshness window (catches Fuel Cell stale since March) | 1 day |
| harden `check_key_conformity` | seed a length mismatch (pad one NSN to 14) and prove raw-vs-normalized match rate flags it | 1 day |

## Step 3 — Scoring & ranked output (½ day)
The starter uses a flat penalty. Improve it: weight by severity **and** by how much of
the data the issue touches (an orphan-join on 8% of rows should hurt more than a
constant column). Keep the 0–100 dial mapping and the HIGH/MED/LOW counters.

## Step 4 — Tie findings to a narrative (½ day, the part that sells)
Write a small function that assembles related findings into one sentence, e.g.:
*"Your headcount is wrong because shop_id is orphaned for 8% of the roster (S99 doesn't
exist in shops), and one NSN is recorded under 5 names."* Part B of the feature-set doc
shows the target sentence. This is what the buyer actually remembers.

---

## Testing discipline (do this throughout, not at the end)
- Every check gets at least one assertion in `test_dhs.py` against the seeded data.
- Two kinds of test: **it finds the real issue** (recall) and **it doesn't flag clean
  columns** (precision). The starter run has obvious false positives — your tuning
  tests lock in the fixes.
- `pytest test_dhs.py -v` must stay green before any merge.

---

## Suggested 4-intern, ~2-week plan

**Week 1 — foundation + easy/medium**
- *Day 1:* everyone does Step 0–1 together (setup, read the code, run the oracle).
- *Days 2–5:* split the work —
  - Intern A: Tier-1 checks + scoring (Step 3)
  - Intern B: `validity` + `cross_field`
  - Intern C: tune `dirty_categorical` + `fake_key`, then start `reference_standardization`
  - Intern D: `near_duplicate` + `stale_data`
- Daily: 15-min standup, all tests green at end of day.

**Week 2 — hard checks + polish**
- Intern A: `units` + harden `key_conformity`
- Intern B: `sensitivity`
- Intern C: finish `reference_standardization` (defense masks)
- Intern D: `undocumented_join`
- Last 1–2 days: the narrative assembler (Step 4), precision tuning, README, demo the
  full scan on both HazMat **and** Education data (same checks, different domain).

**Rough effort math:** foundation ≈ 2 person-days; ~18 checks total but ~11 already
done, so ~10–12 person-days of new check work + ~2 days scoring/narrative/testing ≈
**~14–16 person-days**. Across 4 interns that's about **2 calendar weeks** including
ramp-up. A senior could do the same in ~6–7 focused days.

---

## Definition of done
1. `python dhs.py --data ./data` returns a sensible score and a ranked list with **no
   obvious false positives** (IDs/dates not flagged as dirty categories).
2. `pytest test_dhs.py -v` green, including the precision tests you add.
3. The same code runs on the **Education** dataset and finds its analogous issues
   (Georgia spelled 3 ways, duplicate license_id, Carver missing) — proof the checks
   are domain-agnostic.
4. One narrative sentence is generated from the findings.
5. Output shape matches what the UI needs: `{score, counts{HIGH,MED,LOW}, findings[]}`.
