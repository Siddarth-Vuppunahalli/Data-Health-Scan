# Intern Data-Integrity Tasks — Start Here

**Who:** Sid & Rhea · **When:** 2 weeks (aggressive), after the differentiator / Proving-Ground task · **Goal:** make the interpretDB Data Health Scan find more kinds of data problems, accurately.

## Do this in order
1. **Read** `Intern_DataIntegrity_Primer.docx` (~45 min) — what you're building, why it matters, how the code works, how you'll report. **Read before coding.**
2. **Open** `Intern_DataIntegrity_Sprint.xlsx` — your day-by-day plan, who owns what, acceptance criteria.
3. **Set up** the code:
   ```
   cd kit
   python -m venv .venv && source .venv/bin/activate
   pip install pandas numpy pytest
   python dhs.py --data ./data      # see the scan run
   pytest test_dhs.py -v            # see the 4 starter tests pass
   ```
4. **Build** checks test-first, following the sprint board. The detailed how-to per check is in `kit/DataHealthScan_Intern_Build_Guide.md`.
5. **Every Friday 4pm:** send your status using `Weekly_Status_Template.md`.

## What's in this folder
```
Intern_DataIntegrity_Primer.docx   READ FIRST — the learning doc
Intern_DataIntegrity_Sprint.xlsx   the 2-week plan (Sid/Rhea, daily, aggressive)
Weekly_Status_Template.md          copy each Friday, fill, send
README.md                          this file
kit/
  dhs.py                           the scanner — add your checks here
  test_dhs.py                      tests — write the test first
  DataHealthScan_Intern_Build_Guide.md   step-by-step per check + time estimates
  data/                            synthetic HazMat data with KNOWN seeded issues (your answer key)
```

Definition of done and the help-when-stuck rules are in the primer (sections 8 and 9).
