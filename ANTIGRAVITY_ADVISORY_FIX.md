# ANTIGRAVITY HANDOFF: advisory N/A fix + horizontal concept chart + dashboard safety fixes

Working root: C:\Users\amit0\OneDrive\Desktop\plant_disease_detection. Set KERAS_HOME as in PROGRESS.md before TensorFlow runs.
Scope: app/dashboard.py, scripts/dashboard_utils.py (only if needed), one new test file, one AppTest addition. No retraining.
Do NOT edit: advisory/advisory_table.csv, advisory/safeguards_report.json, advisory/gate_config.json, best_model.keras, manifest.csv, eval_report/*.

## Root cause (verified by reading the files)
Section "4. Agronomic Advisory" in app/dashboard.py reads columns `condition_name`, `symptoms`, `organic_treatment`, `chemical_treatment`.
advisory/advisory_table.csv has NONE of them (its columns: class_name, crop, condition, is_healthy, category, cause, what_to_do_now,
prevention, when_to_seek_help, source_name, source_url, source_status). So every class printed "N/A".
Fix = supply the data and read the real columns. The owner has already added `advisory/advisory_details.csv`
(columns: class_name, symptoms, organic_treatment, chemical_treatment, detail_status; 15 rows, one per class).

## Task 1: data check (do first, report result)
- New test `tests/test_advisory_details.py`:
  a. details file has exactly the 15 names in `data_pipeline.SELECTED_CLASSES`, unique, same names as advisory_table.csv class_name.
  b. symptoms / organic_treatment / chemical_treatment non-empty, at least 40 characters, never "N/A", "nan", "TBD" (case-insensitive).
  c. organic_treatment and chemical_treatment contain no dose patterns: regex `\d+(\.\d+)?\s*(ml|l|g|kg|oz|lb|ppm|%|tbsp|tsp)\b`.
  d. every non-healthy class's chemical_treatment contains the word "label"; every healthy class's chemical_treatment starts with "None needed".
  e. CSV parses with pandas without warnings and has no duplicate columns.
- If the CSV fails to parse or a name does not match, fix ONLY `advisory/advisory_details.csv` (quote fields with commas) and report what you changed.

## Task 2: dashboard advisory section
- Load advisory_table.csv and advisory_details.csv, merge on class_name (left join; assert no missing detail rows; log a warning and show
  "No advisory entry for this class" if one is missing). Do not edit advisory_table.csv. Prefer a helper in dashboard_utils.py with a unit test.
- Render, in this order, using real columns: Condition (`condition`), Symptoms (`symptoms`), Likely cause (`cause`),
  What to do now (`what_to_do_now`), Organic and cultural control (`organic_treatment`), Chemical control (`chemical_treatment`),
  Prevention (`prevention`), When to seek help (`when_to_seek_help`), then a source line (`source_name`, `source_url`, `source_status`).
- Under Chemical control add this caption: "General guidance only, no doses. Use only products registered for this crop in your country,
  follow the label (waiting period, protective equipment), and confirm with your local agriculture extension office."
- Show the advisory ONLY when status is "Confirmed" (quality gate passed AND confidence >= threshold).
  Uncertain: show "Not sure. No advice shown; see the closest matches above." Quality Rejected: show the gate reason, no advice.
- Delete the fallback text "Healthy plant or standard monitoring recommended...". A class with no row must say "No advisory entry".
- Healthy classes: show "Looks healthy" and the details row (symptoms "No visible disease symptoms...", chemical "None needed...").
- There must be no "N/A" string anywhere in the advisory output for any of the 15 classes.

## Task 3: concept reasoning chart horizontal
- Replace the vertical `st.bar_chart` with a horizontal one: `st.bar_chart(df, x="Concept", y="Score (%)", horizontal=True, sort=False,
  y_label="Score (%)", x_label="", height=260)`. Convert scores to percent (x100). Keep order: Is Healthy, Has Spots, Is Blighted, Complex Pathogen.
  (st.bar_chart has `horizontal` and `sort` in the pinned Streamlit; confirm with inspect.signature before use.)
- Keep the caption "These concepts are class-derived auxiliary labels, not independently annotated."
- Also print the four percentages as text under the chart (st.caption), because chart labels are small on phones.

## Task 4: safety and honesty fixes in the same file
1. Add an always-visible banner under the title (st.warning): "Trained on clean lab-style photos. On real field photos (PlantDoc) accuracy was X%
   and the confidence score does not reliably flag mistakes. Decision support only; confirm with an agronomist." X must be read from
   advisory/field_gap_report.json (`field_target_metrics.overall_accuracy`); if the file is missing, omit the number. No hard-coded figures.
2. Tab "Lab Performance": remove the hard-coded "92.8% ... 5-image gap ... cloud sync" markdown and the hard-coded metric("92.8%").
   Read accuracy from eval_report/eval_test.json. Remove fabricated defaults in `.get('threshold', 0.30)`, `.get('test_ece', 0.034)`,
   `.get('clean_test_answered_share', 0.98)`: show "n/a" when a key is missing.
3. Add one caption in that tab: "Threshold and gate values were calibrated on a partial copy of the data; re-calibration pending."
4. The Grad-CAM block recomputes on every slider move: wrap the Grad-CAM computation in `st.cache_data` keyed by (image hash, class index)
   with `_model` excluded from hashing, and run model calls under a `threading.Lock`. Skip if it takes more than one small edit.
5. `use_container_width=True` still works on the pinned Streamlit 1.67 (checked). Leave it unless a deprecation warning fails a test.

## Task 5: tests
- AppTest: select the sample photo path (radio "Pick Sample Preset", pick a sample) and assert: no exception; if status is Confirmed, the page
  text contains "Symptoms", "Organic", "Chemical" and does not contain "N/A"; the concept chart is present.
- Unit test for the merge helper (all 15 classes merge, no N/A, missing class handled).
- Run the FULL suite with KERAS_HOME set, and a fresh-clone run. Expect 0 failures.

## Report (max 10 lines)
Task 1 result (pass or what you fixed in the CSV), pytest counts, list of changed files, a screenshot-free description of what the
Classify tab now shows for one disease sample and one healthy sample, anything unexpected.
Commit with a short message on the current branch. Do not push.
