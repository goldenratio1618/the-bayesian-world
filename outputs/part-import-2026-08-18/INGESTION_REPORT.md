# Part ingestion report — 2026-08-18

This report is the durable audit for the 20-part Luna import inventory in
`agent_jobs.json`. The classifier ran on every part. Modeling was admitted only
for classifications whose required physics is currently implemented. Electrical
fixed resistors were admitted; thermistors were excluded because their existing
interface requires thermal physics.

## Outcome

- New parts classified: **20**
- Classification proposals accepted by the deterministic semantic gate: **20**
- Parts admitted to modeling: **10**
- Parts with host-validated models promoted into `model_catalog`: **10**
- Parts excluded from modeling by the domain gate: **10**
- Additional parts needed to reach 10 modeled: **0** (the first 20 sufficed)
- Final official catalog model-instance count: **27**
- Task spend: **$2.24703214**
- Lifetime ledger after ingestion: **$7.31624566 spent / $92.68375434 remaining**

The spend includes `$0.01226126` for 20 classifications, all successful model
runs, one `$0.60` pre-Luna runner/configuration failure, and one `$0.60` model
dispatch whose locally validated copied PMDL was rejected by the host parser.
The ledger conservatively charged both failed dispatches at their full reserved
amount.

## Modeled and promoted parts

“Luna validator calls” counts every recorded invocation of the isolated model
validator across all dispatches needed to obtain the final host-valid model.
“Failed calls” counts validator responses that reported invalid. A locally valid
call is observability only: the host independently parses and validates the
materialized bytes, generates `README.md`, validates again, and revalidates a
private snapshot during promotion.

| Import target | Catalog part id | Model name | PMDL id | Dispatches | Luna validator calls | Failed calls | Pre-final host/runner rejection |
|---|---|---|---|---:|---:|---:|---|
| `yageo_rc0603_10r` | `yageo_rc0603_10r` | Ideal uncertain resistor | `electrical.resistor.ideal` | 2 | 1 | 0 | Archived CLI rejected current config before Luna validation |
| `yageo_rc0603_47r` | `yageo-rc0603-47r` | Ideal uncertain resistor | `electrical.resistor.ideal` | 1 | 1 | 0 | None |
| `yageo_rc0603_100r` | `yageo_rc0603_100r` | Ideal uncertain resistor | `electrical.resistor.ideal` | 1 | 1 | 0 | None; typo-only shared-interface overwrite excluded during curation |
| `yageo_rc0603_220r` | `yageo-rc0603-220r` | Fixed resistor with fitted parasitics | `electrical.resistor.fixed_parasitic_rlc` | 1 | 4 | 3 | None |
| `yageo_rc0603_1k` | `yageo_rc0603_1k` | Ideal uncertain resistor | `electrical.resistor.ideal` | 1 | 1 | 0 | None |
| `yageo_rc0603_4k7` | `yageo_rc0603_4k7` | Ideal uncertain resistor | `electrical.resistor.ideal` | 1 | 1 | 0 | None |
| `yageo_rc0603_10k` | `yageo-rc0603-10k` | Ideal uncertain resistor | `electrical.resistor.ideal` | 2 | 4 | 2 | First dispatch's copied PMDL had an invalid control character |
| `yageo_rc0603_47k` | `yageo_rc0603_47k` | Ideal uncertain resistor | `electrical.resistor.ideal` | 1 | 1 | 0 | None |
| `yageo_rc0603_100k` | `yageo_rc0603_100k` | Ideal uncertain resistor | `electrical.resistor.ideal` | 1 | 1 | 0 | None |
| `yageo_rc0603_1m` | `yageo_rc0603_1m` | Thick-film resistor with lumped parasitics | `electrical.resistor.thick_film_parasitic` | 1 | 7 | 6 | None; exceeds the five-call prompt/support-material review threshold |

Every promoted part directory contains a deterministic, standalone `README.md`
generated from the final catalog overlay. Each README names every `vN.model`
hypothesis, displays its exact PMDL identity and equations, describes all parent
interfaces, reports notes/metadata/provenance, and separates executable
constraints from prose desires.

## Classified but excluded from modeling

All ten parts below were classified to `thermoelectric → thermistor`, with
required domains `thermoelectric`, `electrical`, and `thermal`. Their classifier
outputs passed the semantic gate, but the modeling gate excluded them because
thermal physics is outside the currently implemented model domains.

| Import target | Manufacturer part number | Classification | Modeling result |
|---|---|---|---|
| `murata_ncp18_100r` | `NCP18XF101J03RB` | Thermistor | Skipped: thermal physics required |
| `murata_ncp18_150r` | `NCP18XF151J03RB` | Thermistor | Skipped: thermal physics required |
| `murata_ncp18_220r` | `NCP18XM221J03RB` | Thermistor | Skipped: thermal physics required |
| `murata_ncp18_330r` | `NCP18XM331J03RB` | Thermistor | Skipped: thermal physics required |
| `murata_ncp18_470r` | `NCP18XQ471J03RB` | Thermistor | Skipped: thermal physics required |
| `murata_ncp18_680r` | `NCP18XQ681J03RB` | Thermistor | Skipped: thermal physics required |
| `murata_ncp18_1k` | `NCP18XQ102J03RB` | Thermistor | Skipped: thermal physics required |
| `murata_ncp18_1k5` | `NCP18XW152J03RB` | Thermistor | Skipped: thermal physics required |
| `murata_ncp18_2k2` | `NCP18XW222J03RB` | Thermistor | Skipped: thermal physics required |
| `murata_ncp18_3k3` | `NCP18XW332J03RB` | Thermistor | Skipped: thermal physics required |

## Sources and qualification

- Yageo part records link to the manufacturer's part-number search for each
  exact RC0603 orderable code.
- Murata thermistor records link to the manufacturer's NCP18 series data sheet.
- Vendor dimensions and nominal values are preserved as provenance, not treated
  as empirical qualification. Per-unit values, parasitics, and applicability
  still require measurement or calibration for a qualified use.

## Verification performed

- Full interface, PMDL, part-instance, exact-hash, parameter, and catalog load.
- Deterministic detailed-shape generation and `--check` with zero pending changes.
- Byte-for-byte README regeneration check for all ten imported parts.
- Canonical scanner `contraption validate` closure: valid.
- Full project test suite after shape generation.
