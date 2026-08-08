# World Model Catalog Mining

Reproducibility tooling for the immutable raw corpus at `D:\WorldModelCatalogs`.
This directory deliberately contains the declarative source manifest alongside the
PowerShell acquisition and verification scripts.  It does **not** contain the
downloaded source data.

## Contents

- `catalog_manifest.csv` — versioned, source-level acquisition decisions and
  integrity expectations.
- `bootstrap_catalogs.ps1` — resumably acquires the selected `core` or
  `assets` sources into a guarded dedicated directory on `D:` or `F:`.
- `verify_catalogs.ps1` — verifies a completed catalog's raw artifacts,
  manifests, archive structures, and Git snapshots without changing them.

## Run on Windows

From the repository root in Windows PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\src\contraption\catalog_mining\bootstrap_catalogs.ps1 `
  -Root D:\WorldModelCatalogs -Profile CoreAndAssets -MinimumFreeGiB 12
```

Then verify the downloaded corpus:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\src\contraption\catalog_mining\verify_catalogs.ps1 `
  -Root D:\WorldModelCatalogs -Profile CoreAndAssets -DeepArchiveValidation
```

The bootstrap script fails closed outside a dedicated `D:` or `F:` directory,
never overwrites completed artifacts, and retains immutable manifest snapshots
and run reports under the target catalog root.  The secure default keeps Windows
certificate-revocation checks enabled.  Use
`-AllowOfflineRevocationFallback` only after confirming the documented
`CRYPT_E_REVOCATION_OFFLINE` host condition; its use is recorded in the run
context.

## Extending the corpus

Add a source as a new manifest record first, including the source URL/version,
method, destination relative path, selection profile, license posture, and
upstream integrity contract where available.  Treat a manifest change as a new
corpus version: review rights and storage cost, run a narrow `-IncludeIds`
bootstrap, then run verification before promoting the source to the trusted raw
layer.  Do not alter previously acquired raw artifacts in place.

See `docs/03_catalog_corpus.md` for the source rationale, rights posture, and
ingestion design.
