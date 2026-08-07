# Open-catalog corpus

Research and bootstrap date: 2026-08-01.

## Purpose and status

The corpus is designed as a **provenance-first seed**, not an indiscriminate data dump. It spans general ontology, units, biology, chemistry, materials, electronics, multiphysics, mechanics/CAD, real-world 3D objects, food and geography. Every located source is classified as one of:

- `core`: relatively compact, high-value source selected for the core bootstrap profile.
- `assets`: larger but potentially commercially usable files selected for the full profile.
- `deferred`: useful and potentially usable, but too large or in need of per-model ingestion work.
- `quarantine`: ambiguous, mixed, noncommercial or record-specific rights; excluded from the trusted corpus pending review.

The authoritative source list is [catalog_manifest.csv](./catalog_manifest.csv). It contains 52 artifact/source decisions: 34 core entries, 6 larger asset entries, 6 deferred sources and 6 quarantine sources.

The bootstrap target is:

```text
D:\WorldModelCatalogs\
├── catalog_manifest.csv
├── raw\
│   └── <domain>\<source>\<version>\...
└── reports\
    ├── bootstrap.log
    ├── download-report-latest.csv/json
    ├── sha256-latest.csv
    ├── git-snapshots-latest.csv
    └── verification-summary-latest.json
```

Archives remain unextracted in the immutable raw layer. This avoids doubling disk usage and preserves the exact upstream artifact. Normalization should happen later into content-addressed staging and derived stores, never by modifying raw files.

All license postures in this report are preliminary research classifications, not legal conclusions. A repository's code license does not necessarily clear embedded datasets, parameter files, meshes, images, documentation or third-party contributions. Before any commercial redistribution, preserve immutable snapshots of the applicable license and terms at acquisition time, resolve per-record exceptions and attribution requirements, and obtain counsel review. Unclear cases belong in quarantine.

### Verified local snapshot

The `CoreAndAssets` profile completed on `D:\WorldModelCatalogs`; F: was not used. A full deep audit passed at **2026-08-01 12:51:09 UTC**:

- **40 selected manifest entries** produced 1,070 acquisition records.
- **61,372 completed raw files**, excluding Git internals, total **41,243,881,651 bytes (38.411 GiB)**.
- **5 Git repositories** passed exact acquisition-commit, origin, tag where applicable, clean-worktree, Git LFS, and `git fsck` checks.
- **111 integrity checks passed; 0 failed**. There were **0 incomplete `.part` files** and **0 Git/LFS/object failures**.
The content-addressed manifest SHA-256 is `a15ffef70edf5035e7a2cd56f2b77c546a1c7bb60a241b32e83d02fad32b2e4a`; the authenticated acquisition CSV SHA-256 is `7e8f05bb581600f275832081792c280aae9f10fc087ffd92a758440a28258e7e`.

Audit artifacts: [verification summary](</D:/WorldModelCatalogs/reports/verification-summary-latest.json>) · [SHA-256 inventory](</D:/WorldModelCatalogs/reports/sha256-latest.csv>) · [Git snapshots](</D:/WorldModelCatalogs/reports/git-snapshots-latest.csv>) · [upstream/acquisition integrity checks](</D:/WorldModelCatalogs/reports/upstream-integrity-latest.csv>).

## Selected first-wave sources and bootstrap status

The tables below describe the selected first wave. The verified state above applies to the selected `core` and `assets` entries; `deferred` and `quarantine` entries were not acquired into the trusted corpus. Inclusion still does not imply scientific fitness for a particular use or final commercial clearance.

### Ontology and physical semantics

| Source | Contents | License posture |
|---|---|---|
| Open English WordNet 2025+ | Lexical concepts and relations | CC BY 4.0; attribution layer |
| YAGO 4.6 | Schema, taxonomy, metadata, sample, labels and facts | CC BY 4.0; attribution layer |
| QUDT 3.2.1 | Quantity kinds, units, dimensions and conversions | CC BY 4.0; attribution layer |
| Gene Ontology | Biological process, component and function DAG | CC BY 4.0; attribution layer |
| NCBI Taxonomy | Stable organism identifiers, ranks, names and parent relations | Keep under NCBI/source review rather than asserting blanket public-domain status |

[Open English WordNet downloads](https://en-word.net/downloads) · [YAGO 4.6](https://yago-knowledge.org/downloads/yago-4-6) · [QUDT repository](https://github.com/qudt/qudt-public-repo) · [Gene Ontology downloads](https://geneontology.org/docs/download-ontology/) · [NCBI taxonomy documentation](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/data-processing/taxonomy-processing/taxonomy/)

These are seeds, not a universal canonical tree. Imported classifications should remain versioned named graphs with mappings, not be flattened destructively into one proprietary hierarchy.

### Chemistry, biology and quantitative models

| Source | Contents | License posture |
|---|---|---|
| ChEBI | Chemical ontology plus full SDF structures/properties | CC BY 4.0 |
| Rhea release 141 | Biochemical reactions, participants and cross-references | CC BY 4.0 |
| BioModels r31 SBML | Published quantitative biological models | CC0 |
| UniProtKB Swiss-Prot 2026_02 | Reviewed protein records and functional annotations | CC BY 4.0 with normal third-party/patent caveats |
| RCSB PDB holdings + component data | Structure IDs and compact chemical-component atoms/bonds | CC0 |
| Cantera 3.2.0 | Reaction mechanisms, thermodynamics, transport and kinetics examples | BSD-3-Clause |

[ChEBI downloads](https://www.ebi.ac.uk/chebi/downloads/) · [Rhea](https://www.rhea-db.org/) · [BioModels terms](https://www.ebi.ac.uk/biomodels/termsofuse) · [UniProt license](https://www.uniprot.org/help/license) · [RCSB usage policy](https://www.rcsb.org/pages/usage-policy) · [Cantera YAML formats](https://www.cantera.org/stable/yaml/index.html)

BioModels is especially relevant because its curated models already encode explicit mechanistic equations and reproduce published simulations. ChEBI and Rhea help connect chemical identity to reaction semantics; RCSB connects chemical components to geometry.

### Materials and properties

| Source | Contents | License posture |
|---|---|---|
| JARVIS-DFT v11 | 75,000+ structures with calculated formation, electronic, elastic, dielectric and other properties | CC BY 4.0; Figshare publishes an MD5 for acquisition-time verification |
| Crystallography Open Database | More than 530,000 experimental crystal structures; current complete archive observed at 18,481,345,940 bytes (17.21 GiB) | CC0 |
| CoolProp 7.2.0 | Equations of state and thermophysical properties for 100+ fluids | MIT |
| ambientCG metadata | CC0 material, texture, HDRI and 3D-model catalog | CC0 |
| Poly Haven metadata | CC0 models, textures and HDRIs with dimensions/polycounts where supplied | Assets CC0; free API requires current attribution/User-Agent terms |

[JARVIS database documentation](https://pages.nist.gov/jarvis/databases/) · [JARVIS-DFT v11 record](https://figshare.com/articles/dataset/jdft_3d-7-7-2018_json/6815699) · [COD](https://www.crystallography.net/cod/) · [CoolProp](https://github.com/CoolProp/CoolProp) · [ambientCG API](https://docs.ambientcg.com/api/v2/) · [Poly Haven API terms](https://polyhaven.com/our-api)

JARVIS is property-rich and computational; COD is larger and experimentally structure-rich but property-light. They should link through composition/structure identifiers rather than be conflated.

### Electronics, mechanics and 3D geometry

| Source | Contents | License posture |
|---|---|---|
| KiCad 10.0.5 symbols/footprints/3D packages | Electronic identity, package geometry and thousands of STEP/VRML assets | CC BY-SA 4.0 plus KiCad exception; isolate collection redistribution obligations |
| LibrePCB Base | Components, packages, devices, symbols and categories | CC0 |
| FreeCAD Parts Library | Approximately 5 GB across mechanical, electrical, HVAC, robot, medical and industrial CAD | CC BY 3.0; preserve Git history for author attribution |
| Google Scanned Objects | Exactly 1,030 GoogleResearch-owned common-object ZIPs, advertised at 8,665,764,398 bytes; visual meshes/textures plus simulation-oriented mass, friction, inertia and collision geometry | CC BY 4.0 |
| Modelica Standard Library 4.1.0 | Electrical, mechanical, thermal, fluid and controls component models | BSD-3-Clause |
| PyBaMM 26.5.0 | Battery models and parameter sets | BSD-3-Clause |

[KiCad libraries and license](https://www.kicad.org/libraries/download/) · [LibrePCB Base](https://github.com/LibrePCB-Libraries/LibrePCB_Base.lplib) · [FreeCAD Parts Library](https://github.com/FreeCAD/FreeCAD-library) · [Google Scanned Objects](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/) · [Modelica Standard Library](https://github.com/modelica/ModelicaStandardLibrary) · [PyBaMM](https://github.com/pybamm-team/PyBaMM)

Google Scanned Objects is unusually useful because it contains simulation metadata rather than only appearance. KiCad gives exact component geometry and identifiers. FreeCAD broadens mechanical and industrial coverage. None of these should be mistaken for validated behavioral models: geometry, identity, material assignment, equations and empirical evidence are separate linked artifacts.

### Behavioral-parameter gap analysis

KiCad and FreeCAD primarily contribute identity, connectivity/package information and geometry. Except for limited metadata or separately linked models, they do not supply the validated behavioral parameters needed to predict how a component responds across operating conditions. Modelica, PyBaMM, CoolProp, Cantera, JARVIS and Google Scanned Objects provide valuable domain-specific footholds, but they do not close the following broad gaps:

- **Electronics:** the current wave lacks a broad, commercially safe catalog of manufacturer-qualified SPICE models, S-parameters, IBIS models, tolerance distributions, temperature coefficients, limits and lifecycle/version metadata.
- **Mechanical engineering:** it lacks a comprehensive engineering material-property catalog covering temperature- and rate-dependent constitutive behavior, fatigue, fracture, creep, wear, friction/contact and manufacturing-process effects with uncertainty and test provenance.
- **Chemical engineering:** it lacks a comprehensive physical-property catalog covering measured phase equilibria, equations of state, transport properties, reaction conditions, mixture behavior, safety limits and uncertainty across industrially relevant chemicals.

These are substantive corpus gaps, not fields that should be inferred from a mesh, a part number or a generic material label.

### Follow-on acquisition strategy

1. Choose a narrow commercial wedge and enumerate the exact behavioral evidence required for each model family before expanding source count. For an electronics wedge, that might be passive components, MOSFETs and sensors with SPICE/IBIS/S-parameter data across temperature and tolerance.
2. Acquire manufacturer- or customer-authorized datasheets, simulation models, characterization data and CAD through explicit download grants, partner feeds, distribution agreements or customer-provided materials. Do not assume that public web access grants extraction or redistribution rights.
3. Run a rights-reviewed extraction pipeline that snapshots the governing terms, records ownership and permitted uses at artifact and field level, retains the original source locator and hash, and links every extracted parameter to its page/table/model evidence. Use automated extraction only behind schema, unit, plausibility and cross-source checks, with human review for safety-critical or conflicting values.
4. Seek direct licensing or data partnerships for the missing mechanical and chemical property layers, prioritizing sources that include test method, conditions, uncertainty and revision history. Keep customer-confidential models in tenant-scoped stores and publish only rights-compatible derived artifacts.
5. Pilot deferred **SkyWater SKY130** as a bounded electronics process-model corpus after reviewing its repository, submodule and embedded-file rights; it is not a substitute for a broad component catalog. Pilot **OpenKIM** model by model after checking each model implementation, parameter file and dependency license; it is a strong template for executable, tested physical models but not blanket-cleared by the surrounding repository's code license.

### Food, environment and geography

| Source | Contents | License posture |
|---|---|---|
| USDA FoodData Central 2026-04-30 | Foods, ingredients, nutrients, branded products and analytical values | US federal/public-domain posture with source fields retained |
| GeoNames | More than 11 million named geographic features | CC BY 4.0 |
| Natural Earth | Global physical and cultural vector layers | Public domain |

[FoodData Central downloads](https://fdc.nal.usda.gov/download-datasets/) · [GeoNames dump](https://download.geonames.org/export/dump/) · [Natural Earth terms](https://www.naturalearthdata.com/about/terms-of-use/)

These are useful for real-world identity and environmental context, though not direct physical simulators.

## Deferred sources

| Source | Why valuable | Why deferred |
|---|---|---|
| SkyWater SKY130 PDK | SPICE primitives, device parameters, design rules, cells and timing | Approximately 7 GB with submodules; upstream calls it experimental and archived the GitHub repository in 2026 |
| OpenKIM | Versioned executable interatomic potentials, verification tests and predictions | Must ingest and approve each model’s license; best installed in WSL through OpenKIM tooling |
| OQMD 1.8 | Large DFT materials/property database | 21.1 GB compressed plus roughly 100 GB import space; use F: later |
| ChEMBL 37 | Molecules, targets, assays and bioactivity | CC BY-SA 3.0; valuable but needs an isolated share-alike layer |
| Wikidata | Broad CC0 entity/property graph | Current JSON dump is approximately 102 GB compressed; select a dated F:-drive snapshot only if needed |
| NOMAD | Millions of normalized materials calculations | Raw files exceed 100 TB; use a narrow federated API connector rather than mirroring |

The right principle is “federate the gigantic, snapshot the compact and high-value.”

## Quarantined sources

- **Amazon Berkeley Objects:** its official landing page currently says CC BY 4.0 while the AWS Open Data Registry says CC BY-NC 4.0. No commercial ingestion without written clarification from Amazon. [ABO landing page](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) · [AWS Registry entry](https://registry.opendata.aws/amazon-berkeley-objects/)
- **ABC CAD:** repository metadata says MIT, but underlying Onshape document rights can depend on creation date and per-document license state. Preserve and evaluate each `meta.yml` before use.
- **PubChem:** contributors set their own terms, including some NC/ND restrictions. Use source-filtered subsets rather than treating the whole dump as public domain. [PubChem download policy](https://pubchem.ncbi.nlm.nih.gov/docs/downloads)
- **Materials Project:** current release documentation identifies GNoME-originated structures with noncommercial terms. Filter at record level. [Materials Project database versions](https://docs.materialsproject.org/changes/database-versions)
- **Objaverse/Sketchfab:** index/code licensing does not replace the license of each underlying asset.
- **ShapeNetCore:** research/noncommercial terms make it inappropriate for the commercial core.

## Integrity and reproducibility

The supplied scripts are:

- [bootstrap_catalogs.ps1](../work/bootstrap_catalogs.ps1): validates a marked dedicated directory on D: or F:, enforces path containment and a configurable free-space reserve, downloads through conditionally resumable `.part` files, never overwrites a completed raw artifact, validates known sizes/checksums, captures HTTP validators, pins and checks Git snapshots, retains immutable manifest snapshots and writes per-run success/failure provenance.
- [verify_catalogs.ps1](../work/verify_catalogs.ps1): validates the root marker and exact manifest, hashes every completed raw file outside Git internals, checks acquisition SHA-256 baselines plus known upstream MD5/length contracts, archive signatures and ZIP central directories, verifies the exact Google owner/license/count/size/index contract, records Git origins/commits/tags/LFS/cleanliness and full FreeCAD history, and fails on missing artifacts, integrity errors or incomplete `.part` files. `-DeepArchiveValidation` additionally reads every ZIP member and complete compressed stream and asks `tar` to list complete tar-family archives.

Re-run the transfer safely with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\work\bootstrap_catalogs.ps1 `
  -Root D:\WorldModelCatalogs -Profile CoreAndAssets -MinimumFreeGiB 12
```

The secure default leaves Windows certificate-revocation checks enabled. On this host the upstreams triggered `CRYPT_E_REVOCATION_OFFLINE`; only after confirming that condition, the actual bootstrap used `-AllowOfflineRevocationFallback`. The flag maps to curl's Windows-specific `--ssl-no-revoke` fallback and is recorded in each run context; it does not disable certificate-chain or hostname validation.

Then verify with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\work\verify_catalogs.ps1 `
  -Root D:\WorldModelCatalogs -Profile CoreAndAssets -DeepArchiveValidation
```

## Ingestion design

Use a one-way pipeline:

```text
quarantine/review
      ↓
immutable raw snapshot + source/license record
      ↓
format-specific parser in a sandbox
      ↓
normalized staging records
      ↓
identity resolution and external-IRI mappings
      ↓
typed ontology/model/evidence graphs
      ↓
validation and human/license review
      ↓
published versioned registry artifact
```

Each normalized record should retain:

- Source identifier, release, URL, access time and local SHA-256.
- Exact license/SPDX expression, owner, required attribution and redistribution constraints.
- Parser/version and transform history.
- Original record locator and unmodified raw payload hash.
- Whether rights attach to database, record, code, parameters, mesh, image or documentation.
- Confidence and human-review state for cross-catalog mappings.

Keep permissive, attribution, share-alike and quarantine sources in separate named graphs/tables. A query may join them, but a commercial export should be able to select a rights-compatible closure automatically.

## What to ingest first

1. QUDT, Open English WordNet, YAGO schema/taxonomy and the small YAGO sample.
2. KiCad/LibrePCB component identity and port/package metadata for the mechatronics wedge.
3. Modelica electrical/rotational/thermal components as reference semantics.
4. JARVIS/COD/ChEBI identity and property schemas.
5. Google Scanned Objects metadata and simulation parameters, leaving meshes content-addressed.
6. BioModels/ChEBI/Rhea as the first cross-domain test of biological equation import.

The milestone should be 30–50 deeply mapped, evidence-backed model families in one vertical—not millions of superficially labeled files.
