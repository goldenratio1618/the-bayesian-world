# Verified world-model startup research

Prepared 2026-08-01.

## Current strategic framing

The current hypothesis is consumer-first:

> Build a micro-appliance foundry that lets a nonexpert submit a bounded physical chore and receive a tested, repairable custom device with clear operating limits. The internal Bayesian engineering harness should make each related device cheaper and more predictable as its model and evidence library grows.

The company should initially target high-intent consumers and microbusiness owners whose repetitive, low-risk task cannot justify a conventional engineering team. Humans assemble and release the first devices; self-service design and automated assembly come later.

The first recommended engineering proof is a guarded tabletop trading-card workstation family. Device A feeds, scans, rejects and sorts; it is an internal platform, not a price-competitive product claim. A held-out Device B targets automatic sleeving, with a pre-registered mechanically meaningful fallback. The second device must require substantially less expert work and fewer calibration experiments through unchanged exact-hash dependencies and explicit delta qualification. If the differentiated capability cannot earn paid demand at its real price, retain the apparatus as a benchmark and choose another launch niche. A single hero prototype does not validate the compounding thesis.

## Current deliverables

- [Competitive landscape v2: consumer custom devices](./01_competitive_landscape_v2_consumer.md)
- [Implementation plan v2: consumer engineering harness](./04_implementation_plan_v2_consumer.md)
- [Probabilistic programming framework assessment](./02_probabilistic_frameworks.md)
- [Machine-readable PPL compatibility smoke results](./ppl_smoke_results.json)
- [Open-catalog corpus and licensing plan](./03_catalog_corpus.md)
- [Catalog artifact/source manifest](../src/contraption/catalog_mining/catalog_manifest.csv)
- [Resumable catalog bootstrap script](../src/contraption/catalog_mining/bootstrap_catalogs.ps1)
- [SHA-256 catalog verification script](../src/contraption/catalog_mining/verify_catalogs.ps1)
- [Reproducible PPL smoke-test notes](../work/ppl-smoke/README.md)
- [Structured formats and DSL reference](./structured_formats/README.md)

## Preserved original framing

The earlier enterprise-facing work remains useful as an alternative strategy and as technical foundation:

- [Original competitive landscape](./01_competitive_landscape.md)
- [Original enterprise-first implementation plan](./04_implementation_plan.md)

The framework and catalog assessments remain current. In particular, use a backend-neutral typed IR and NumPyro/JAX as the first GPU-capable backend, hand-translate the few decision-critical models into Stan as an independent oracle, and keep a bounded 60–90-day GenJAX pilot off the device-delivery critical path. GenJAX's Apache-2.0 license is commercially suitable in principle; ecosystem maturity and version stability remain the important validation questions.

## Local corpus result

The `CoreAndAssets` corpus is downloaded and deep integrity-verified at [D:\WorldModelCatalogs](</D:/WorldModelCatalogs>): **61,372 files / 38.411 GiB**, **111/111 integrity checks passed**, five Git repositories clean and pinned, and zero incomplete `.part` files. F: was not used. See the [verification summary](</D:/WorldModelCatalogs/reports/verification-summary-latest.json>) and the [catalog report](./03_catalog_corpus.md).

## Decisions to make now

1. Pre-register the 16-week card-workstation build phase and week-22 field/economics decision, including autonomy, physical, market and kill criteria.
2. Run the same frozen brief through the closest prompt-to-hardware competitors and physically build the strongest accessible output.
3. Recruit a concentrated prosumer cohort, show a frozen intended price, use nonbinding reservations until terms/compliance review, and then require refundable deposits before deep family-specific engineering.
4. In the build phase, create only 15–20 task-relevant E0/E1 packages and calibrate 5–8 consequential interactions to E2; treat 30–50 packages as the month-six target and do not horizontally model the full downloaded catalog yet.
5. Require at least 70% weighted reuse of unchanged exact-hash qualified dependencies, explicit delta tests for every changed dependency, and at least 50% fewer expert hours on the held-out variant.
6. Engage consumer-product safety, product-liability and compliance counsel before taking deposits or placing any device offsite with a consumer, including a free beta.

## Principal risks

- Prompt-to-hardware, world-model and verification language is already crowded; generated artifacts alone are not a moat.
- Cheap design does not guarantee cheap hardware. Assembly, sourcing, packaging, compliance, support and returns can dominate.
- Unbounded customization destroys qualification and turns the company into a bespoke consultancy.
- A wrong explicit model can be confidently wrong; physical acceptance tests, model-discrepancy checks and independent safety controls are mandatory.
- A large library matters only if it improves held-out predictions, first-build success and marginal launch cost.
- Fixed products win once a niche becomes large, while general-purpose robots will become cheaper. The company must win on cost per completed chore and continuously reuse its platform across underserved tasks.
