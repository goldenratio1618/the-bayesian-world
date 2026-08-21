# Procurement records (`procurement-record-1`)

Procurement records hold product identity, documents, lifecycle observations,
supplier offers, and exact mappings to static parts. They are deliberately
separate from `static.part`: availability, price, purchase URLs, and lifecycle
can change without changing physical identity or invalidating a physical
assembly hash. The authoritative implementation is
`contraption.catalog.procurement`.

## Placement and top-level fields

A record lives in a part instantiation directory, beside `static.part` and
`v1.model` when those physical artifacts exist:

`model_catalog/<domain>/<category>/instantiations/<part>/<record-id>.procurement`

For a multi-part kit, the record lives beside its first, canonical provision
while `provides` may bind every exact kit part and quantity. An evidenced
identity deferred because its physics is not implemented may occupy a reviewed,
planned instantiation directory without `static.part` or `v1.model`; it must
retain `provides: []`. For example, an NTC thermistor identity may live below
`thermoelectric/thermistors/instantiations/` while thermal modeling remains
deferred.

Placement is organizational only. It never asserts a purchasable-to-physical
mapping: the `provides` array is the sole binding authority. Importers must not
choose a directory heuristically. Bound records use the first exact provision;
unbound records require an explicit, reviewed location.

The filename stem must equal `id`. Records outside a direct child of an
`instantiations` directory, symlinks, duplicate ids/JSON keys, and unknown
fields are invalid. Every top-level field is required:

| Field | Meaning |
|---|---|
| `format` | exactly `procurement-record-1` |
| `id` | stable record identifier |
| `version` | nonempty procurement-record version |
| `manufacturer` | nonempty string or null when the evidence does not identify one |
| `identifiers` | nonempty evidenced identity array |
| `documents` | document/link array; may be empty |
| `offers` | time-stamped supplier-offer array; may be empty |
| `lifecycle` | typed lifecycle observation |
| `provides` | exact static-part provision array; may be empty |
| `evidence` | nonempty source-evidence array |

Absence is preserved. If source data has no purchasing or identity facts, no
record is created. A record requires at least one explicit identifier and one
source-evidence item; parsers must not invent a manufacturer, MPN, supplier,
URL, offer, price, lifecycle state, or part mapping. An evidenced product with
no safely resolved catalog part remains useful with `provides: []`.

## Identifiers, documents, and evidence

An identifier has required `scheme` and `value`, plus optional `issuer` and
`scheme_uri`. Supported schemes are `manufacturer_part_number`,
`manufacturer_item_number`, `supplier_sku`, `product_name`, `gtin`, `upc`,
`ean`, `standard_designation`, and `extension`.
Manufacturer/supplier identifiers require `issuer`; GTIN-family values require
a valid length and check digit. A rare identifier namespace uses
`scheme: extension` and a mandatory authoritative HTTP(S) `scheme_uri`; it is
not mislabeled as a common scheme.

A document requires `kind` and absolute HTTP(S) `url`. Kinds are `datasheet`,
`product_page`, `purchase_page`, `drawing`, `certificate`,
`lifecycle_notice`, and `other`. Optional fields retain `media_type`, source and
extracted-text SHA-256, retrieval timestamp, title, and page count. A document
URL is evidence, not automatically an offer.

Each evidence item requires human/audit `source` and prefixed `sha256`; optional
`locator` pinpoints the source field, page, or structured-record path. The
deterministic importer hashes input JSON and vetted PDF/ECAD extractions before
creating records. A URL or part-like substring is never promoted to an
identifier without source-format semantics supporting that claim.

## Lifecycle and offers

Lifecycle status is one of `active`, `not_recommended_for_new_design`,
`last_time_buy`, `obsolete`, or `unknown`. A known state requires both
timezone-aware `observed_at` and `source_url`; `unknown` may omit them.

An offer is a volatile observation with required `supplier`,
`supplier_part_number`, `purchase_url`, timezone-aware `observed_at`, and
`availability`. Availability is `in_stock`, `backorder`, `preorder`,
`out_of_stock`, `discontinued`, or `unknown`. `currency` and positive
`unit_price` are supplied together; optional minimum order quantity is a
positive integer. Old observations remain old evidence rather than silently
becoming current prices.

## Exact static-part provisions

A provision has exactly:

~~~json
{
  "part": "C1210C476K8RAC",
  "version": "1.0.0",
  "static_sha256": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "quantity": 1
}
~~~

`part` names one `static-part-2`; `version` and `static_sha256` must match its
current canonical record exactly. One procurement record may provide several
different parts (for example, an evidenced kit), but may list each part only
once. `quantity` expresses how many copies of that static part the purchased
item provides. A stale hash, unknown part, or ambiguous kit-to-part mapping is
rejected rather than repaired heuristically.

## Example without an asserted catalog mapping

~~~json
{
  "format": "procurement-record-1",
  "id": "yageo_rc0603fr_0710kl",
  "version": "1.0.0",
  "manufacturer": "Yageo",
  "identifiers": [
    {
      "scheme": "manufacturer_part_number",
      "value": "RC0603FR-0710KL",
      "issuer": "Yageo"
    }
  ],
  "documents": [
    {
      "kind": "product_page",
      "url": "https://www.yageo.com/en/ProductSearch/PartNumberSearch?partNo=RC0603FR-0710KL"
    }
  ],
  "offers": [],
  "lifecycle": {"status": "unknown"},
  "provides": [],
  "evidence": [
    {
      "source": "outputs/part-import-2026-08-18/component_inputs/yageo_rc0603_10k.json",
      "sha256": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "locator": "$.manufacturer and $.product"
    }
  ]
}
~~~

`provides` is empty in this example on purpose: identity evidence alone does
not claim that a particular catalog static part is exactly the purchased item.

## Hash separation

Three digests serve different purposes:

- `StaticPartSpec.sha256` hashes canonical `static-part-2` physical content and
  excludes all procurement state.
- `ProcurementRecord.sha256` hashes one canonical procurement record;
  `ProcurementRegistry.sha256` hashes all records sorted by id.
- The build plan records `assembly_sha256` and `procurement_sha256` separately.

Updating a price, supplier, purchase URL, lifecycle observation, or unbound
identity record changes the procurement closure but not physical assembly
identity. Changing a provision's static-part target requires the exact new
physical version/hash and therefore fails closed until deliberately updated.
