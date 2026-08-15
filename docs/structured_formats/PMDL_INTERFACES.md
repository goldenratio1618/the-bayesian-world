# PMDL interface declarations (`pmdl-interface-1`)

An `interface.pmdl` file is a strict JSON declaration for one filesystem catalog
layer. It is an abstract contract, not an executable physical model. The
authoritative parser is `contraption.catalog.interfaces`.

## Filesystem meaning

| Catalog path | `kind` | Meaning |
|---|---|---|
| `model_catalog/<domain>/interface.pmdl` | `domain` | physical domain |
| `model_catalog/<domain>/<category>/interface.pmdl` | `category` | reusable component contract |
| `model_catalog/<domain>/<category>/<device>/interface.pmdl` | `device` | more-specific device family |

There are exactly three levels. Interface ids are globally unique. Category and
device ids/names may not create punctuation- or case-only semantic duplicates.
A concrete `pmdl-1` file at a category or device layer names the implemented
interface through its `implements` field.

## Common header

All kinds require:

| Field | Type | Meaning |
|---|---|---|
| `format` | string | exactly `pmdl-interface-1` |
| `kind` | string | exactly `domain`, `category`, or `device` as appropriate |
| `abstract` | boolean | exactly `true` |
| `id` | identifier | stable machine identity |
| `name` | non-empty string | human display name |
| `version` | non-empty string | interface version |
| `description` | string, optional | explanatory text; default empty |

Identifiers match `^[A-Za-z][A-Za-z0-9_.-]*$`. Port names use the PMDL symbol
grammar `^[A-Za-z][A-Za-z0-9_]*$`.

## Domain interface

Required fields beyond the common header:

| Field | Type | Meaning |
|---|---|---|
| `requires_physics` | identifier array | physics engines needed to implement this domain |
| `allowed_port_domains` | identifier array | domains that a model in this domain may expose |

Every `allowed_port_domains` entry must name an existing domain interface.

Example:

~~~json
{
  "format": "pmdl-interface-1",
  "kind": "domain",
  "abstract": true,
  "id": "optical",
  "name": "Optical",
  "version": "1.0.0",
  "requires_physics": ["optical"],
  "allowed_port_domains": ["optical", "electrical", "signal"],
  "description": "Light sources, transport, sensing, and optical observations."
}
~~~

## Category interface

Required `domains` declares every physical domain implemented by the component.
Optional arrays default empty.

| Field | Type | Meaning |
|---|---|---|
| `domains` | identifier array | implemented physical domains |
| `required_power_ports` | power-port-interface array | minimum acausal power contract |
| `required_signal_ports` | signal-port-interface array | minimum scalar directed-signal contract |
| `ideal_models` | model-id array | ideal PMDL implementations defined at this category level |
| `constraints` | string array | human-readable physical contract constraints |

A power-port-interface has exactly `name`, `domain`, `effort_unit`, and
`flow_unit`. A signal-port-interface has `name` and `direction`, plus optional
`unit` defaulting to `1`. Direction is `input` or `output`.

Example:

~~~json
{
  "format": "pmdl-interface-1",
  "kind": "category",
  "abstract": true,
  "id": "camera",
  "name": "Camera",
  "version": "1.0.0",
  "domains": ["optical", "electrical"],
  "required_power_ports": [
    {
      "name": "supply",
      "domain": "electrical",
      "effort_unit": "V",
      "flow_unit": "A"
    }
  ],
  "required_signal_ports": [],
  "ideal_models": ["optical.camera.ideal"],
  "constraints": ["Exposure duration must be positive"],
  "description": "An optical observation sensor."
}
~~~

Typed non-scalar PMDL artifact ports are declared by concrete PMDL models. The
current interface format constrains power and scalar signal ports only; do not
invent `required_artifact_ports` unless the parser is deliberately versioned.

## Device interface

A device narrows one category and must explain its additional model specificity.

| Field | Type | Meaning |
|---|---|---|
| `parent` | category identifier | existing category contract |
| `model_specificity` | string | at least 12 non-whitespace characters explaining the more-specific physics |
| `changes_contract` | boolean, optional | whether the device adds required ports; default `false` |
| `required_power_ports` | array, optional | additional/overriding required power ports |
| `required_signal_ports` | array, optional | additional/overriding scalar signal ports |
| `models` | model-id array, optional | PMDL implementations defined at this device level |
| `constraints` | string array, optional | device-specific physical constraints |

If either required-port array is nonempty, `changes_contract` must be true.

## Catalog-wide validation

The loader also requires:

- each top-level directory name to match its domain id;
- each category/device location to match the declared ancestry;
- every category `domains` entry to name an existing domain;
- every device `parent` to name an existing category;
- concrete model ports to cover the required interface ports with compatible
  domains, directions, and units;
- `ideal_models` and `models` entries to resolve to concrete PMDL identities at
  the appropriate layer.

Unknown fields and duplicate JSON keys are errors. Interface `constraints` are
descriptive requirements; executable physics still belongs in concrete PMDL
relations and machine-checkable properties.
