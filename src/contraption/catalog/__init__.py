"""Filesystem-backed model interfaces and part instantiations."""

from .instantiations import validate_optical_sensors
from .procurement import (
    ProcurementDocumentSpec,
    ProcurementEvidenceSpec,
    ProcurementIdentifierSpec,
    ProcurementLifecycleSpec,
    ProcurementOfferSpec,
    ProcurementProvisionSpec,
    ProcurementRecord,
    ProcurementRegistry,
    ProcurementSpecError,
)


__all__ = [
    "ProcurementDocumentSpec",
    "ProcurementEvidenceSpec",
    "ProcurementIdentifierSpec",
    "ProcurementLifecycleSpec",
    "ProcurementOfferSpec",
    "ProcurementProvisionSpec",
    "ProcurementRecord",
    "ProcurementRegistry",
    "ProcurementSpecError",
    "validate_optical_sensors",
]
