import re

from modal_shared.decisions import DECISION_OBJECTIVES
from modal_shared.stacks import NATIVE_FOUNDATION_STACKS
from overbae.core.errors import InputValidationError
from overbae.modal.model_registry import (
    get_all_models_by_backend,
    get_hf_base,
    get_model_config_any_backend,
    get_unsloth_image,
)
from overbae.models import FinetuningJob


def catalog(project):
    options = [
        {
            "id": model,
            "name": config.get("display", model),
            "kind": "foundation",
            "qualification": "catalog_eligible",
        }
        for model, config in get_all_models_by_backend("baseten", include_disabled=False).items()
        if get_unsloth_image(model) in NATIVE_FOUNDATION_STACKS
    ]
    for job in FinetuningJob.objects.filter(project=project, status="succeeded").order_by(
        "-created_at"
    ):
        if (job.hyperparameters or {}).get("objective") in DECISION_OBJECTIVES:
            options.append(
                {
                    "id": str(job.pk),
                    "name": job.name,
                    "kind": "trained",
                    "qualification": "checkpoint_available",
                }
            )
    return options


def qualify_participants(project, selected):
    if not isinstance(selected, list) or not 1 <= len(selected) <= 8:
        raise InputValidationError("Choose one to eight model participants")
    result = []
    seen = set()
    for participant in selected:
        if not isinstance(participant, dict) or set(participant) - {
            "key",
            "name",
            "kind",
            "job",
            "model",
        }:
            raise InputValidationError("Unknown participant fields")
        key = participant.get("key", "")
        if (
            not isinstance(key, str)
            or re.fullmatch(r"[a-z][a-z0-9-]{0,47}", key) is None
            or key == "prepare"
            or key in seen
        ):
            raise InputValidationError("Participant keys must be distinct lowercase names")
        seen.add(key)
        name = participant.get("name")
        kind = participant.get("kind")
        if (
            not isinstance(name, str)
            or not 1 <= len(name) <= 255
            or kind not in {"foundation", "trained"}
        ):
            raise InputValidationError("Choose a named foundation or trained participant")
        reference = {"key": key, "name": name, "kind": kind}
        if participant.get("job"):
            job = FinetuningJob.objects.filter(pk=participant["job"], project=project).first()
            if (
                job is None
                or (job.hyperparameters or {}).get("objective") not in DECISION_OBJECTIVES
            ):
                raise InputValidationError("Select a native decision training job in this project")
            if participant.get("model") not in {None, job.base_model}:
                raise InputValidationError(
                    "A trained reference gets its model identity from the saved job"
                )
            reference.update(job=str(job.pk), model=job.base_model)
        elif kind == "foundation":
            model = participant.get("model")
            if (
                not model
                or get_model_config_any_backend(model) is None
                or get_unsloth_image(model) not in NATIVE_FOUNDATION_STACKS
            ):
                raise InputValidationError(
                    "This foundation has no supported native evaluation runtime"
                )
            reference.update(model=model, hf_model=get_hf_base(model, backend="modal"))
        else:
            raise InputValidationError("A trained participant requires a job")
        result.append(reference)
    return result
