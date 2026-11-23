import logging
import logging.config
import os

from celery import Celery
from celery.signals import setup_logging, worker_init

import overbae.worker_watchdog  # noqa: F401 - connects its worker signal

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "overbae.settings")

app = Celery("overbae")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks(["overbae.tasks"])


@setup_logging.connect
def _configure_celery_logging(**_kwargs: object) -> None:
    """Use Django's LOGGING config: Celery's own formatters emit multi-line
    output that CloudWatch splits into separate log entries.
    """
    from django.conf import settings

    logging.config.dictConfig(settings.LOGGING)


@worker_init.connect
def _load_masking_model(**_kwargs: object) -> None:
    """The masking worker loads the name model once, here, before the pool forks, so its
    children share one copy. A failed load is logged and leaves every item to fail closed to
    the local judge (reason ``model_unavailable``); other workers never import torch."""
    from django.conf import settings

    if not settings.MASKING_ENABLED:
        return
    from overbae.services.masking import ner
    from overbae.services.masking.errors import MaskingError

    try:
        ner.backend()
    except MaskingError as exc:
        logging.getLogger(__name__).error("Masking model not loaded: %s", exc.reason)
    else:
        logging.getLogger(__name__).info("Masking model loaded")


def get_celery_app() -> Celery:
    return app
