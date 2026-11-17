from __future__ import annotations
import logging
import os
import sys
from typing import Any, Dict
LOG_FORMAT = '%(asctime)s [%(levelname)s] (%(name)s) %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_LEVEL_ENV = 'OILSPILL_LOG_LEVEL'
_CONFIGURED = False

def configure_logging(level: str | int | None=None, *, force: bool=False) -> Dict[str, Any]:
    global _CONFIGURED
    resolved: int | str = level if level is not None else os.environ.get(LOG_LEVEL_ENV, 'INFO')
    if isinstance(resolved, str):
        resolved = getattr(logging, resolved.upper(), logging.INFO)
    if _CONFIGURED and (not force):
        logging.getLogger().setLevel(resolved)
        return {'level': logging.getLevelName(resolved), 'configured': False}
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved)
    for noisy, noisy_level in (('opendrift', logging.WARNING), ('rasterio', logging.WARNING), ('matplotlib', logging.WARNING), ('urllib3', logging.WARNING)):
        logging.getLogger(noisy).setLevel(noisy_level)
    _CONFIGURED = True
    logging.getLogger('OilSpillAPI').debug('Logging configured at level %s', logging.getLevelName(resolved))
    return {'level': logging.getLevelName(resolved), 'configured': True}