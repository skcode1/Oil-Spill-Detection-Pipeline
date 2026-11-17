from __future__ import annotations
import logging
import os
from typing import Any, Dict, List, Optional
import yaml
logger = logging.getLogger(__name__)
DEFAULT_REGIONS: List[Dict[str, Any]] = [{'name': 'Strait of Malacca', 'lat': 2.5, 'lon': 101.5, 'corridor_bearing': 310.0}, {'name': 'Persian Gulf TSS', 'lat': 26.5, 'lon': 53.0, 'corridor_bearing': 290.0}, {'name': 'Singapore Strait', 'lat': 1.25, 'lon': 103.9, 'corridor_bearing': 70.0}, {'name': 'English Channel TSS', 'lat': 50.2, 'lon': -0.5, 'corridor_bearing': 65.0}, {'name': 'Gulf of Guinea', 'lat': 4.5, 'lon': 6.0, 'corridor_bearing': 90.0}]

def load_regions(config_path: str='config.yaml') -> List[Dict[str, Any]]:
    regions: Optional[List[Dict[str, Any]]] = None
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as handle:
                cfg = yaml.safe_load(handle) or {}
            regions = cfg.get('regions') or None
        except Exception as exc:
            logger.warning('Could not read regions from %s (%s); using defaults.', config_path, exc)
    if not regions:
        return [dict(region) for region in DEFAULT_REGIONS]
    validated: List[Dict[str, Any]] = []
    for entry in regions:
        if not isinstance(entry, dict) or 'name' not in entry or 'lat' not in entry or ('lon' not in entry):
            raise ValueError(f'Invalid region entry (needs name/lat/lon): {entry!r}')
        region = {'name': str(entry['name']), 'lat': float(entry['lat']), 'lon': float(entry['lon']), 'corridor_bearing': float(entry.get('corridor_bearing', 0.0)) % 360.0}
        if not -90.0 <= region['lat'] <= 90.0 or not -180.0 <= region['lon'] <= 180.0:
            raise ValueError(f'Region coordinates out of range: {region!r}')
        validated.append(region)
    return validated

def get_region(index: int=0, name: Optional[str]=None, config_path: str='config.yaml') -> Dict[str, Any]:
    regions = load_regions(config_path)
    if name:
        for region in regions:
            if region['name'].casefold() == str(name).casefold():
                return region
        raise KeyError(f'Region {name!r} not found in registry.')
    return regions[index % len(regions)]