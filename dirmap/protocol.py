"""Compatibility versions shared with the built student application."""
import json
from pathlib import Path

_schemas = json.loads(Path(__file__).with_name('schemas.json').read_text())
ROUTING_SCHEMA = _schemas['routing']
TERRAIN_SCHEMA = _schemas['terrain']
