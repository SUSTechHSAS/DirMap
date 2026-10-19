#!/usr/bin/env python3
"""Prepare the supplied real maps; existing teacher edits are preserved by default."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dirmap import store
from dirmap.service import process_asset, get_graph, refresh_ocr, refresh_terrain
from dirmap.vision import TERRAIN_SCHEMA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--force', action='store_true', help='Reprocess and replace prior recognition/teacher edits')
    parser.add_argument('--asset', help='Process only this map ID')
    actions.add_argument('--refresh-ocr', action='store_true', help='Refresh automatic labels, retaining teacher labels and geometry')
    actions.add_argument('--refresh-terrain', action='store_true', help='Refresh traversability while preserving controls, registration and edits')
    args = parser.parse_args()
    for asset in store.catalog():
        if args.asset and asset['id'] != args.asset:
            continue
        current = store.get_state(asset['id'])
        if args.refresh_terrain or (current and not args.force and current.get('terrainSchema') != TERRAIN_SCHEMA):
            state = refresh_terrain(asset['id'], current['version'] if current else None,
                                    lambda progress, message: print(f"{asset['id']}: {progress}% {message}", flush=True))
            print(asset['id'], 'terrain updated, version', state['version'], flush=True)
            if not args.refresh_ocr:
                continue
        if args.refresh_ocr:
            state = refresh_ocr(asset['id'])
            print(asset['id'], state['statistics']['ocr'], flush=True)
            continue
        if current and not args.force:
            print(f"{asset['id']}: keeping existing version {current['version']}", flush=True)
            if current.get('registration'):
                get_graph(current)
            continue
        started = time.perf_counter()
        state = process_asset(asset['id'], lambda progress, message: print(f"{asset['id']}: {progress}% {message}", flush=True))
        if state.get('registration'):
            get_graph(state)
        print(json.dumps({'assetId': asset['id'], 'elapsedSec': round(time.perf_counter()-started, 2),
                          'controls': len(state['controls']),
                          'calibratedControls': sum(c.get('calibrated', False) for c in state['controls']),
                          'quality': {k: v for k, v in (state.get('quality') or {}).items() if k != 'landmarks'}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
