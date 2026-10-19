#!/usr/bin/env python3
"""Fetch a bounded public OSM reference for SZU; no tile scraping or API key."""

import json
import argparse
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request
import subprocess
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dirmap import store


def fetch(query):
    failures = []
    for host in ('https://overpass-api.de/api/interpreter', 'https://overpass.kumi.systems/api/interpreter'):
        try:
            request = urllib.request.Request(host, data=urllib.parse.urlencode({'data': query}).encode(),
                                             headers={'User-Agent': 'DirMap/0.1 educational campus reference'})
            with urllib.request.urlopen(request, timeout=55) as response:
                data = json.load(response)
            if 'elements' not in data:
                raise ValueError('Overpass returned no elements')
            return data, host
        except Exception as exc:
            failures.append(f'{host}: {exc}')
    raise RuntimeError('; '.join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--building-relations-only', action='store_true', help='Refresh complete western campus building relations using the public OSM map API')
    args = parser.parse_args()
    if args.building_relations_only:
        url = 'https://api.openstreetmap.org/api/0.6/map?bbox=113.924,22.529,113.927,22.532'
        response = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--max-time', '45', url],
                                  capture_output=True, check=True, timeout=50)
        root = ET.fromstring(response.stdout)
        nodes = {e.attrib['id']: {'lon': float(e.attrib['lon']), 'lat': float(e.attrib['lat'])} for e in root.findall('node')}
        ways = {e.attrib['id']: e for e in root.findall('way')}
        elements = []
        for relation in root.findall('relation'):
            tags = {t.attrib['k']: t.attrib['v'] for t in relation.findall('tag')}
            if not tags.get('building'):
                continue
            members = []
            for member in relation.findall('member'):
                way = ways.get(member.attrib['ref'])
                if way is None:
                    continue
                geometry = [nodes.get(n.attrib['ref']) for n in way.findall('nd')]
                if None not in geometry:
                    members.append({'type': 'way', 'ref': int(member.attrib['ref']), 'role': member.attrib.get('role', ''), 'geometry': geometry})
            elements.append({'type': 'relation', 'id': int(relation.attrib['id']), 'tags': tags, 'members': members})
        store.atomic_json(store.DATA/'reference'/'yuehai-building-relations.json', {'elements': elements})
        store.atomic_json(store.DATA/'reference'/'yuehai-building-relations-source.json',
                          {'url': url, 'retrievedAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                           'crs': 'WGS84', 'license': 'ODbL 1.0', 'attribution': '© OpenStreetMap contributors',
                           'scope': 'Western campus window; complete building relation outer members.'})
        print(f'{len(elements)} building relations')
        return
    bbox = '(22.521,113.920,22.547,113.946)'
    queries = {
        'yuehai-osm.json': '[out:json][timeout:45];('+''.join(f'way{tag}{bbox};' for tag in
                           ('["building"]', '["natural"="water"]', '["waterway"]', '["highway"]'))+');out geom;',
        'yuehai-osm-extra.json': f'[out:json][timeout:45];(way["leisure"="swimming_pool"]{bbox};relation["natural"="water"]{bbox};);out geom;',
        'yuehai-sports.json': f'[out:json][timeout:45];way["leisure"~"^(pitch|track|stadium)$"]{bbox};out geom;',
        'yuehai-building-relations.json': f'[out:json][timeout:45];relation["building"]{bbox};out geom;'}
    sources = []
    for name, query in queries.items():
        data, host = fetch(query)
        store.atomic_json(store.DATA/'reference'/name, data)
        sources.append({'file': name, 'url': host, 'query': query, 'elements': len(data['elements'])})
        print(f'{name}: {len(data["elements"])} features')
    store.atomic_json(store.DATA/'reference'/'source.json', {'retrievedAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      'sources': sources, 'crs': 'WGS84', 'license': 'ODbL 1.0', 'copyright': '© OpenStreetMap contributors'})


if __name__ == '__main__':
    main()
