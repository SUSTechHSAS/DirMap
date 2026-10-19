"""Campus OSM reference. It is geodetic evidence, not a complete route graph."""

import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import Frame


def polygon_properties(points):
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    moments = cv2.moments(contour)
    area = abs(moments['m00'])
    if area < 1:
        return None
    center = [moments['m10']/moments['m00'], moments['m01']/moments['m00']]
    _, sides, angle = cv2.minAreaRect(contour)
    ratio = max(sides) / max(1e-6, min(sides))
    return {'center': center, 'area': area, 'aspect': ratio, 'points': np.asarray(points).tolist()}


def load_reference(path, frame=None):
    frame = frame or Frame()
    data = json.loads(Path(path).read_text())
    for name in ('yuehai-osm-extra.json', 'yuehai-sports.json', 'yuehai-building-relations.json'):
        extra = Path(path).with_name(name)
        if extra.exists():
            data['elements'] += json.loads(extra.read_text())['elements']
    features = []
    for element in data['elements']:
        tags = element.get('tags', {})
        coordinates = [[p['lon'], p['lat']] for p in element.get('geometry', []) if p]
        if element.get('type') == 'relation':
            # Join open OSM outer-member ways by endpoint. Inner islands are
            # irrelevant to a matching outline; route generation never uses
            # the filled reference polygon as a traversability mask.
            segments = [[[p['lon'], p['lat']] for p in m.get('geometry', []) if p]
                        for m in element.get('members', []) if m.get('role') == 'outer' and m.get('geometry')]
            coordinates = segments.pop(0) if segments else []
            while segments:
                matched = False
                for i, segment in enumerate(segments):
                    if coordinates[-1] == segment[0]:
                        coordinates += segment[1:]
                    elif coordinates[-1] == segment[-1]:
                        coordinates += list(reversed(segment))[1:]
                    elif coordinates[0] == segment[-1]:
                        coordinates = segment[:-1]+coordinates
                    elif coordinates[0] == segment[0]:
                        coordinates = list(reversed(segment))[:-1]+coordinates
                    else:
                        continue
                    segments.pop(i); matched = True; break
                if not matched:
                    break
            if segments or not coordinates or coordinates[0] != coordinates[-1]:
                continue
        if len(coordinates) < 2:
            continue
        kind = ('building' if tags.get('building') else
                'water' if tags.get('natural') == 'water' or tags.get('leisure') == 'swimming_pool' else
                'pitch' if tags.get('leisure') == 'pitch' else
                'road' if tags.get('highway') else None)
        if kind is None:
            continue
        local = frame.to_local(coordinates)
        properties = polygon_properties(local) if kind != 'road' else None
        if kind != 'road' and (not properties or properties['area'] < 40):
            continue
        features.append({'id': str(element['id']), 'kind': kind, 'tags': tags,
                         'lnglat': coordinates, 'points': local.tolist(), **(properties or {})})
    return features


def geojson(features):
    return {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'id': item['id'],
         'properties': {'kind': item['kind'], **item['tags']},
         'geometry': {'type': 'LineString' if item['kind'] == 'road' else 'Polygon',
                      'coordinates': item['lnglat'] if item['kind'] == 'road' else [item['lnglat']]}}
        for item in features]}
