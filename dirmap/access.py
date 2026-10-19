"""Legal approach candidates for cartographic point symbols and building edges.

The circle centre remains the target's exact map position. Approaches end on
walkable ground and never add an edge through an obstacle. A long wall, a
forbidden area, conflicting sides, and a large enclosing feature need review.
"""

import cv2
import numpy as np

from .graph import segment_pixels


class AccessResolver:
    def __init__(self, graph, registration, semantics):
        self.graph = graph
        self.registration = registration
        self.semantics = semantics
        self.components = {}
        self.scale = float(np.sqrt(abs(np.linalg.det(registration.affine[:, :2]))))

    def _components(self, kind):
        if kind not in self.components:
            mask = self.semantics.get(kind)
            if mask is None:
                return None
            mask = (mask > 0).astype(np.uint8)
            if kind == 'water':
                # This semantic footprint only joins the quadrants of a
                # tiny fountain's cross symbol. It never modifies terrain.
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            self.components[kind] = (labels, stats)
        return self.components[kind]

    def _feature(self, kind, point, circle_radius):
        point = np.asarray(point, dtype=float)
        parts = self._components(kind)
        if parts is None:
            return None
        labels, stats = parts
        x, y = np.rint(point).astype(int)
        ids = []
        if labels[y, x]:
            ids.append(int(labels[y, x]))
        near = labels[max(0, y-2):y+3, max(0, x-2):x+3]
        ids += [int(i) for i in np.unique(near) if i and i not in ids]
        # White interiors of a small ring/cross can be isolated graph nodes;
        # look for an enclosing point-symbol contour, not an enclosing wall.
        radius = max(8, round(min(circle_radius, 12/max(self.scale, .1))))
        x0, y0 = max(0, x-radius), max(0, y-radius)
        x1, y1 = min(labels.shape[1], x+radius+1), min(labels.shape[0], y+radius+1)
        crop = (labels[y0:y1, x0:x1] > 0).astype(np.uint8)
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.pointPolygonTest(contour, (float(point[0]-x0), float(point[1]-y0)), False) >= 0:
                px, py = contour[0, 0]
                label = int(labels[py+y0, px+x0])
                if label and label not in ids:
                    ids.append(label)
        for label in ids:
            bx, by, width, height, area = stats[label]
            largest, smallest = max(width, height)*self.scale, min(width, height)*self.scale
            if kind in ('point_ink', 'vegetation'):
                if largest > (18 if kind == 'point_ink' else 8) or max(width, height) > max(12, circle_radius):
                    continue
                local = (labels[by:by+height, bx:bx+width] == label).astype(np.uint8)
                yy, xx = np.nonzero(local)
                if len(xx) < 3:
                    continue
                covariance = np.cov(np.column_stack((xx, yy)).T)
                eigenvalues = np.linalg.eigvalsh(covariance)
                if eigenvalues[0] < .08*max(1e-6, eigenvalues[1]):
                    continue  # narrow line / fence, regardless of angle
                if kind == 'point_ink' and area/(width*height) < .22:
                    continue  # a sparse angular wall is not a solid sculpture
                limit, method = min(8., max(3., largest/2+1)), 'point_symbol_edge'
            elif kind == 'building':
                # Reject a narrow elongated grey wall as well as a linear
                # black barrier; do not pick its other side as a checkpoint.
                if smallest < 3.5 and largest > 3*max(.1, smallest):
                    continue
                if area*self.scale*self.scale < 8:
                    continue
                local = (labels[by:by+height, bx:bx+width] == label).astype(np.uint8)
                limit, method = 3., 'building_edge'
                if largest <= 26 and max(width, height) <= circle_radius*1.5 and area/(width*height) < .55:
                    contours, hierarchy = cv2.findContours(local, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    rings = []
                    for contour, relation in zip(contours, hierarchy[0] if hierarchy is not None else []):
                        if relation[3] < 0:
                            continue
                        _, ring_radius = cv2.minEnclosingCircle(contour)
                        if cv2.contourArea(contour) >= 6 and cv2.contourArea(contour)/max(1, np.pi*ring_radius**2) >= .75:
                            rings.append(contour)
                    if len(rings) >= 2 and any(cv2.pointPolygonTest(c, tuple((point-[bx, by]).astype(float)), False) >= 0 for c in rings):
                        limit, method = 8., 'compound_point_symbol_edge'
            else:
                local = (labels[by:by+height, bx:bx+width] == label).astype(np.uint8)
                outline, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not outline:
                    continue
                contour = max(outline, key=cv2.contourArea)
                circularity = 4*np.pi*cv2.contourArea(contour)/max(1., cv2.arcLength(contour, True)**2)
                if largest <= 18 and smallest >= largest*.6 and area*self.scale*self.scale <= 240 and circularity >= .65:
                    limit, method = min(9., largest/2+1.5), 'small_water_feature_edge'
                else:
                    # A bank can be approached only immediately at its edge;
                    # an interior lake point cannot be moved to a distant road.
                    limit, method = 3., 'water_edge'
            # A symbol footprint may be filled to locate its outside edge;
            # this fill is NEVER applied to the graph's navigable cost mask.
            contour_list, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            footprint = np.zeros_like(local)
            cv2.drawContours(footprint, contour_list, -1, 1, cv2.FILLED)
            footprint = cv2.copyMakeBorder(footprint, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
            # Include the anti-aliased outline around a roof/fountain.
            footprint = cv2.dilate(footprint, np.ones((3, 3), np.uint8))
            origin = np.array([bx-2, by-2])
            at = np.rint(point-origin).astype(int)
            if not (0 <= at[0] < footprint.shape[1] and 0 <= at[1] < footprint.shape[0] and footprint[at[1], at[0]]):
                continue
            # The filled, borderless silhouette is the symbol itself; the
            # dilated footprint only bounds the search. Crossing the symbol's
            # own body is expected for a checkpoint drawn on it, while any
            # other blocked pixel still disqualifies the approach.
            symbol = np.zeros_like(local)
            cv2.drawContours(symbol, contour_list, -1, 1, cv2.FILLED)
            return {'origin': origin, 'mask': footprint, 'limitM': limit, 'method': method, 'kind': kind,
                    'symbol': symbol, 'symbolOrigin': np.array([bx, by])}
        return None

    @staticmethod
    def _in_feature(feature, points):
        xy = np.rint(np.asarray(points)-feature['origin']).astype(int)
        valid = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < feature['mask'].shape[1]) & (xy[:, 1] < feature['mask'].shape[0])
        inside = np.zeros(len(xy), dtype=bool)
        inside[valid] = feature['mask'][xy[valid, 1], xy[valid, 0]] > 0
        return inside

    @staticmethod
    def _over_symbol(feature, points):
        """Whether each pixel lies on the feature itself, not just its halo."""
        symbol = feature.get('symbol')
        if symbol is None:
            return AccessResolver._in_feature(feature, points)
        xy = np.rint(np.asarray(points)-feature['symbolOrigin']).astype(int)
        valid = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < symbol.shape[1]) & (xy[:, 1] < symbol.shape[0])
        inside = np.zeros(len(xy), dtype=bool)
        inside[valid] = symbol[xy[valid, 1], xy[valid, 0]] > 0
        return inside

    def _complete_outline(self, feature, circle_radius):
        """Recover a small symbol's dark rim, never a connected wall.

        A fountain's black outline is not in the blue mask; compression can
        likewise leave grey edge pixels around a black point symbol. Absorb
        only complete, tightly bounded obstacle components. A component
        continuing beyond the symbol's small halo is a separate/attached
        barrier, and cannot be licensed by its connection to the symbol.
        """
        if feature.get('stroke') or feature['method'] in ('building_edge', 'water_edge'):
            return feature
        symbol = feature['symbol']
        sx, sy = feature['symbolOrigin']
        height, width = self.graph.costs.shape
        padding = max(2, min(4, round(circle_radius*.10)))
        x0, y0 = max(0, sx-padding-1), max(0, sy-padding-1)
        x1 = min(width, sx+symbol.shape[1]+padding+1)
        y1 = min(height, sy+symbol.shape[0]+padding+1)
        hard = (self.graph.costs[y0:y1, x0:x1] <= 0).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(hard, 8)
        owned = np.zeros_like(hard)
        owned[sy-y0:sy-y0+symbol.shape[0], sx-x0:sx-x0+symbol.shape[1]] = symbol
        for label in np.unique(labels[owned > 0]):
            if not label:
                continue
            bx, by, bw, bh, _ = stats[label]
            # Complete component must have a free border on all four sides.
            if bx == 0 or by == 0 or bx+bw >= hard.shape[1] or by+bh >= hard.shape[0]:
                continue
            component = labels == label
            if any(mask is not None and np.any(mask[y0:y1, x0:x1][component])
                   for mask in (self.semantics.get('restricted'), getattr(self.graph, 'closed_edits', None))):
                continue
            owned[component] = 1
        contours, _ = cv2.findContours(owned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(owned, contours, -1, 1, cv2.FILLED)
        return {**feature, 'symbol': owned, 'symbolOrigin': np.array([x0, y0]),
                'mask': cv2.dilate(owned, np.ones((3, 3), np.uint8)), 'origin': np.array([x0, y0])}

    def _ink_feature(self, point, circle_radius):
        """Footprint of thin black strokes under a circle centre.

        A control centred on a fence, rock edge, or contour touches only a
        few native pixels of that line. Region-size and elongation filters
        drop those strokes from `point_ink`, but they still mark the feature
        the checkpoint stands on, so approach them from outside the strokes.
        """
        ink = self.semantics.get('point_ink')
        if ink is None:
            return None
        ink = (ink > 0).astype(np.uint8)
        height, width = ink.shape
        radius = max(6, int(round(min(circle_radius, 12))))
        x, y = np.rint(point).astype(int)
        x0, y0 = max(0, x-radius), max(0, y-radius)
        x1, y1 = min(width, x+radius+1), min(height, y+radius+1)
        local = ink[y0:y1, x0:x1]
        if not local.any():
            return None
        # Only the stroke under the circle counts. A wall a few pixels away
        # must never be bridged just because the checkpoint happens to be
        # near ink, and a long barrier is not a course-line-width stroke.
        _, components, stats, _ = cv2.connectedComponentsWithStats(local, 8)
        label = int(components[round(point[1])-y0, round(point[0])-x0])
        if label == 0:
            return None
        if max(stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]) > max(6, radius*1.5):
            return None
        stroke = (components == label).astype(np.uint8)
        # The stroke's own anti-aliased edge belongs to the symbol; the wider
        # footprint only bounds the search, and blocked pixels outside the
        # stroke itself are still refused in `_stroke_component`.
        symbol = cv2.dilate(stroke, np.ones((3, 3), np.uint8))
        grown = cv2.dilate(stroke, np.ones((5, 5), np.uint8))
        if not grown[round(point[1])-y0, round(point[0])-x0]:
            return None
        return {'origin': np.array([x0, y0]), 'mask': grown, 'limitM': 4.,
                'method': 'ink_stroke_edge', 'kind': 'point_ink', 'stroke': True,
                'strokeLabel': label, 'symbol': symbol, 'symbolOrigin': np.array([x0, y0])}

    @staticmethod
    def _stroke_component(feature, blocked):
        """True when every blocked pixel lies in the checkpoint's own stroke."""
        symbol = feature.get('symbol')
        if symbol is None:
            return None
        xy = np.rint(np.asarray(blocked)-feature['symbolOrigin']).astype(int)
        if (xy < 0).any() or (xy[:, 0] >= symbol.shape[1]).any() or (xy[:, 1] >= symbol.shape[0]).any():
            return None
        return True if np.all(symbol[xy[:, 1], xy[:, 0]] > 0) else None

    def resolve(self, control, required_component=None):
        point = np.asarray(control.get('approachPixel') or control['pixel'], dtype=float)
        world, valid = self.registration.forward([point])
        if not valid[0]:
            raise ValueError('目标超出有效配准范围，需要补充参照')
        x, y = np.rint(point).astype(int)
        if not (0 <= x < self.graph.costs.shape[1] and 0 <= y < self.graph.costs.shape[0]):
            raise ValueError('到访位置超出原图')
        if getattr(self.graph, 'closed_edits', None) is not None and self.graph.closed_edits[y, x]:
            raise ValueError('教师已封闭该位置，不能自动改用地物外沿')
        try:
            node = self.graph.snap(point, max_distance_px=4)
            if required_component is None or self.graph.components[node] == required_component:
                return {'node': node, 'pixel': point.tolist(), 'local': world[0].tolist(),
                        'method': 'manual' if control.get('approachPixel') else 'exact', 'offsetM': 0., 'automatic': False}
        except ValueError:
            pass
        if control.get('approachPixel'):
            raise ValueError('指定到访位置不可达，请核对门洞和到访侧')
        if self.semantics.get('restricted') is not None and self.semantics['restricted'][y, x]:
            raise ValueError('圈心位于禁入区域，不能自动移到外部道路')
        radius = control.get('radiusPx', 15)
        feature = None
        for kind in ('water', 'vegetation', 'point_ink', 'building'):
            feature = self._feature(kind, point, radius)
            if feature:
                break
        if feature is None:
            feature = self._ink_feature(point, radius)
        if feature is None:
            raise ValueError('圈心处的地物或通行侧不明确，请核对实际到访位置')
        feature = self._complete_outline(feature, radius)
        nearby = self.graph.tree.query_ball_point(point, feature['limitM']/max(self.scale, .1)+2)
        if not feature.get('stroke'):
            nearby = [n for n in nearby if not self._in_feature(feature, self.graph.pixels[[n]])[0]]
        candidates = []
        for node in nearby:
            offset = float(np.linalg.norm(self.graph.local[node]-world[0]))
            if offset > feature['limitM']:
                continue
            end = self.graph.pixels[node]
            pixels = segment_pixels(point, end)
            # This ray associates a target with its approach; it is NEVER a
            # route edge. Only the target's bounded silhouette may be crossed.
            # Connected obstacle blobs are not object identities: a tree can
            # touch a wall and a building can touch a forbidden garden.
            over_symbol = self._over_symbol(feature, pixels)
            hard = self.graph.costs[pixels[:, 1], pixels[:, 0]] == 0
            if np.any(hard & ~over_symbol):
                continue
            restricted = self.semantics.get('restricted')
            closed = getattr(self.graph, 'closed_edits', None)
            if ((restricted is not None and np.any(restricted[pixels[:, 1], pixels[:, 0]])) or
                    (closed is not None and np.any(closed[pixels[:, 1], pixels[:, 0]]))):
                continue
            # A circle drawn in the hollow of a compound symbol legitimately
            # leaves and re-enters the symbol's own arc; the class check above
            # and the side check below already exclude a hidden obstacle, so a
            # re-entry is not by itself evidence of crossing a wall.
            candidates.append((offset, int(node)))
        if not candidates:
            restricted = self.semantics.get('restricted')
            if restricted is not None:
                reach = max(2, round(feature['limitM']/max(self.scale, .1)))
                patch = restricted[max(0, y-reach):y+reach+1, max(0, x-reach):x+reach+1]
                if patch.size and np.mean(patch) > .45:
                    raise ValueError('目标地物被禁入区域包围，请核对点位；不能跨禁区自动接入道路')
            raise ValueError('地物外沿没有无障碍的近距离到访位置')
        # Local topology, not just global connectivity: the two sides of a
        # wall can connect far away and still be ambiguous at this control.
        local_radius = max(8, int(np.ceil(feature['limitM']*2/max(self.scale, .1))),
                           max(feature['symbol'].shape)+3)
        x0, y0 = max(0, x-local_radius), max(0, y-local_radius)
        x1, y1 = min(self.graph.costs.shape[1], x+local_radius+1), min(self.graph.costs.shape[0], y+local_radius+1)
        local_free = (self.graph.costs[y0:y1, x0:x1] > 0).astype(np.uint8)
        _, local_labels = cv2.connectedComponents(local_free, connectivity=4)
        side_of = {n: int(local_labels[round(self.graph.pixels[n, 1])-y0, round(self.graph.pixels[n, 0])-x0]) for _, n in candidates}
        if feature['method'] in ('point_symbol_edge', 'small_water_feature_edge', 'compound_point_symbol_edge'):
            # The white holes inside a printed sculpture/fountain are not
            # ground on another approach side. Keep the local exterior of
            # the complete point symbol; both sides of a wall still qualify
            # as exterior and therefore remain ambiguous below.
            exterior = set(np.concatenate((local_labels[0], local_labels[-1],
                                           local_labels[:, 0], local_labels[:, -1])).tolist())
            candidates = [(d, n) for d, n in candidates if side_of[n] in exterior]
            if not candidates:
                raise ValueError('地物符号外部没有可确认的到访位置')
        groups = {}
        for distance, node in candidates:
            groups.setdefault(side_of[node], []).append((distance, node))
        if len(groups) != 1:
            raise ValueError('地物附近存在不同通行侧，需要选择实际到访侧')
        if required_component is not None:
            candidates = [(d, n) for d, n in candidates if self.graph.components[n] == required_component]
        if not candidates:
            raise ValueError('合法到访侧与起点不连通，请核对通道')
        offset, node = min(candidates)
        return {'node': node, 'pixel': self.graph.pixels[node].astype(float).tolist(),
                'local': self.graph.local[node].tolist(), 'method': feature['method'],
                'featureKind': feature['kind'], 'offsetM': offset, 'automatic': True,
                'reviewRequired': True, 'targetPixel': list(control['pixel'])}
