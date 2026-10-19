import cv2
import numpy as np
import pytest

from dirmap import store
from dirmap.geometry import Registration, affine_points
from dirmap.graph import build_graph
from dirmap.vision import analyze, colour_masks, read_image


def paper(kind):
    image = np.full((300, 600, 3), 245, np.uint8)
    if kind == 'lake':
        image[60:240, 240:360] = [215, 180, 80]
    elif kind == 'building':
        image[60:240, 240:360] = 145
    elif kind == 'wall':
        cv2.line(image, (300, 0), (300, 299), (20, 20, 20), 2)
    elif kind == 'narrow-road':
        for x in (294, 306):
            cv2.line(image, (x, 0), (x, 299), (20, 20, 20), 1)
    purple = (190, 35, 165)
    for point in ((60, 150), (540, 150)):
        cv2.circle(image, point, 13, purple, 3, cv2.LINE_AA)
    cv2.line(image, (74, 150), (526, 150), purple, 5, cv2.LINE_AA)
    return image


def graph_for(result):
    source = [[0, 0], [599, 0], [599, 299], [0, 299]]
    matrix = [[1, 0, 0], [0, -1, 299]]
    return build_graph(result['costs'], Registration(matrix, source, affine_points(matrix, source)), 'overprint-regression')


@pytest.mark.parametrize('kind', ['lake', 'building'])
def test_course_line_cannot_open_a_corridor_through_an_area_obstacle(kind):
    result = analyze(paper(kind))
    assert len(result['controls']) == 2
    assert result['semantics']['course_ink'][150, 300]
    assert not np.any(result['costs'][70:230, 250:350])
    graph = graph_for(result)
    assert not graph.clear_segment([60, 150], [540, 150])
    a, b = graph.snap([60, 150]), graph.snap([540, 150])
    distance, parents = graph.shortest_paths([a], metric='distance')
    assert distance[0, b] > 500  # direct purple connection is only 480 pixels
    route = graph.expand(parents[0], a, b)
    assert all(graph.clear_segment(graph.pixels[x], graph.pixels[y]) for x, y in zip(route, route[1:]))


def test_course_line_is_removed_on_pavement_and_does_not_become_a_wall():
    result = analyze(paper('pavement'))
    assert result['semantics']['course_ink'][150, 300]
    assert graph_for(result).clear_segment([60, 150], [540, 150])


def test_hidden_wall_is_continued_without_closing_a_real_parallel_road():
    wall = graph_for(analyze(paper('wall')))
    a, b = wall.snap([60, 150]), wall.snap([540, 150])
    assert wall.components[a] != wall.components[b]
    road = graph_for(analyze(paper('narrow-road')))
    assert road.clear_segment([300, 20], [300, 280])


def test_supplied_map_lake_under_purple_connection_has_no_free_antialias_channel():
    # Frozen by visual inspection of the original, following the user's
    # report: this entire 23x22 patch is water, not a bridge or shore.
    image = read_image(store.get_asset('szu-3')['path'])
    assert colour_masks(image)['purple'][1811, 638] > 0
    result = analyze(image)
    assert result['semantics']['course_ink'][1811, 638]
    assert result['costs'][1811, 638] == 0
    assert np.count_nonzero(result['costs'][1800:1822, 627:650]) == 0
    # The roof crossed by 31->32 is visibly one solid grey area. JPEG noise
    # near the old grey threshold must not turn it into a walkable channel.
    assert np.count_nonzero(result['costs'][1180:1205, 1145:1180]) == 0


@pytest.mark.parametrize('shade', [145, 175, 180, 190])
def test_medium_grey_roof_under_course_ink_stays_blocked_but_canopy_stays_open(shade):
    image = paper('building')
    image[60:240, 240:360] = shade
    image[208:217, 240:360] = 215  # a genuine light-grey covered passage
    cv2.line(image, (74, 150), (526, 150), (190, 35, 165), 5, cv2.LINE_AA)
    result = analyze(image)
    graph = graph_for(result)
    assert np.all(result['costs'][70:140, 250:350] == 0)
    assert np.all(result['costs'][140:200, 250:350] == 0)
    assert graph.clear_segment([240, 212], [359, 212])


@pytest.mark.parametrize('angle', [0, 33, 78])
def test_course_circle_over_a_narrow_covered_exit_keeps_the_exit_and_roof_separate(angle):
    image = np.full((300, 600, 3), 245, np.uint8)
    image[75:250, 235:365] = 175
    image[126:148, 284:316] = 215
    image[147:250, 298:303] = 215
    transform = cv2.getRotationMatrix2D((300, 150), angle, 1)
    image = cv2.warpAffine(image, transform, (600, 300), borderValue=(245, 245, 245))
    center = (transform @ np.array([300, 137, 1])).tolist()
    cv2.circle(image, tuple(np.rint(center).astype(int)), 16, (190, 35, 165), 4, cv2.LINE_AA)
    result = analyze(image)
    assert len(result['controls']) == 1
    graph = graph_for(result)
    exit_point = (transform @ np.array([300, 265, 1])).tolist()
    first, last = graph.snap(center), graph.snap(exit_point)
    assert graph.components[first] == graph.components[last]
    # A roof next to the exit remains a roof, including under the course ink.
    roof_point = np.rint(transform @ np.array([309, 149, 1])).astype(int)
    assert result['costs'][roof_point[1], roof_point[0]] == 0


def test_actual_low_resolution_checkpoint_has_a_southern_canopy_exit_after_overprint_removal():
    # Original image and high-resolution counterpart independently show a
    # light-grey southern passage. No map-specific coordinates enter vision.
    image = read_image(store.get_asset('szu-2')['path'])
    result = analyze(image)
    _, labels = cv2.connectedComponents((result['costs'][183:237, 402:451] > 0).astype(np.uint8), connectivity=4)
    assert labels[192-183, 428-402] > 0
    assert labels[192-183, 428-402] == labels[233-183, 429-402]
    # The pillar in this passage and the surrounding roofs remain obstacles.
    assert result['costs'][196, 427] == 0
    assert result['costs'][208, 413] == 0
