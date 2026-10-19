import Map from 'ol/Map.js';
import View from 'ol/View.js';
import TileLayer from 'ol/layer/Tile.js';
import ImageLayer from 'ol/layer/Image.js';
import VectorLayer from 'ol/layer/Vector.js';
import OSM from 'ol/source/OSM.js';
import VectorSource from 'ol/source/Vector.js';
import ImageStatic from 'ol/source/ImageStatic.js';
import Feature from 'ol/Feature.js';
import Point from 'ol/geom/Point.js';
import LineString from 'ol/geom/LineString.js';
import CircleGeometry from 'ol/geom/Circle.js';
import GeoJSON from 'ol/format/GeoJSON.js';
import { Circle, Fill, Stroke, Style, Text } from 'ol/style.js';
import { fromLonLat, toLonLat } from 'ol/proj.js';
import { ScaleLine, defaults } from 'ol/control.js';
import type { Approach, Control, MapState, Pixel, Route, StaticRelease } from './types';
import { mapAssetUrl, releaseOverlayUrl } from './data';

export const teamColors = ['#ec693f', '#416edb', '#9370ca', '#159386', '#bb8426', '#d95492', '#56974e', '#a5513f', '#54869f', '#887760'];
export class CampusMap {
  private base = new TileLayer({ source: new OSM(), opacity: .65 });
  private referenceSource = new VectorSource({ attributions: '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap contributors</a>' });
  private routes = new VectorSource();
  private controls = new VectorSource();
  private location = new VectorSource();
  private endpoints = new VectorSource();
  private approaches = new VectorSource();
  private overlays: ImageLayer<ImageStatic>[] = [];
  private terrain: ImageLayer<ImageStatic> | null = null;
  map: Map;
  onClick: (lnglat: Pixel, controlId: string | null) => void = () => {};
  onDrag: () => void = () => {};

  constructor(target: HTMLElement) {
    const reference = new VectorLayer({ source: this.referenceSource, style: feature => {
      const kind = feature.get('kind');
      if (kind === 'water') return new Style({ fill: new Fill({ color: '#b4dce5' }), stroke: new Stroke({ color: '#8cbdc9', width: 1 }) });
      if (kind === 'building') return new Style({ fill: new Fill({ color: '#d4dace' }), stroke: new Stroke({ color: '#adb8a9', width: .6 }) });
      const highway = feature.get('highway');
      return new Style({ stroke: new Stroke({ color: ['footway', 'path', 'steps'].includes(highway) ? '#cbd3c4' : '#fcfcf7', width: ['primary', 'secondary'].includes(highway) ? 6 : 2.4 }) });
    }});
    this.map = new Map({ target, layers: [this.base, reference,
      new VectorLayer({ source: this.routes, zIndex: 30 }), new VectorLayer({ source: this.controls, zIndex: 40 }),
      new VectorLayer({ source: this.endpoints, zIndex: 45 }), new VectorLayer({ source: this.approaches, zIndex: 46 }),
      new VectorLayer({ source: this.location, zIndex: 50 })],
      view: new View({ center: fromLonLat([113.9304, 22.535]), zoom: 16.2, minZoom: 13, maxZoom: 21, enableRotation: true }),
      controls: defaults({ rotate: false }).extend([new ScaleLine({ units: 'metric' })]) });
    this.map.on('singleclick', event => {
      const feature = this.map.forEachFeatureAtPixel(event.pixel, feature => feature, { hitTolerance: 9 });
      this.onClick(toLonLat(event.coordinate) as Pixel, feature?.get('controlId') ?? null);
    });
    this.map.on('pointerdrag', () => this.onDrag());
  }

  setReference(data: object) { this.referenceSource.clear(); this.referenceSource.addFeatures(new GeoJSON().readFeatures(data, { featureProjection: 'EPSG:3857' })); }
  setStandardMap(visible: boolean) { this.base.setVisible(visible); }
  setControls(controls: Control[]) {
    this.controls.clear();
    for (const c of controls) {
      if (!c.lnglat || c.state === 'rejected') continue;
      const feature = new Feature({ geometry: new Point(fromLonLat(c.lnglat)), controlId: c.id });
      const color = c.required ? '#9c3fa6' : '#798773';
      feature.setStyle(new Style({ image: new Circle({ radius: 12, fill: new Fill({ color: '#fffffff2' }), stroke: new Stroke({ color, width: 2.5, lineDash: c.state === 'candidate' ? [5, 2] : undefined }) }),
        text: new Text({ text: c.label.replace('待核', '?'), font: '600 12px sans-serif', fill: new Fill({ color }), offsetY: -23, stroke: new Stroke({ color: '#fff', width: 3 }) }) }));
      this.controls.addFeature(feature);
    }
  }
  setOverlays(states: MapState[], opacity: number, activeId: string) {
    for (const layer of this.overlays) this.map.removeLayer(layer);
    this.overlays = [];
    for (const state of [...states].sort((a, b) => Number(a.assetId === activeId) - Number(b.assetId === activeId))) {
      if (!state.extent || !state.registration || state.registrationStale) continue;
      const layer = new ImageLayer({ source: new ImageStatic({ url: mapAssetUrl(state.assetId, 'overlay.png', state.version), imageExtent: state.extent, projection: 'EPSG:3857' }),
        opacity: state.assetId === activeId ? opacity : opacity * .6, zIndex: 10 });
      this.overlays.push(layer); this.map.addLayer(layer);
    }
  }
  setReleaseOverlay(release: StaticRelease, opacity: number) {
    const layer = new ImageLayer({ source: new ImageStatic({ url: releaseOverlayUrl(release), imageExtent: release.extent, projection: 'EPSG:3857' }), opacity, zIndex: 10 });
    this.overlays.push(layer); this.map.addLayer(layer); this.fit(release.extent);
  }
  setOpacity(opacity: number) { for (const layer of this.overlays) layer.setOpacity(opacity); }
  setTerrain(state: MapState | null, visible: boolean) {
    if (this.terrain) this.map.removeLayer(this.terrain);
    this.terrain = null;
    if (!state?.extent || state.registrationStale || !visible) return;
    this.terrain = new ImageLayer({ source: new ImageStatic({ url: mapAssetUrl(state.assetId, 'terrain-warp.png', state.version), imageExtent: state.extent, projection: 'EPSG:3857' }), opacity: .8, zIndex: 20 });
    this.map.addLayer(this.terrain);
  }
  setRoutes(routes: Route[], selected: string | null = null) {
    this.routes.clear();
    routes.forEach((route, i) => {
      if (selected && route.teamId !== selected) return;
      const feature = new Feature(new LineString(route.geometry.map(p => fromLonLat(p))));
      feature.setStyle([new Style({ stroke: new Stroke({ color: '#ffffffe6', width: 7 }) }), new Style({ stroke: new Stroke({ color: teamColors[i % teamColors.length], width: 4 }) })]);
      this.routes.addFeature(feature);
    });
  }
  setApproaches(approaches: Approach[]) {
    this.approaches.clear();
    for (const a of approaches) {
      if (a.offsetM < .1) continue;
      const feature = new Feature({ geometry: new Point(fromLonLat(a.lnglat)), controlId: a.controlId });
      feature.setStyle(new Style({ image: new Circle({ radius: 5, fill: new Fill({ color: '#e58824' }), stroke: new Stroke({ color: '#fff', width: 2 }) }),
        text: new Text({ text: a.automatic ? '建议到访' : '到访', offsetY: 16, font: '11px sans-serif', fill: new Fill({ color: '#864809' }), stroke: new Stroke({ color: '#fff', width: 3 }) }) }));
      this.approaches.addFeature(feature);
    }
  }
  setEndpoint(name: 'start' | 'finish', lnglat: Pixel | null) {
    const previous = this.endpoints.getFeatureById(name);
    if (previous) this.endpoints.removeFeature(previous);
    if (!lnglat) return;
    const feature = new Feature(new Point(fromLonLat(lnglat))); feature.setId(name);
    feature.setStyle(new Style({ image: new Circle({ radius: 10, fill: new Fill({ color: name === 'start' ? '#266b54' : '#253b30' }), stroke: new Stroke({ color: '#fff', width: 2 }) }),
      text: new Text({ text: name === 'start' ? '起' : '终', fill: new Fill({ color: '#fff' }), font: 'bold 11px sans-serif' }) }));
    this.endpoints.addFeature(feature);
  }
  setLocation(position: GeolocationPosition, follow: boolean) {
    this.location.clear();
    const coords: Pixel = [position.coords.longitude, position.coords.latitude], point = fromLonLat(coords);
    const accuracy = new Feature(new CircleGeometry(point, position.coords.accuracy / Math.cos(coords[1] * Math.PI / 180)));
    accuracy.setStyle(new Style({ fill: new Fill({ color: '#347fe520' }), stroke: new Stroke({ color: '#347fe550', width: 1 }) }));
    const dot = new Feature(new Point(point));
    dot.setStyle(new Style({ image: new Circle({ radius: 7, fill: new Fill({ color: '#347fe5' }), stroke: new Stroke({ color: '#fff', width: 3 }) }) }));
    this.location.addFeatures([accuracy, dot]);
    if (follow) this.map.getView().setCenter(point);
  }
  setHeading(degrees: number) { this.map.getView().setRotation(-degrees * Math.PI / 180); }
  northUp() { this.map.getView().animate({ rotation: 0, duration: 250 }); }
  fit(extent: number[]) { this.map.getView().fit(extent, { padding: [60, 60, 60, 60], duration: 400, maxZoom: 18 }); }
  destroy() { this.map.setTarget(undefined); }
}
