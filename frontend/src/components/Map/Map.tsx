import { useEffect, Fragment } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, CircleMarker, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import { useFleetStore } from '../../store/useFleetStore';
import 'leaflet/dist/leaflet.css';

const MapEvents = () => {
  const { createEmergency, dispatchMode } = useFleetStore();
  
  useMapEvents({
    click(e) {
      if (dispatchMode) {
        createEmergency(e.latlng.lat, e.latlng.lng, 'high');
      }
    },
  });
  return null;
};

const MapComponent = () => {
  const { ambulances, predictions, activeRoute, fetchAmbulances, fetchPredictions, fetchEmergencies, fetchMetrics } = useFleetStore();

  useEffect(() => {
    fetchAmbulances();
    fetchPredictions();
    fetchEmergencies();
    fetchMetrics();
    const interval = setInterval(() => {
      fetchAmbulances();
      fetchPredictions();
      fetchEmergencies();
      fetchMetrics();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const ambulanceIcon = (status: string) => L.divIcon({
    className: 'custom-div-icon',
    html: `<div style="background-color: ${status === 'idle' ? '#22c55e' : status === 'busy' ? '#ef4444' : '#eab308'}; width: 24px; height: 24px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 10px rgba(0,0,0,0.5);"></div>`,
    iconSize: [24, 24],
    iconAnchor: [12, 12]
  });

  return (
    <MapContainer 
      center={[18.51957, 73.85529]} 
      zoom={13} 
      className="w-full h-full"
    >
      <TileLayer
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        attribution='&copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom, 2012'
      />

      <MapEvents />

      {/* Heatmap Layer */}
      {predictions.map((p) => (
        <Fragment key={p.cluster_id}>
          {/* Outer Glow */}
          <CircleMarker
            center={[p.lat, p.lon]}
            radius={p.score * 12}
            pathOptions={{
              fillColor: '#ef4444',
              fillOpacity: 0.15,
              color: '#ef4444',
              weight: 1,
              opacity: 0.3,
              className: 'high-alert-zone'
            }}
          />
          {/* Inner Core */}
          <CircleMarker
            center={[p.lat, p.lon]}
            radius={p.score * 6}
            pathOptions={{
              fillColor: '#ef4444',
              fillOpacity: 0.5,
              color: '#ffffff',
              weight: 2,
              opacity: 0.8,
            }}
          >
            <Popup>
              <div className="text-sm font-sans">
                <p className="font-bold text-red-600">High Alert Zone</p>
                <p>Demand Score: {p.score.toFixed(2)}</p>
              </div>
            </Popup>
          </CircleMarker>
        </Fragment>
      ))}

      {/* Active Route Overlay */}
      {activeRoute && (
        <Polyline
          positions={activeRoute.path_coords}
          pathOptions={{ color: '#3b82f6', weight: 6, opacity: 0.8 }}
        />
      )}

      {/* Ambulance Markers */}
      {ambulances.map((a) => (
        <Marker
          key={a.id}
          position={[a.lat, a.lon]}
          icon={ambulanceIcon(a.status)}
        >
          <Popup>
            <div className="text-sm font-sans">
              <p className="font-bold">Ambulance #{a.id}</p>
              <p>Status: <span className="capitalize">{a.status}</span></p>
              {a.destination && <p>Heading to node: {a.destination}</p>}
            </div>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
};

export default MapComponent;
