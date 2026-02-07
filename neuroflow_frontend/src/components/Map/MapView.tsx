import { useCallback, useState, useMemo, useEffect, useRef } from 'react';
import { GoogleMap, LoadScript, Marker, Polyline } from '@react-google-maps/api';
import { useMapStore } from '@/stores/mapStore';
import { useRouteStore } from '@/stores/routeStore';
import { GOOGLE_MAPS_API_KEY } from '@/utils/constants';

const containerStyle = {
  width: '100%',
  height: '100%',
};

import { SILVER_MAP_STYLE } from '@/utils/mapStyles';

const mapOptions: google.maps.MapOptions = {
  mapTypeId: 'roadmap',
  styles: SILVER_MAP_STYLE,
  tilt: 0,
  zoomControl: false, // Clean look
  mapTypeControl: false,
  streetViewControl: false,
  fullscreenControl: false,
};

// Traffic colors - matches Google Maps style
const TRAFFIC_COLORS = {
  FREE: '#0ea5e9',      // Blue - free flowing
  LIGHT: '#22c55e',     // Green - light traffic
  MODERATE: '#f59e0b',  // Yellow/Orange - moderate
  HEAVY: '#ef4444',     // Red - heavy traffic
  SEVERE: '#7f1d1d',    // Dark red - severe
};

// Determine color based on congestion ratio
function getTrafficColor(durationSeconds: number, durationInTrafficSeconds: number): string {
  if (!durationInTrafficSeconds || durationInTrafficSeconds <= durationSeconds) {
    return TRAFFIC_COLORS.FREE;
  }

  const ratio = durationInTrafficSeconds / durationSeconds;

  if (ratio < 1.15) return TRAFFIC_COLORS.FREE;
  if (ratio < 1.35) return TRAFFIC_COLORS.LIGHT;
  if (ratio < 1.6) return TRAFFIC_COLORS.MODERATE;
  if (ratio < 2.0) return TRAFFIC_COLORS.HEAVY;
  return TRAFFIC_COLORS.SEVERE;
}

interface RouteSegment {
  path: google.maps.LatLng[];
  color: string;
  isMainRoute: boolean;
}

export default function MapView() {
  const viewState = useMapStore((s) => s.viewState);
  const pickMode = useMapStore((s) => s.pickMode);
  const setPickMode = useMapStore((s) => s.setPickMode);
  const setOrigin = useRouteStore((s) => s.setOrigin);
  const setDestination = useRouteStore((s) => s.setDestination);
  const origin = useRouteStore((s) => s.origin);
  const destination = useRouteStore((s) => s.destination);

  // Actions to update store with route data
  const setGoogleRoute = useRouteStore((s) => s.setGoogleRoute);
  const setComputingRoute = useRouteStore((s) => s.setComputingRoute);
  const setError = useRouteStore((s) => s.setError);

  const [map, setMap] = useState<google.maps.Map | null>(null);
  const [routeSegments, setRouteSegments] = useState<RouteSegment[]>([]);
  const [apiLoaded, setApiLoaded] = useState(false);
  const directionsServiceRef = useRef<google.maps.DirectionsService | null>(null);

  const center = useMemo(() => ({
    lat: viewState.latitude,
    lng: viewState.longitude,
  }), [viewState.latitude, viewState.longitude]);

  // Auto-start in pick mode if no origin set
  useEffect(() => {
    if (apiLoaded && !origin && !pickMode) {
      setPickMode('origin');
    }
  }, [apiLoaded, origin, pickMode, setPickMode]);

  // Initialize DirectionsService once API is loaded
  useEffect(() => {
    if (apiLoaded && !directionsServiceRef.current) {
      directionsServiceRef.current = new google.maps.DirectionsService();
      console.log('[MapView] DirectionsService initialized');
    }
  }, [apiLoaded]);

  // Enable Traffic Layer on the map
  useEffect(() => {
    if (map && apiLoaded) {
      const trafficLayer = new google.maps.TrafficLayer();
      trafficLayer.setMap(map);
      console.log('[MapView] 🚦 Traffic layer enabled');

      return () => {
        trafficLayer.setMap(null);
      };
    }
  }, [map, apiLoaded]);

  // Fetch directions and push to store
  useEffect(() => {
    if (!origin || !destination || !directionsServiceRef.current) {
      setRouteSegments([]);
      setGoogleRoute(null);
      return;
    }

    setComputingRoute(true);
    console.log('[MapView] 🚀 Fetching traffic-aware directions...');

    directionsServiceRef.current.route(
      {
        origin: { lat: origin[0], lng: origin[1] },
        destination: { lat: destination[0], lng: destination[1] },
        travelMode: google.maps.TravelMode.DRIVING,
        provideRouteAlternatives: false,
        drivingOptions: {
          departureTime: new Date(),
          trafficModel: google.maps.TrafficModel.BEST_GUESS,
        },
      },
      (result, status) => {
        if (status !== google.maps.DirectionsStatus.OK || !result) {
          console.error('[MapView] ❌ Directions failed:', status);
          setRouteSegments([]);
          setError(`Route calculation failed: ${status}`);
          return;
        }

        console.log('[MapView] ✅ Directions loaded, processing...');

        const route = result.routes[0];
        if (!route || !route.legs || !route.legs[0]) {
          setError('Invalid route data received');
          return;
        }

        const leg = route.legs[0];
        const segments: RouteSegment[] = [];

        // Process each step and color based on traffic
        leg.steps.forEach((step) => {
          const baseDuration = step.duration?.value || 1;
          const overallRatio = (leg.duration_in_traffic?.value || leg.duration?.value || baseDuration) / (leg.duration?.value || baseDuration);
          const estimatedTrafficDuration = baseDuration * overallRatio;
          const color = getTrafficColor(baseDuration, estimatedTrafficDuration);

          segments.push({
            path: step.path || [],
            color,
            isMainRoute: true,
          });
        });

        setRouteSegments(segments);

        // Push real route data to the store for Sidebar to consume
        setGoogleRoute({
          distance: leg.distance?.text || '',
          duration: leg.duration?.text || '',
          durationInTraffic: leg.duration_in_traffic?.text,
          distanceMeters: leg.distance?.value || 0,
          durationSeconds: leg.duration?.value || 0,
          durationInTrafficSeconds: leg.duration_in_traffic?.value,
        });

        // Fetch route-specific forecast from ST-GCN model
        import('@/utils/api').then(({ getRouteForecast }) => {
          import('@/stores/trafficStore').then(({ useTrafficStore }) => {
            const setRouteForecast = useTrafficStore.getState().setRouteForecast;
            getRouteForecast(
              [origin![0], origin![1]],
              [destination![0], destination![1]],
              'bengaluru'
            ).then(forecast => {
              console.log('[MapView] 🔮 Route forecast received:', forecast);
              setRouteForecast(forecast);
            }).catch(err => {
              console.error('[MapView] ❌ Forecast fetch failed:', err);
            });
          });
        });

        // Fit map to route bounds
        if (map && route.bounds) {
          map.fitBounds(route.bounds, 50);
        }

        console.log('[MapView] ✅ Route data pushed to store');
      }
    );
  }, [origin, destination, map, setGoogleRoute, setComputingRoute, setError]);

  const onLoad = useCallback((mapInstance: google.maps.Map) => {
    setMap(mapInstance);
  }, []);

  const handleApiLoad = useCallback(() => {
    console.log('[MapView] ✅ Google Maps API loaded');
    setApiLoaded(true);
  }, []);

  const handleClick = useCallback((e: google.maps.MapMouseEvent) => {
    if (!e.latLng || !pickMode) return;

    const lat = e.latLng.lat();
    const lng = e.latLng.lng();

    console.log('[MapView] 📍 Setting', pickMode, 'to', lat.toFixed(4), lng.toFixed(4));

    if (pickMode === 'origin') {
      setOrigin([lat, lng]);
      setPickMode('destination');
    } else {
      setDestination([lat, lng]);
      setPickMode(null);
    }
  }, [pickMode, setOrigin, setDestination, setPickMode]);

  return (
    <div className="relative w-full h-full">
      <LoadScript googleMapsApiKey={GOOGLE_MAPS_API_KEY} onLoad={handleApiLoad}>
        <GoogleMap
          mapContainerStyle={containerStyle}
          center={center}
          zoom={viewState.zoom}
          options={mapOptions}
          onLoad={onLoad}
          onClick={handleClick}
        >
          {/* Traffic-colored route segments */}
          {routeSegments.map((segment, index) => (
            <Polyline
              key={index}
              path={segment.path}
              options={{
                strokeColor: segment.color,
                strokeOpacity: 1,
                strokeWeight: 6,
                zIndex: 10,
              }}
            />
          ))}

          {/* Route outline for visibility */}
          {routeSegments.length > 0 && (
            <Polyline
              path={routeSegments.flatMap(s => s.path)}
              options={{
                strokeColor: '#1e293b',
                strokeOpacity: 0.3,
                strokeWeight: 10,
                zIndex: 5,
              }}
            />
          )}

          {/* Origin Marker */}
          {origin && (
            <Marker
              position={{ lat: origin[0], lng: origin[1] }}
              icon={{
                url: 'data:image/svg+xml,' + encodeURIComponent(`
                  <svg xmlns="http://www.w3.org/2000/svg" width="44" height="44" viewBox="0 0 24 24">
                    <circle cx="12" cy="12" r="10" fill="#10b981" stroke="white" stroke-width="3"/>
                    <circle cx="12" cy="12" r="4" fill="white"/>
                  </svg>
                `),
                scaledSize: new google.maps.Size(44, 44),
                anchor: new google.maps.Point(22, 22),
              }}
              title="Origin"
              zIndex={100}
            />
          )}

          {/* Destination Marker */}
          {destination && (
            <Marker
              position={{ lat: destination[0], lng: destination[1] }}
              icon={{
                url: 'data:image/svg+xml,' + encodeURIComponent(`
                  <svg xmlns="http://www.w3.org/2000/svg" width="44" height="52" viewBox="0 0 24 30">
                    <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z" fill="#ef4444" stroke="white" stroke-width="2"/>
                    <circle cx="12" cy="9" r="3" fill="white"/>
                  </svg>
                `),
                scaledSize: new google.maps.Size(44, 52),
                anchor: new google.maps.Point(22, 52),
              }}
              title="Destination"
              zIndex={100}
            />
          )}
        </GoogleMap>
      </LoadScript>

      {/* Pick mode indicator */}
      {pickMode && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-gradient-to-r from-green-500 to-emerald-600 text-white px-6 py-3 rounded-xl shadow-lg text-sm z-20 animate-pulse">
          👆 Click on map to set <span className="font-bold uppercase">{pickMode}</span>
        </div>
      )}

      {/* Live Traffic Badge */}
      <div className="absolute top-4 right-20 z-10">
        <div className="bg-white/90 backdrop-blur-sm px-3 py-1.5 rounded-lg shadow-md border border-slate-200">
          <div className="flex items-center gap-2 text-xs font-medium text-slate-600">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            Live Traffic
          </div>
        </div>
      </div>
    </div>
  );
}
