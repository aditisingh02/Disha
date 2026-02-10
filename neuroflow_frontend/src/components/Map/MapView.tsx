import { useCallback, useState, useMemo, useEffect, useRef } from 'react';
import { GoogleMap, LoadScript, Marker, Polyline, OverlayView } from '@react-google-maps/api';
import { useMapStore } from '@/stores/mapStore';
import { useRouteStore } from '@/stores/routeStore';
import { GOOGLE_MAPS_API_KEY } from '@/utils/constants';
import TrafficCameras from './TrafficCameras';
import TrafficEvents from './TrafficEvents';
import { SINGAPORE_LOCATIONS } from '@/utils/locations';

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
  routeIndex: number;
  durationText?: string;
}

export default function MapView() {
  // ... (store hooks unchanged) ...
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
  const [selectedRouteInfo, setSelectedRouteInfo] = useState<{ 
    routeIndex: number; 
    latLng: google.maps.LatLng; 
    duration: string;
    isMain: boolean;
  } | null>(null);
  const [apiLoaded, setApiLoaded] = useState(false);
  const directionsServiceRef = useRef<google.maps.DirectionsService | null>(null);
  const clickLockRef = useRef(false); // Prevents multiple overlapping click handlers

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

        // Google Maps Directions API: departureTime (now or future) + trafficModel required for duration_in_traffic
        const now = new Date();
        directionsServiceRef.current.route(
          {
            origin: { lat: origin[0], lng: origin[1] },
            destination: { lat: destination[0], lng: destination[1] },
            travelMode: google.maps.TravelMode.DRIVING,
            provideRouteAlternatives: true, // Enable alternative routes
            drivingOptions: {
              departureTime: now,
              trafficModel: google.maps.TrafficModel.BEST_GUESS,
            },
          },
          (result, status) => {
            // Always clear computing state so dashboard and Sidebar stay in sync (Google Maps API compliance)
            const clearComputing = () => {
              setComputingRoute(false);
            };

            if (status !== google.maps.DirectionsStatus.OK || !result) {
              console.error('[MapView] ❌ Directions failed:', status);
              setRouteSegments([]);
              setGoogleRoute(null);
              setError(`Route calculation failed: ${status}`);
              clearComputing();
              return;
            }

            console.log(`[MapView] ✅ Directions loaded. Found ${result.routes.length} routes.`);

            // Process ALL routes (Main + Alternatives)
            const segments: RouteSegment[] = [];

            // Limit to Top 3 Routes (Main + 2 Alternatives) to avoid clutter
            result.routes.slice(0, 3).forEach((route, routeIndex) => {
                if (!route.legs || !route.legs[0]) return;
                
                const leg = route.legs[0];
                const isMain = routeIndex === 0; // First route is main
                const durationText = leg.duration?.text || '';

                if (isMain) {
                    // Main Route: Keep granular steps for Traffic Colors
                    leg.steps.forEach((step) => {
                      const baseDuration = step.duration?.value || 1;
                      const overallRatio = (leg.duration_in_traffic?.value || leg.duration?.value || baseDuration) / (leg.duration?.value || baseDuration);
                      const estimatedTrafficDuration = baseDuration * overallRatio;
                      
                      segments.push({
                        path: step.path || [],
                        color: getTrafficColor(baseDuration, estimatedTrafficDuration),
                        isMainRoute: true,
                        routeIndex,
                        durationText,
                      });
                    });
                } else {
                    // Alternative Route: Merge all steps into SINGLE path for cleaner rendering
                    const fullPath: google.maps.LatLng[] = [];
                    leg.steps.forEach((step) => {
                        if (step.path) fullPath.push(...step.path);
                    });
                    
                    segments.push({
                        path: fullPath,
                        color: '#4b5563', // Dark Grey default
                        isMainRoute: false,
                        routeIndex,
                        durationText,
                    });
                }
            });

            // Sort segments so main route is drawn LAST (on top)
            segments.sort((a, b) => (a.isMainRoute === b.isMainRoute ? 0 : a.isMainRoute ? 1 : -1));

            setRouteSegments(segments);

            // Push PRIMARY route data to store for statistics
            const primaryLeg = result.routes[0].legs[0];
            setGoogleRoute({
              distance: primaryLeg.distance?.text || '',
              duration: primaryLeg.duration?.text || '',
              durationInTraffic: primaryLeg.duration_in_traffic?.text,
              distanceMeters: primaryLeg.distance?.value || 0,
              durationSeconds: primaryLeg.duration?.value || 0,
              durationInTrafficSeconds: primaryLeg.duration_in_traffic?.value,
            });

            // Fetch route-specific forecast from ST-GCN model
            import('@/utils/api').then(({ getRouteForecast, getOrchestratorRouteForecast }) => {
              import('@/stores/trafficStore').then(({ useTrafficStore }) => {
                const setRouteForecast = useTrafficStore.getState().setRouteForecast;
                const setOrchestratorRouteForecast = useTrafficStore.getState().setOrchestratorRouteForecast;
                getRouteForecast(
                  [origin![0], origin![1]],
                  [destination![0], destination![1]],
                  'singapore'
                ).then(forecast => {
                  console.log('[MapView] 🔮 Route forecast received:', forecast);
                  setRouteForecast(forecast);
                }).catch(err => {
                  console.error('[MapView] ❌ Forecast fetch failed:', err);
                });
                // Same output as terminal CLI: multi-horizon, congestion level, risk, routes (for Forecast modal)
                getOrchestratorRouteForecast({
                  origin_lat: origin![0],
                  origin_lon: origin![1],
                  destination_lat: destination![0],
                  destination_lon: destination![1],
                }).then(orch => {
                  if (!orch.error) {
                    console.log('[MapView] 📊 Orchestrator route forecast (terminal-equivalent):', orch);
                    setOrchestratorRouteForecast(orch);
                  } else {
                    setOrchestratorRouteForecast(null);
                  }
                }).catch(() => setOrchestratorRouteForecast(null));
              });
            });

            // Fit map to route bounds
            if (map && result.routes[0].bounds) {
              map.fitBounds(result.routes[0].bounds, 50);
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

      // Handler for clicking on a route to show/hide tooltip
      const handleRouteClick = useCallback((e: google.maps.MapMouseEvent, segment: RouteSegment, segmentKey: string) => {
        // Stop event propagation
        if (e.stop) e.stop();
        
        // Prevent multiple overlapping segment clicks (critical fix for double tooltip)
        if (clickLockRef.current) return;
        clickLockRef.current = true;
        setTimeout(() => { clickLockRef.current = false; }, 100);
        
        if (!e.latLng) return;
        
        // Toggle: if clicking the same route type (main/alt), close tooltip
        const isSameRouteType = selectedRouteInfo?.isMain === segment.isMainRoute && 
                                 selectedRouteInfo?.routeIndex === segment.routeIndex;
        
        if (isSameRouteType) {
          setSelectedRouteInfo(null);
        } else {
          setSelectedRouteInfo({
            routeIndex: segment.routeIndex,
            latLng: e.latLng,
            duration: segment.durationText || 'Calculating...',
            isMain: segment.isMainRoute,
          });
        }
      }, [selectedRouteInfo]);
    
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
          
          {/* Alternative Routes (Grey, Click for Tooltip) */}
          {routeSegments.filter(s => !s.isMainRoute).map((segment, index) => {
            const isSelected = selectedRouteInfo?.routeIndex === segment.routeIndex && !selectedRouteInfo?.isMain;
            const segmentKey = `alt-${segment.routeIndex}`;
            return (
            <Polyline
              key={segmentKey}
              path={segment.path}
              onClick={(e) => handleRouteClick(e, segment, segmentKey)}
              options={{
                strokeColor: isSelected ? '#3b82f6' : '#6b7280',
                strokeOpacity: isSelected ? 1 : 0.6,
                strokeWeight: isSelected ? 7 : 4,
                zIndex: isSelected ? 20 : 5,
                clickable: true,
                cursor: 'pointer',
              }}
            />
            );
          })}

          {/* Main Route Casing (White Outline for visibility) */}
          {routeSegments.filter(s => s.isMainRoute).map((segment, index) => (
             <Polyline
              key={`main-casing-${index}`}
              path={segment.path}
              options={{
                strokeColor: '#ffffff',
                strokeOpacity: 0.9,
                strokeWeight: 10,
                zIndex: 10,
                clickable: false,
              }}
            />
          ))}

          {/* Main Route Traffic (Actual Traffic Colors, Clickable) */}
          {(() => {
            const mainSegments = routeSegments.filter(s => s.isMainRoute);
            if (mainSegments.length === 0) return null;
            
            const isSelected = selectedRouteInfo?.isMain === true;
            const firstSegment = mainSegments[0]; // Use first segment for duration data
            
            return mainSegments.map((segment, index) => (
              <Polyline
                key={`main-traffic-${index}`}
                path={segment.path}
                onClick={(e) => handleRouteClick(e, firstSegment, 'main-route')}
                options={{
                  strokeColor: segment.color, // Keep individual segment colors
                  strokeOpacity: 1,
                  strokeWeight: isSelected ? 8 : 6,
                  zIndex: 11,
                  clickable: true,
                  cursor: 'pointer',
                }}
              />
            ));
          })()}

          {/* Custom Tooltip for Clicked Route */}
          {selectedRouteInfo && (
              <OverlayView
                  position={selectedRouteInfo.latLng}
                  mapPaneName={OverlayView.OVERLAY_MOUSE_TARGET}
                  getPixelPositionOffset={(width, height) => ({ x: -(width / 2), y: -height - 20 })}
              >
                  <div 
                    style={{
                      position: 'relative',
                      backgroundColor: 'white',
                      padding: '12px 16px',
                      borderRadius: '12px',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                      minWidth: '140px',
                      maxWidth: '200px',
                      pointerEvents: 'auto',
                    }}
                  >
                    {/* Close Button */}
                    <button
                      onClick={() => setSelectedRouteInfo(null)}
                      style={{
                        position: 'absolute',
                        top: '4px',
                        right: '4px',
                        background: 'transparent',
                        border: 'none',
                        fontSize: '20px',
                        cursor: 'pointer',
                        color: '#64748b',
                        padding: '0',
                        width: '24px',
                        height: '24px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                      aria-label="Close"
                    >
                      ×
                    </button>

                    {/* Duration */}
                    <div style={{ 
                      fontWeight: 'bold', 
                      fontSize: '18px', 
                      color: '#0f172a', 
                      marginBottom: '8px',
                      paddingRight: '20px',
                    }}>
                      ⏱️ {selectedRouteInfo.duration}
                    </div>
                    
                    {/* Route Type Badge */}
                    <div style={{ 
                      fontSize: '11px', 
                      color: selectedRouteInfo.isMain ? '#059669' : '#6366f1', 
                      fontWeight: '600',
                      textTransform: 'uppercase', 
                      letterSpacing: '0.05em',
                      padding: '4px 10px',
                      borderRadius: '6px',
                      backgroundColor: selectedRouteInfo.isMain ? '#d1fae5' : '#e0e7ff',
                      display: 'inline-block',
                    }}>
                      {selectedRouteInfo.isMain ? '✓ Primary Route' : 'Alternative Route'}
                    </div>

                    {/* Arrow pointing down */}
                    <div style={{
                      position: 'absolute',
                      bottom: '-8px',
                      left: '50%',
                      transform: 'translateX(-50%)',
                      width: 0,
                      height: 0,
                      borderLeft: '8px solid transparent',
                      borderRight: '8px solid transparent',
                      borderTop: '8px solid white',
                    }} />
                  </div>
              </OverlayView>
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

          {/* LTA Traffic Cameras */}
          <TrafficCameras map={map} visible={true} />

          {/* PredictHQ Events */}
          <TrafficEvents map={map} visible={true} />

          {/* Singapore Significant Places */}
          {apiLoaded && SINGAPORE_LOCATIONS.map((loc, idx) => (
            <Marker
              key={idx}
              position={{ lat: loc.lat, lng: loc.lng }}
              label={{
                text: loc.label || loc.name,
                color: 'white',
                fontWeight: 'bold',
                fontSize: '12px',
                className: 'bg-slate-800/80 px-2 py-1 rounded-md'
              }}
              icon={{
                path: google.maps.SymbolPath.CIRCLE,
                scale: 12,
                fillColor: loc.label === 'START' ? '#10b981' : '#3b82f6',
                fillOpacity: 1,
                strokeColor: 'white',
                strokeWeight: 2,
              }}
              onClick={() => {
                if(loc.label === 'START') {
                   setOrigin([loc.lat, loc.lng]);
                   setPickMode('destination');
                } else if(loc.label === 'END') {
                   setDestination([loc.lat, loc.lng]);
                   setPickMode(null);
                } else {
                   // Fallback logic
                   if (pickMode === 'origin') {
                      setOrigin([loc.lat, loc.lng]);
                   } else {
                      setDestination([loc.lat, loc.lng]);
                   }
                }
              }}
            />
          ))}
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
