/**
 * Traffic Cameras Component for Google Maps
 * Displays LTA traffic cameras as markers on the Google Map
 */

import { useEffect, useCallback, useState, memo } from 'react';
import { Marker, InfoWindow } from '@react-google-maps/api';
import { useLTAStore } from '@/stores/ltaStore';
import type { TrafficCamera } from '@/stores/ltaStore';

interface TrafficCamerasGoogleProps {
    map: google.maps.Map | null;
    visible?: boolean;
}

function TrafficCamerasGoogle({ map, visible = true }: TrafficCamerasGoogleProps) {
    const { cameras, fetchCameras, selectCamera } = useLTAStore();
    const [hoveredCamera, setHoveredCamera] = useState<TrafficCamera | null>(null);
    const [imageError, setImageError] = useState<Record<string, boolean>>({});

    // Fetch cameras on mount
    useEffect(() => {
        fetchCameras();
        // Refresh every 4 minutes (images expire in 5)
        const interval = setInterval(fetchCameras, 240000);
        return () => clearInterval(interval);
    }, [fetchCameras]);

    const handleCameraClick = useCallback((camera: TrafficCamera) => {
        selectCamera(camera);
    }, [selectCamera]);

    const handleImageError = useCallback((cameraId: string) => {
        setImageError(prev => ({ ...prev, [cameraId]: true }));
    }, []);

    if (!visible || cameras.length === 0 || !map) {
        return null;
    }

    // Camera marker icon
    const cameraIcon = {
        url: 'data:image/svg+xml,' + encodeURIComponent(`
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">
                <circle cx="16" cy="16" r="14" fill="#3b82f6" stroke="white" stroke-width="2"/>
                <path d="M10 12h8a2 2 0 012 2v4a2 2 0 01-2 2h-8a2 2 0 01-2-2v-4a2 2 0 012-2z" fill="white"/>
                <path d="M20 13l4-2v8l-4-2" fill="white"/>
            </svg>
        `),
        scaledSize: new google.maps.Size(32, 32),
        anchor: new google.maps.Point(16, 16),
    };

    return (
        <>
            {cameras.map((camera) => (
                <Marker
                    key={camera.id}
                    position={{ lat: camera.latitude, lng: camera.longitude }}
                    icon={cameraIcon}
                    onClick={() => handleCameraClick(camera)}
                    onMouseOver={() => setHoveredCamera(camera)}
                    onMouseOut={() => setHoveredCamera(null)}
                    title={camera.description}
                    zIndex={50}
                />
            ))}

            {/* Hover InfoWindow with Image Preview */}
            {hoveredCamera && (
                <InfoWindow
                    position={{ lat: hoveredCamera.latitude, lng: hoveredCamera.longitude }}
                    onCloseClick={() => setHoveredCamera(null)}
                    options={{
                        pixelOffset: new google.maps.Size(0, -20),
                        disableAutoPan: true,
                    }}
                >
                    <div style={{ minWidth: 250, maxWidth: 300 }}>
                        <div style={{
                            fontWeight: 600,
                            fontSize: 13,
                            marginBottom: 8,
                            color: '#1e293b'
                        }}>
                            📹 {hoveredCamera.description}
                        </div>
                        <div style={{
                            width: '100%',
                            height: 150,
                            background: '#1e293b',
                            borderRadius: 8,
                            overflow: 'hidden',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                        }}>
                            {imageError[hoveredCamera.id] ? (
                                <span style={{ color: '#64748b', fontSize: 12 }}>
                                    Image unavailable
                                </span>
                            ) : (
                                <img
                                    src={hoveredCamera.image_url}
                                    alt={hoveredCamera.description}
                                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                    onError={() => handleImageError(hoveredCamera.id)}
                                    loading="lazy"
                                />
                            )}
                        </div>
                        <div style={{
                            marginTop: 8,
                            fontSize: 11,
                            color: '#64748b',
                            display: 'flex',
                            justifyContent: 'space-between'
                        }}>
                            <span>ID: {hoveredCamera.id}</span>
                            <span style={{ color: '#3b82f6', fontWeight: 500 }}>
                                Click for full view →
                            </span>
                        </div>
                    </div>
                </InfoWindow>
            )}
        </>
    );
}

export default memo(TrafficCamerasGoogle);
