/**
 * Traffic Cameras Component for Google Maps
 * Displays LTA traffic cameras as markers on the Google Map
 */

import { useEffect, useCallback, useState, memo, useRef, useMemo } from 'react';
import { Marker, InfoWindow } from '@react-google-maps/api';
import { useLTAStore } from '@/stores/ltaStore';
import type { TrafficCamera } from '@/stores/ltaStore';

interface TrafficCamerasGoogleProps {
    map: google.maps.Map | null;
    visible?: boolean;
}

function TrafficCamerasGoogle({ map, visible = true }: TrafficCamerasGoogleProps) {
    const { cameras, fetchCameras, selectCamera, analyzeCamera, cameraAnalysis } = useLTAStore();
    const [hoveredCamera, setHoveredCamera] = useState<TrafficCamera | null>(null);
    const [imageError, setImageError] = useState<Record<string, boolean>>({});
    
    // Timer for debounce to prevent flickering/ghosting
    const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    // Fetch cameras on mount
    useEffect(() => {
        fetchCameras();
        // Refresh every 4 minutes (images expire in 5)
        const interval = setInterval(fetchCameras, 240000);
        return () => clearInterval(interval);
    }, [fetchCameras]);
    
    // Clear timeout on unmount
    useEffect(() => {
        return () => {
            if (hoverTimeoutRef.current) {
                clearTimeout(hoverTimeoutRef.current);
            }
        };
    }, []);

    // Analyze on hover
    useEffect(() => {
        if (hoveredCamera && !cameraAnalysis[hoveredCamera.id]) {
            analyzeCamera(hoveredCamera.id);
        }
    }, [hoveredCamera, cameraAnalysis, analyzeCamera]);

    const handleCameraClick = useCallback((camera: TrafficCamera) => {
        selectCamera(camera);
    }, [selectCamera]);

    const handleMouseOver = useCallback((camera: TrafficCamera) => {
        if (hoverTimeoutRef.current) {
            clearTimeout(hoverTimeoutRef.current);
            hoverTimeoutRef.current = null;
        }
        setHoveredCamera(camera);
    }, []);

    const handleMouseOut = useCallback(() => {
        hoverTimeoutRef.current = setTimeout(() => {
            setHoveredCamera(null);
        }, 100);
    }, []);

    const handleInfoWindowClose = useCallback(() => {
        setHoveredCamera(null);
    }, []);

    const handleImageError = useCallback((cameraId: string) => {
        setImageError(prev => ({ ...prev, [cameraId]: true }));
    }, []);

    // Memoize InfoWindow options to prevent re-creation
    const infoWindowOptions = useMemo(() => ({
        pixelOffset: new google.maps.Size(0, -20),
        disableAutoPan: true,
    }), []);

    // Get analysis for hovered camera
    const analysis = hoveredCamera ? cameraAnalysis[hoveredCamera.id] : null;

    // Congestion color helper
    const getCongestionColor = (level: string) => {
        switch (level) {
            case 'low': return '#22c55e';
            case 'medium': return '#f59e0b';
            case 'high': return '#ef4444';
            case 'severe': return '#7f1d1d';
            default: return '#6b7280';
        }
    };

    if (!visible || cameras.length === 0 || !map) {
        return null;
    }

    // Camera marker icon
    const cameraIcon = {
        url: 'data:image/svg+xml,' + encodeURIComponent(`
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">
                <circle cx="16" cy="16" r="14" fill="#3b82f6" stroke="white" stroke-width="2"/>
                <path d="M10 12h8a2 2 0 012 2v4a2 2 0 01-2 2h-8a2 2 0 01-2 2h-8a2 2 0 01-2 2v-4a2 2 0 012-2z" fill="white"/>
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
                    onMouseOver={() => handleMouseOver(camera)}
                    onMouseOut={handleMouseOut}
                    zIndex={50}
                />
            ))}

            {/* Hover InfoWindow with Image Preview & Analysis */}
            {hoveredCamera && (
                <InfoWindow
                    position={{ lat: hoveredCamera.latitude, lng: hoveredCamera.longitude }}
                    onCloseClick={handleInfoWindowClose}
                    options={infoWindowOptions}
                >
                    <div style={{ minWidth: 260, maxWidth: 300, fontFamily: 'Inter, sans-serif' }}>
                        {/* Header */}
                        <div style={{
                            fontWeight: 600,
                            fontSize: 13,
                            marginBottom: 8,
                            color: '#1e293b',
                            display: 'flex',
                            alignItems: 'center',
                            gap: 6
                        }}>
                            <span>📹</span>
                            <span style={{ flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                {hoveredCamera.description}
                            </span>
                        </div>

                        {/* Image */}
                        <div style={{
                            width: '100%',
                            height: 150,
                            background: '#1e293b',
                            borderRadius: 8,
                            overflow: 'hidden',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            marginBottom: 8,
                            position: 'relative'
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
                            
                            {/* Congestion Badge Overlay */}
                            {analysis && (
                                <div style={{
                                    position: 'absolute',
                                    top: 8,
                                    right: 8,
                                    background: getCongestionColor(analysis.congestion_level),
                                    color: 'white',
                                    padding: '4px 8px',
                                    borderRadius: 4,
                                    fontSize: 10,
                                    fontWeight: 700,
                                    textTransform: 'uppercase',
                                    boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                                }}>
                                    {analysis.congestion_level} Traffic
                                </div>
                            )}
                        </div>

                        {/* Analysis Results or Loading */}
                        <div style={{
                            background: '#f1f5f9',
                            borderRadius: 6,
                            padding: 8,
                            marginBottom: 8
                        }}>
                            {analysis ? (
                                <div>
                                    <div style={{ 
                                        display: 'flex', 
                                        justifyContent: 'space-between',
                                        marginBottom: 6,
                                        fontSize: 12,
                                        fontWeight: 600,
                                        color: '#334155'
                                    }}>
                                        <span>Congestion Score</span>
                                        <span style={{ color: getCongestionColor(analysis.congestion_level) }}>
                                            {analysis.congestion_score}%
                                        </span>
                                    </div>
                                    <div style={{ 
                                        display: 'flex', 
                                        gap: 8,
                                        fontSize: 11,
                                        color: '#64748b'
                                    }}>
                                        <span title="Cars">🚗 {analysis.vehicle_counts.car}</span>
                                        <span title="Trucks">🚛 {analysis.vehicle_counts.truck}</span>
                                        <span title="Buses">🚌 {analysis.vehicle_counts.bus}</span>
                                        <span title="Bikes">🏍️ {analysis.vehicle_counts.motorcycle}</span>
                                    </div>
                                </div>
                            ) : (
                                <div style={{ 
                                    display: 'flex', 
                                    alignItems: 'center', 
                                    gap: 8,
                                    color: '#64748b',
                                    fontSize: 11
                                }}>
                                    <div className="spinner" style={{
                                        width: 12,
                                        height: 12,
                                        border: '2px solid #cbd5e1',
                                        borderTopColor: '#3b82f6',
                                        borderRadius: '50%',
                                        animation: 'spin 1s linear infinite'
                                    }}/>
                                    Analyzing traffic...
                                </div>
                            )}
                        </div>

                        {/* Footer */}
                        <div style={{
                            fontSize: 10,
                            color: '#94a3b8',
                            display: 'flex',
                            justifyContent: 'space-between'
                        }}>
                            <span>ID: {hoveredCamera.id}</span>
                            <span style={{ color: '#3b82f6', cursor: 'pointer' }} onClick={() => handleCameraClick(hoveredCamera)}>
                                View Details →
                            </span>
                        </div>
                    </div>
                </InfoWindow>
            )}
        </>
    );
}

export default memo(TrafficCamerasGoogle);
