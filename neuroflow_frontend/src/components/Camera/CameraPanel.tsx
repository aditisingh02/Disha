/**
 * Camera Panel Component
 * Side panel for viewing traffic camera images in detail
 */

import { useEffect, useState, useCallback } from 'react';
import { X, RefreshCw, MapPin, Clock, Camera, ChevronLeft, ChevronRight, AlertCircle, Car, Truck, Bus, Bike, Activity } from 'lucide-react';
import { useLTAStore, VehicleAnalysis } from '@/stores/ltaStore';
import './CameraPanel.css';

export default function CameraPanel() {
    const {
        cameras,
        selectedCamera,
        isCameraPanelOpen,
        selectCamera,
        toggleCameraPanel,
        fetchCameras,
        analyzeCamera,
        cameraAnalysis,
        isAnalyzing
    } = useLTAStore();

    const [isRefreshing, setIsRefreshing] = useState(false);
    const [imageError, setImageError] = useState(false);
    const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

    // Find current camera index for navigation
    const currentIndex = cameras.findIndex(c => c.id === selectedCamera?.id);
    const hasPrev = currentIndex > 0;
    const hasNext = currentIndex < cameras.length - 1;

    // Refresh camera image
    const handleRefresh = useCallback(async () => {
        setIsRefreshing(true);
        setImageError(false);
        await fetchCameras();
        setLastRefresh(new Date());
        setIsRefreshing(false);
    }, [fetchCameras]);

    // Analyze camera with YOLO
    const handleAnalyze = useCallback(async () => {
        if (selectedCamera) {
            await analyzeCamera(selectedCamera.id);
        }
    }, [selectedCamera, analyzeCamera]);

    // Get current analysis for selected camera
    const currentAnalysis: VehicleAnalysis | null = selectedCamera 
        ? cameraAnalysis[selectedCamera.id] || null 
        : null;

    // Congestion level colors
    const getCongestionColor = (level: string) => {
        switch (level) {
            case 'low': return '#22c55e';
            case 'medium': return '#f59e0b';
            case 'high': return '#ef4444';
            case 'severe': return '#7f1d1d';
            default: return '#6b7280';
        }
    };

    // Navigate to previous/next camera
    const goToPrev = useCallback(() => {
        if (hasPrev) {
            const prevCamera = cameras[currentIndex - 1];
            if (prevCamera) {
                setImageError(false);
                selectCamera(prevCamera);
            }
        }
    }, [hasPrev, currentIndex, cameras, selectCamera]);

    const goToNext = useCallback(() => {
        if (hasNext) {
            const nextCamera = cameras[currentIndex + 1];
            if (nextCamera) {
                setImageError(false);
                selectCamera(nextCamera);
            }
        }
    }, [hasNext, currentIndex, cameras, selectCamera]);

    // Keyboard navigation
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (!isCameraPanelOpen) return;
            if (e.key === 'ArrowLeft') goToPrev();
            if (e.key === 'ArrowRight') goToNext();
            if (e.key === 'Escape') toggleCameraPanel();
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [isCameraPanelOpen, goToPrev, goToNext, toggleCameraPanel]);

    // Reset image error when camera changes
    useEffect(() => {
        setImageError(false);
    }, [selectedCamera?.id]);

    if (!isCameraPanelOpen || !selectedCamera) {
        return null;
    }

    return (
        <div className="camera-panel">
            {/* Header */}
            <div className="camera-panel__header">
                <div className="camera-panel__title">
                    <Camera size={18} />
                    <span>Traffic Camera</span>
                </div>
                <button
                    className="camera-panel__close"
                    onClick={toggleCameraPanel}
                    title="Close (Esc)"
                >
                    <X size={20} />
                </button>
            </div>

            {/* Camera Info */}
            <div className="camera-panel__info">
                <div className="camera-panel__location">
                    <MapPin size={14} />
                    <span>{selectedCamera.description}</span>
                </div>
                <div className="camera-panel__meta">
                    <span className="camera-panel__id">ID: {selectedCamera.id}</span>
                    {lastRefresh && (
                        <span className="camera-panel__time">
                            <Clock size={12} />
                            Refreshed: {lastRefresh.toLocaleTimeString()}
                        </span>
                    )}
                </div>
            </div>

            {/* Image Viewer */}
            <div className="camera-panel__viewer">
                {imageError ? (
                    <div className="camera-panel__error">
                        <AlertCircle size={48} />
                        <h3>Image Unavailable</h3>
                        <p>Camera feed may be temporarily offline</p>
                        <button onClick={handleRefresh} disabled={isRefreshing}>
                            <RefreshCw size={16} className={isRefreshing ? 'spinning' : ''} />
                            Retry
                        </button>
                    </div>
                ) : (
                    <img
                        key={selectedCamera.id + selectedCamera.image_url}
                        src={selectedCamera.image_url}
                        alt={selectedCamera.description}
                        className="camera-panel__image"
                        onError={() => setImageError(true)}
                    />
                )}

                {/* Navigation Arrows */}
                <button
                    className="camera-panel__nav camera-panel__nav--prev"
                    onClick={goToPrev}
                    disabled={!hasPrev}
                    title="Previous camera (←)"
                >
                    <ChevronLeft size={24} />
                </button>
                <button
                    className="camera-panel__nav camera-panel__nav--next"
                    onClick={goToNext}
                    disabled={!hasNext}
                    title="Next camera (→)"
                >
                    <ChevronRight size={24} />
                </button>
            </div>

            {/* Actions */}
            <div className="camera-panel__actions">
                <button
                    className="camera-panel__refresh"
                    onClick={handleRefresh}
                    disabled={isRefreshing}
                >
                    <RefreshCw size={16} className={isRefreshing ? 'spinning' : ''} />
                    {isRefreshing ? 'Refreshing...' : 'Refresh Image'}
                </button>
                <button
                    className="camera-panel__analyze"
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                >
                    <Activity size={16} className={isAnalyzing ? 'spinning' : ''} />
                    {isAnalyzing ? 'Analyzing...' : 'Analyze Traffic'}
                </button>
                <div className="camera-panel__counter">
                    {currentIndex + 1} / {cameras.length}
                </div>
            </div>

            {/* Vehicle Analysis Results */}
            {currentAnalysis && (
                <div className="camera-panel__analysis">
                    <div className="camera-panel__analysis-header">
                        <Activity size={14} />
                        <span>Traffic Analysis</span>
                        <span className="camera-panel__analysis-time">
                            {currentAnalysis.processing_time_ms}ms
                        </span>
                    </div>
                    
                    {/* Congestion Meter */}
                    <div className="camera-panel__congestion">
                        <div className="camera-panel__congestion-label">
                            <span>Congestion</span>
                            <span 
                                className="camera-panel__congestion-level"
                                style={{ color: getCongestionColor(currentAnalysis.congestion_level) }}
                            >
                                {currentAnalysis.congestion_level.toUpperCase()}
                            </span>
                        </div>
                        <div className="camera-panel__congestion-bar">
                            <div 
                                className="camera-panel__congestion-fill"
                                style={{ 
                                    width: `${currentAnalysis.congestion_score}%`,
                                    backgroundColor: getCongestionColor(currentAnalysis.congestion_level)
                                }}
                            />
                        </div>
                        <div className="camera-panel__congestion-score">
                            {currentAnalysis.congestion_score}%
                        </div>
                    </div>

                    {/* Vehicle Counts */}
                    <div className="camera-panel__vehicles">
                        <div className="camera-panel__vehicle-item">
                            <Car size={18} />
                            <span className="camera-panel__vehicle-count">
                                {currentAnalysis.vehicle_counts.car}
                            </span>
                            <span className="camera-panel__vehicle-label">Cars</span>
                        </div>
                        <div className="camera-panel__vehicle-item">
                            <Truck size={18} />
                            <span className="camera-panel__vehicle-count">
                                {currentAnalysis.vehicle_counts.truck}
                            </span>
                            <span className="camera-panel__vehicle-label">Trucks</span>
                        </div>
                        <div className="camera-panel__vehicle-item">
                            <Bus size={18} />
                            <span className="camera-panel__vehicle-count">
                                {currentAnalysis.vehicle_counts.bus}
                            </span>
                            <span className="camera-panel__vehicle-label">Buses</span>
                        </div>
                        <div className="camera-panel__vehicle-item">
                            <Bike size={18} />
                            <span className="camera-panel__vehicle-count">
                                {currentAnalysis.vehicle_counts.motorcycle}
                            </span>
                            <span className="camera-panel__vehicle-label">Bikes</span>
                        </div>
                    </div>

                    <div className="camera-panel__total-vehicles">
                        Total Vehicles: <strong>{currentAnalysis.total_vehicles}</strong>
                    </div>
                </div>
            )}

            {/* Camera List */}
            <div className="camera-panel__list">
                <div className="camera-panel__list-header">
                    <Camera size={14} />
                    <span>All Cameras ({cameras.length})</span>
                </div>
                <div className="camera-panel__list-scroll">
                    {cameras.map((cam, idx) => (
                        <button
                            key={cam.id}
                            className={`camera-panel__list-item ${cam.id === selectedCamera.id ? 'active' : ''}`}
                            onClick={() => {
                                setImageError(false);
                                selectCamera(cam);
                            }}
                        >
                            <span className="camera-panel__list-idx">{idx + 1}</span>
                            <span className="camera-panel__list-desc">{cam.description}</span>
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
}
