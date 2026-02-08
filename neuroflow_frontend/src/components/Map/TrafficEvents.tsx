import { useEffect, useState, memo, useRef } from 'react';
import { Marker, OverlayView } from '@react-google-maps/api';
import { fetchUpcomingEvents } from '@/utils/api';
import type { Event } from '@/types';
import { useMapStore } from '@/stores/mapStore';

interface TrafficEventsProps {
    map: google.maps.Map | null;
    visible?: boolean;
}

function TrafficEvents({ map, visible = true }: TrafficEventsProps) {
    const viewState = useMapStore((s) => s.viewState);
    const [events, setEvents] = useState<Event[]>([]);
    const [hoveredEvent, setHoveredEvent] = useState<Event | null>(null);
    const lastFetchRef = useRef<{lat: number, lng: number} | null>(null);

    // Initial load only. We don't want to spam the API on drag.
    useEffect(() => {
        if (!visible) return;

        const loadEvents = async () => {
            try {
                // Use current view state as center
                const { latitude, longitude } = viewState;
                console.log("Fetching PredictHQ events for:", latitude, longitude);
                const data = await fetchUpcomingEvents(latitude, longitude, '20km'); // Reasonably large radius
                setEvents(data);
                lastFetchRef.current = { lat: latitude, lng: longitude };
            } catch (err) {
                console.error("Failed to fetch events:", err);
            }
        };
        
        // Only load if we haven't loaded yet or if forced (could add force refresh later)
        if (!lastFetchRef.current) {
            loadEvents();
        }
        
        // Refresh every 10 minutes
        const interval = setInterval(loadEvents, 600000); 
        return () => clearInterval(interval);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [visible]); // Exclude viewState to prevent re-fetch on drag


    if (!visible || !map) return null;

    const getCrowdFlag = (attendance?: number) => {
        if (!attendance) return { label: 'Unknown Crowd', color: '#94a3b8' }; // Gray
        if (attendance > 5000) return { label: 'Heavy Crowd', color: '#ef4444' }; // Red
        if (attendance > 500) return { label: 'Medium Crowd', color: '#f59e0b' };  // Orange
        return { label: 'Low Crowd', color: '#22c55e' };                           // Green
    };

    const getEventIcon = (category: string) => {
        return {
            path: google.maps.SymbolPath.CIRCLE,
            fillColor: '#8b5cf6', // Violet
            fillOpacity: 1,
            strokeWeight: 2,
            strokeColor: 'white',
            scale: 8,
        };
    };

    return (
        <>
            {events.map((event) => (
                <Marker
                    key={event.id}
                    position={{ lat: event.location[1], lng: event.location[0] }}
                    icon={getEventIcon(event.category)}
                    onMouseOver={() => setHoveredEvent(event)}
                    onMouseOut={() => setHoveredEvent(null)}
                    zIndex={60}
                    onClick={() => setHoveredEvent(event)}
                    title={event.title} // Fixes "empty tooltip" by providing actual text
                />
            ))}

            {hoveredEvent && (
                <OverlayView
                    position={{ lat: hoveredEvent.location[1], lng: hoveredEvent.location[0] }}
                    mapPaneName={OverlayView.OVERLAY_MOUSE_TARGET}
                    getPixelPositionOffset={(width, height) => ({ x: -(width / 2), y: -height - 20 })}
                >
                    <div className="bg-white rounded-lg shadow-xl border border-slate-100 p-2 min-w-[200px] max-w-[240px] pointer-events-auto transform transition-all duration-200 origin-bottom">
                         {/* Close Button (top-right absolute) - Optional but good for UX */}
                         <button 
                            onClick={() => setHoveredEvent(null)}
                            className="absolute -top-2 -right-2 bg-slate-100 hover:bg-slate-200 text-slate-500 rounded-full w-5 h-5 flex items-center justify-center text-xs shadow-sm"
                         >
                            ✕
                         </button>

                        <h3 className="font-bold text-slate-800 text-sm mb-1 leading-tight pr-4">{hoveredEvent.title}</h3>
                        
                        <div className="flex flex-wrap gap-1 mb-2 items-center">
                             <span className="text-[10px] px-1.5 py-0.5 rounded text-white font-medium shadow-sm" 
                                   style={{ backgroundColor: getCrowdFlag(hoveredEvent.attendance).color }}>
                                {getCrowdFlag(hoveredEvent.attendance).label}
                             </span>
                             <span className="text-[10px] text-slate-500 bg-slate-50 px-1.5 py-0.5 rounded border border-slate-100">
                                {new Date(hoveredEvent.start).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                             </span>
                        </div>

                        {hoveredEvent.description && (
                            <p className="text-[10px] text-slate-600 mb-2 line-clamp-3 leading-snug bg-slate-50 p-1 rounded">
                                {hoveredEvent.description}
                            </p>
                        )}
                        
                        <div className="flex justify-between items-center text-[9px] text-slate-400 border-t border-slate-100 pt-1 mt-1">
                             <span>Rank: {hoveredEvent.rank}</span>
                             <span className="capitalize px-1 bg-violet-50 text-violet-600 rounded">{hoveredEvent.category}</span>
                        </div>
                        
                        {/* Triangle Arrow */}
                        <div className="absolute left-1/2 -translate-x-1/2 -bottom-1 w-3 h-3 bg-white border-r border-b border-slate-100 transform rotate-45 shadowed-sm"></div>
                    </div>
                </OverlayView>
            )}
        </>
    );
}

export default memo(TrafficEvents);
