import { useState, useCallback, useEffect } from 'react';
import Header from '@/components/Header';
import LoadingScreen from '@/components/LoadingScreen';
import MapView from '@/components/Map/MapView';
import Sidebar from '@/components/Dashboard/Sidebar';
import StatsPanel from '@/components/Dashboard/StatsPanel';
import EmissionCard from '@/components/Dashboard/EmissionCard';
import BraessVisualizer from '@/components/Analytics/BraessVisualizer';
import CorridorStats from '@/components/Analytics/CorridorStats';
import ForecastModal from '@/components/Dashboard/ForecastModal';
import ChatWidget from '@/components/Chat/ChatWidget';
import CameraPanel from '@/components/Camera/CameraPanel';
import { useWebSocket } from '@/hooks/useWebSocket';

export default function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [showForecast, setShowForecast] = useState(false);

  // Connect to live traffic WebSocket
  useWebSocket();

  const handleLoadComplete = useCallback(() => {
    setIsLoading(false);
  }, []);

  // Listen for forecast open event from Sidebar
  useEffect(() => {
    const handleOpen = () => setShowForecast(true);
    window.addEventListener('open-forecast-modal', handleOpen);
    return () => window.removeEventListener('open-forecast-modal', handleOpen);
  }, []);

  if (isLoading) {
    return <LoadingScreen onComplete={handleLoadComplete} />;
  }

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-slate-50">
      <ForecastModal isOpen={showForecast} onClose={() => setShowForecast(false)} />

      {/* 1. Background Map Layer */}
      <div className="absolute inset-0 z-0">
        <MapView />

        {/* Subtle Overlay Gradient for Depth (optional) */}
        <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-white/40 via-transparent to-transparent z-[1]" />
      </div>

      {/* 2. UI Overlay Layer - Pass through clicks to map, but catch on children */}
      <div className="relative z-10 h-full flex flex-col pointer-events-none">

        {/* Floating Header */}
        <div className="pointer-events-auto">
          <Header />
        </div>

        {/* Main Workspace */}
        <div className="flex flex-1 overflow-hidden p-6 gap-6">

          {/* Left Panel - Floating Glass Sidebar */}
          <div className="pointer-events-auto h-full flex flex-col justify-center">
            <Sidebar />
          </div>

          {/* Spacer to push panels apart */}
          <div className="flex-1" />

          {/* Right Panel - Analytics Stack */}
          <aside className="pointer-events-auto w-80 h-full overflow-y-auto flex flex-col gap-4 no-scrollbar pb-20">
            <div className="animate-enter delay-100">
              <StatsPanel />
            </div>
            <div className="animate-enter delay-200">
              <EmissionCard />
            </div>
            <div className="animate-enter delay-300">
              <CorridorStats />
            </div>
            <div className="animate-enter delay-400">
              <BraessVisualizer />
            </div>
          </aside>
        </div>
      </div>

      {/* Branding Overlay (Bottom Left) */}
      <div className="absolute bottom-6 left-6 z-20 pointer-events-none">
        <div className="glass-panel px-4 py-2 flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
            Live System Active • 12ms Latency
          </p>
        </div>
      </div>

      {/* Chat Widget - Bottom Right */}
      <ChatWidget />

      {/* LTA Camera Panel - Right Sidebar */}
      <CameraPanel />
    </div>
  );
}
