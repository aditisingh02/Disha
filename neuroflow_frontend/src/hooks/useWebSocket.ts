import { useEffect, useRef, useCallback } from 'react';
import { useTrafficStore } from '@/stores/trafficStore';
import { WS_URL } from '@/utils/constants';
import type { WSTrafficUpdate } from '@/types';

const RECONNECT_BASE_MS = 2000;
const RECONNECT_MAX_MS = 30_000;

/**
 * WebSocket hook – connects to the live traffic feed and
 * pushes updates into the Zustand traffic store.
 * Uses exponential backoff for reconnection.
 */
export function useWebSocket(): void {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const backoffRef = useRef(RECONNECT_BASE_MS);
  const updateAll = useTrafficStore((s) => s.updateAll);
  const setConnected = useTrafficStore((s) => s.setConnected);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('[WS] Connected to live traffic feed');
        setConnected(true);
        backoffRef.current = RECONNECT_BASE_MS; // reset on success
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as WSTrafficUpdate;
          if (data.event === 'traffic_update' || data.type === 'traffic_update') {
            updateAll(data);
          }
        } catch {
          // ignore parse errors
        }
      };

      ws.onclose = () => {
        setConnected(false);
        const delay = backoffRef.current;
        backoffRef.current = Math.min(delay * 2, RECONNECT_MAX_MS);
        console.log(`[WS] Disconnected — reconnecting in ${(delay / 1000).toFixed(0)}s...`);
        reconnectTimer.current = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch {
      // WebSocket constructor can throw if URL is invalid
      const delay = backoffRef.current;
      backoffRef.current = Math.min(delay * 2, RECONNECT_MAX_MS);
      reconnectTimer.current = setTimeout(connect, delay);
    }
  }, [updateAll, setConnected]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);
}
