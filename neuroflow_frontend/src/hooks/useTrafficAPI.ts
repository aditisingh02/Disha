import { useState, useEffect, useCallback, useRef } from 'react';
import * as api from '@/utils/api';
import { useTrafficStore } from '@/stores/trafficStore';
import type { TrafficReading } from '@/types';

/** Poll interval for analytics endpoints (ms) */
const POLL_MS = 10_000;
/** Poll interval for live traffic/predictions (ms) */
const LIVE_POLL_MS = 8_000;
/** Initial delay before first fetch — let backend finish booting (ms) */
const INITIAL_DELAY_MS = 2000;

function usePollAPI<T>(
  fetcher: () => Promise<T>,
  interval: number = POLL_MS,
) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const failCount = useRef(0);

  const refetch = useCallback(async () => {
    try {
      const result = await fetcher();
      setData(result);
      setError(null);
      failCount.current = 0;
    } catch (e) {
      failCount.current += 1;
      // Only show error after 3 consecutive failures (backend may still be starting)
      if (failCount.current > 2) {
        setError(e instanceof Error ? e.message : 'Failed');
      }
    }
  }, [fetcher]);

  useEffect(() => {
    // Initial delay so backend has time to start
    const initTimer = setTimeout(() => {
      refetch();
      // Then poll on interval
    }, INITIAL_DELAY_MS);

    const pollTimer = setInterval(refetch, interval);

    return () => {
      clearTimeout(initTimer);
      clearInterval(pollTimer);
    };
  }, [refetch, interval]);

  return { data, error, refetch };
}

export function useCorridorStats() {
  return usePollAPI(api.getCorridorStats, POLL_MS);
}

export function useEmissionSavings() {
  return usePollAPI(api.getEmissionSavings, POLL_MS);
}

export function useBraessParadox() {
  return usePollAPI(api.getBraessParadox, POLL_MS);
}

export function useSystemHealth() {
  return usePollAPI(api.getSystemHealth, 5000);
}

/** Consider "live" when WebSocket is connected OR REST data was updated in last 35s (dynamic dashboard without WS) */
export function useDataLive(): boolean {
  const isConnected = useTrafficStore((s) => s.isConnected);
  const lastUpdate = useTrafficStore((s) => s.lastUpdate);
  const [live, setLive] = useState(false);
  useEffect(() => {
    const check = () => {
      if (isConnected) {
        setLive(true);
        return;
      }
      if (lastUpdate) {
        setLive(Date.now() - new Date(lastUpdate).getTime() < 35000);
        return;
      }
      setLive(false);
    };
    check();
    const t = setInterval(check, 5000);
    return () => clearInterval(t);
  }, [isConnected, lastUpdate]);
  return live;
}

/**
 * Polls /predict/traffic and /traffic/live and pushes results into the traffic store.
 * Ensures StatsPanel, map layers, and footer see dynamic values from the backend.
 */
export function useLiveDataPoll(): void {
  const setPredictions = useTrafficStore((s) => s.setPredictions);
  const setReadings = useTrafficStore((s) => s.setReadings);
  const setLastUpdate = useTrafficStore((s) => s.setLastUpdate);

  useEffect(() => {
    let mounted = true;
    const poll = async () => {
      if (!mounted) return;
      try {
        const [predRes, liveRes] = await Promise.all([
          api.getTrafficPredictionsResponse().catch(() => null),
          api.getLiveReadings().catch(() => null),
        ]);
        if (!mounted) return;
        if (predRes?.predictions?.length) {
          setPredictions(predRes.predictions);
          setLastUpdate(predRes.timestamp ?? new Date().toISOString());
        }
        if (liveRes?.readings?.length) {
          setReadings(liveRes.readings as unknown as TrafficReading[]);
          if (!predRes?.timestamp) setLastUpdate(liveRes.timestamp ?? new Date().toISOString());
        }
      } catch {
        // ignore
      }
    };
    const t0 = setTimeout(poll, INITIAL_DELAY_MS);
    const interval = setInterval(poll, LIVE_POLL_MS);
    return () => {
      mounted = false;
      clearTimeout(t0);
      clearInterval(interval);
    };
  }, [setPredictions, setReadings, setLastUpdate]);
}
