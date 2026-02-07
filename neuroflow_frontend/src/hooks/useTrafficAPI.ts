import { useState, useEffect, useCallback, useRef } from 'react';
import * as api from '@/utils/api';

/** Poll interval for analytics endpoints (ms) */
const POLL_MS = 10_000;
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
