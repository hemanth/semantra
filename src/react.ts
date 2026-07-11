/**
 * React integration for semantra.
 * Import from 'semantra/react' — tree-shakeable, only loaded if you use it.
 *
 * @example
 * ```tsx
 * import { useSearch } from 'semantra/react';
 *
 * function SearchBox() {
 *   const { search, results, loading } = useSearch(faqDocs);
 *
 *   return (
 *     <div>
 *       <input onChange={e => search(e.target.value)} />
 *       {loading && <span>Searching...</span>}
 *       {results.map(r => <p key={r.index}>{r.text} — {r.score}</p>)}
 *     </div>
 *   );
 * }
 * ```
 *
 * @packageDocumentation
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { Semvec } from './semvec.js';
import type { SearchResult, SemvecOptions } from './types.js';

/** Options for the useSearch hook. */
interface UseSearchOptions extends SemvecOptions {
  /** Debounce delay in ms for search-as-you-type. @default 150 */
  debounce?: number;
  /** Maximum results to return. @default 5 */
  topK?: number;
  /** Minimum similarity score. @default 0 */
  threshold?: number;
}

/** Return value of the useSearch hook. */
interface UseSearchReturn<T> {
  /** Trigger a search. Debounced by default. */
  search: (query: string) => void;
  /** Current search results. */
  results: SearchResult<T>[];
  /** Whether a search is in progress. */
  loading: boolean;
  /** Any error from the last operation. */
  error: Error | null;
}

/**
 * React hook for semantic search over a corpus.
 * Handles initialization, debouncing, and cleanup automatically.
 *
 * @param data - Array of strings or objects to search over.
 * @param options - Search and Semvec options.
 * @returns Object with search function, results, loading state, and errors.
 */
export function useSearch<T>(
  data: T[],
  options: UseSearchOptions = {},
): UseSearchReturn<T> {
  const { debounce = 150, topK = 5, threshold = 0, ...semvecOpts } = options;

  const [results, setResults] = useState<SearchResult<T>[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const svRef = useRef<Semvec | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestQueryRef = useRef<string>('');

  // Create/recreate Semvec instance when data changes
  useEffect(() => {
    const sv = new Semvec(data as unknown[], semvecOpts);
    svRef.current = sv;

    return () => {
      sv.destroy();
      svRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  const search = useCallback(
    (query: string) => {
      latestQueryRef.current = query;

      // Clear previous debounce timer
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }

      if (!query.trim()) {
        setResults([]);
        setLoading(false);
        return;
      }

      setLoading(true);

      timerRef.current = setTimeout(async () => {
        try {
          const sv = svRef.current;
          if (!sv) return;

          const res = await sv.search(query, { topK, threshold });

          // Only update if this is still the latest query
          if (latestQueryRef.current === query) {
            setResults(res as SearchResult<T>[]);
            setError(null);
          }
        } catch (err) {
          if (latestQueryRef.current === query) {
            setError(err instanceof Error ? err : new Error(String(err)));
          }
        } finally {
          if (latestQueryRef.current === query) {
            setLoading(false);
          }
        }
      }, debounce);
    },
    [debounce, topK, threshold],
  );

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  return { search, results, loading, error };
}
