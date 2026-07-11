import { useState, useRef, useEffect, useCallback } from "react";
import { S as Semvec } from "./semvec-iHPOsTC3.js";
function useSearch(data, options = {}) {
  const { debounce = 150, topK = 5, threshold = 0, ...semvecOpts } = options;
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const svRef = useRef(null);
  const timerRef = useRef(null);
  const latestQueryRef = useRef("");
  useEffect(() => {
    const sv = new Semvec(data, semvecOpts);
    svRef.current = sv;
    return () => {
      sv.destroy();
      svRef.current = null;
    };
  }, [data]);
  const search = useCallback(
    (query) => {
      latestQueryRef.current = query;
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
          if (latestQueryRef.current === query) {
            setResults(res);
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
    [debounce, topK, threshold]
  );
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);
  return { search, results, loading, error };
}
export {
  useSearch
};
