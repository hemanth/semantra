/**
 * IndexedDB persistence layer for embedding cache.
 * - Namespace-isolated: different Semvec instances / models never collide.
 * - Best-effort: silently falls back to in-memory Map if IndexedDB is unavailable.
 * - Content-hash keyed: same text → same hash → skip re-embedding.
 */

import type { CacheEntry } from './types.js';

const STORE_NAME = 'embeddings';
const DB_VERSION = 1;

/** Simple FNV-1a hash for content-based dedup. Fast and good enough for cache keys. */
export function contentHash(text: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = (hash * 0x01000193) >>> 0;
  }
  return 'sv_' + hash.toString(36);
}

/** Store interface — implemented by both IndexedDB and in-memory fallback. */
export interface EmbeddingStore {
  get(hash: string): Promise<CacheEntry | undefined>;
  getMany(hashes: string[]): Promise<Map<string, CacheEntry>>;
  put(entry: CacheEntry): Promise<void>;
  putMany(entries: CacheEntry[]): Promise<void>;
  delete(hash: string): Promise<void>;
  clear(): Promise<void>;
}

/**
 * Create a store for the given namespace and model.
 * Attempts IndexedDB first, falls back to in-memory Map.
 */
export async function createStore(
  namespace: string,
  model: string,
): Promise<EmbeddingStore> {
  // Sanitize for IndexedDB database name
  const dbName = `semantra_${namespace}_${model.replace(/[^a-zA-Z0-9]/g, '_')}`;

  try {
    return await createIDBStore(dbName);
  } catch {
    // IndexedDB unavailable (private browsing, SSR, etc.) — fall back silently
    return createMemoryStore();
  }
}

/** IndexedDB-backed store. */
async function createIDBStore(dbName: string): Promise<EmbeddingStore> {
  const db = await openDB(dbName);

  return {
    async get(hash) {
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readonly');
        const req = tx.objectStore(STORE_NAME).get(hash);
        req.onsuccess = () => resolve(req.result as CacheEntry | undefined);
        req.onerror = () => reject(req.error);
      });
    },

    async getMany(hashes) {
      const map = new Map<string, CacheEntry>();
      if (hashes.length === 0) return map;

      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);
        let completed = 0;

        for (const hash of hashes) {
          const req = store.get(hash);
          req.onsuccess = () => {
            if (req.result) map.set(hash, req.result as CacheEntry);
            completed++;
            if (completed === hashes.length) resolve(map);
          };
          req.onerror = () => reject(req.error);
        }
      });
    },

    async put(entry) {
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const req = tx.objectStore(STORE_NAME).put(entry);
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
      });
    },

    async putMany(entries) {
      if (entries.length === 0) return;
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        for (const entry of entries) {
          store.put(entry);
        }
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
    },

    async delete(hash) {
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const req = tx.objectStore(STORE_NAME).delete(hash);
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
      });
    },

    async clear() {
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const req = tx.objectStore(STORE_NAME).clear();
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
      });
    },
  };
}

/** Open (or create) an IndexedDB database. */
function openDB(dbName: string): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(dbName, DB_VERSION);

    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'hash' });
      }
    };

    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/** In-memory fallback store (used when IndexedDB is unavailable). */
function createMemoryStore(): EmbeddingStore {
  const map = new Map<string, CacheEntry>();

  return {
    async get(hash) {
      return map.get(hash);
    },

    async getMany(hashes) {
      const result = new Map<string, CacheEntry>();
      for (const hash of hashes) {
        const entry = map.get(hash);
        if (entry) result.set(hash, entry);
      }
      return result;
    },

    async put(entry) {
      map.set(entry.hash, entry);
    },

    async putMany(entries) {
      for (const entry of entries) {
        map.set(entry.hash, entry);
      }
    },

    async delete(hash) {
      map.delete(hash);
    },

    async clear() {
      map.clear();
    },
  };
}
