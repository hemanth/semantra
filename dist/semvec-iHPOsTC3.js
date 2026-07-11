class SemvecError extends Error {
  code;
  detail;
  constructor(code, message, detail) {
    super(message);
    this.name = "SemvecError";
    this.code = code;
    this.detail = detail;
  }
}
const DEFAULT_MODEL = "Snowflake/snowflake-arctic-embed-s";
const DEFAULT_DTYPE = "q8";
const engines = /* @__PURE__ */ new Map();
function engineKey(model, dtype) {
  return `${model}:${dtype}`;
}
async function getEngine(model = DEFAULT_MODEL, dtype = DEFAULT_DTYPE, onProgress) {
  const key = engineKey(model, dtype);
  if (!engines.has(key)) {
    const initPromise = initEngine(model, dtype, onProgress);
    engines.set(key, initPromise);
    initPromise.catch(() => engines.delete(key));
  }
  return engines.get(key);
}
async function initEngine(model, dtype, onProgress) {
  let pipeline;
  try {
    const transformers = await import("@huggingface/transformers");
    pipeline = transformers.pipeline;
  } catch {
    throw new SemvecError(
      "WASM_UNSUPPORTED",
      "Failed to load @huggingface/transformers. Ensure the package is installed and WASM is supported in this environment."
    );
  }
  let extractor;
  try {
    extractor = await pipeline("feature-extraction", model, {
      dtype,
      device: "wasm",
      // safe default; webgpu can be added later
      progress_callback: (event) => {
        if (onProgress && event.status === "progress" && typeof event.progress === "number") {
          onProgress({
            phase: "model",
            percent: Math.round(event.progress),
            detail: event.file
          });
        }
      }
    });
  } catch (err) {
    if (typeof navigator !== "undefined" && !navigator.onLine) {
      throw new SemvecError(
        "MODEL_OFFLINE",
        `Model "${model}" is not cached and the device is offline. Connect to the internet for the initial download (~30 MB, one-time).`,
        { model }
      );
    }
    throw new SemvecError(
      "MODEL_LOAD_FAILED",
      `Failed to load model "${model}": ${err instanceof Error ? err.message : String(err)}`,
      { model, originalError: err }
    );
  }
  return {
    async embed(text) {
      const output = await extractor(text, {
        pooling: "mean",
        normalize: true
      });
      const data = output.tolist()[0];
      return new Float32Array(data);
    },
    async embedBatch(texts, onBatchProgress) {
      const results = [];
      const BATCH_SIZE = 16;
      for (let i = 0; i < texts.length; i += BATCH_SIZE) {
        const batchTexts = texts.slice(i, i + BATCH_SIZE);
        for (const text of batchTexts) {
          const output = await extractor(text, {
            pooling: "mean",
            normalize: true
          });
          const data = output.tolist()[0];
          results.push(new Float32Array(data));
        }
        if (onBatchProgress) {
          onBatchProgress(Math.min(i + BATCH_SIZE, texts.length), texts.length);
        }
      }
      return results;
    }
  };
}
function cosineSim(a, b) {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return Math.max(-1, Math.min(1, dot));
}
function topK(query, corpus, k, threshold = 0) {
  const scored = [];
  for (let i = 0; i < corpus.length; i++) {
    const score = cosineSim(query, corpus[i]);
    if (score >= threshold) {
      scored.push({ index: i, score });
    }
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}
function estimateTokens(text) {
  return Math.ceil(text.length / 4);
}
const AUTO_CHUNK_THRESHOLD = 256;
const CHUNK_SIZE = 200;
const CHUNK_OVERLAP = 30;
const OVERLAP_CHARS = CHUNK_OVERLAP * 4;
const CHUNK_CHARS = CHUNK_SIZE * 4;
const SENTENCE_END = /[.!?]\s+/g;
function chunk(text, sourceIndex) {
  const trimmed = text.trim();
  if (!trimmed) return [];
  if (estimateTokens(trimmed) <= AUTO_CHUNK_THRESHOLD) {
    return [{ text: trimmed, sourceIndex, chunkIndex: 0 }];
  }
  const sentences = [];
  let lastEnd = 0;
  let match;
  SENTENCE_END.lastIndex = 0;
  while ((match = SENTENCE_END.exec(trimmed)) !== null) {
    sentences.push(trimmed.slice(lastEnd, match.index + match[0].length));
    lastEnd = match.index + match[0].length;
  }
  if (lastEnd < trimmed.length) {
    sentences.push(trimmed.slice(lastEnd));
  }
  const chunks = [];
  let current = "";
  let chunkIndex = 0;
  for (const sentence of sentences) {
    if (current.length + sentence.length > CHUNK_CHARS && current.length > 0) {
      chunks.push({ text: current.trim(), sourceIndex, chunkIndex });
      chunkIndex++;
      const overlapStart = Math.max(0, current.length - OVERLAP_CHARS);
      current = current.slice(overlapStart) + sentence;
    } else {
      current += sentence;
    }
  }
  if (current.trim()) {
    chunks.push({ text: current.trim(), sourceIndex, chunkIndex });
  }
  return chunks;
}
function chunkAll(texts) {
  const allChunks = [];
  for (let i = 0; i < texts.length; i++) {
    allChunks.push(...chunk(texts[i], i));
  }
  return allChunks;
}
const STORE_NAME = "embeddings";
const DB_VERSION = 1;
function contentHash(text) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = hash * 16777619 >>> 0;
  }
  return "sv_" + hash.toString(36);
}
async function createStore(namespace, model) {
  const dbName = `semantra_${namespace}_${model.replace(/[^a-zA-Z0-9]/g, "_")}`;
  try {
    return await createIDBStore(dbName);
  } catch {
    return createMemoryStore();
  }
}
async function createIDBStore(dbName) {
  const db = await openDB(dbName);
  return {
    async get(hash) {
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readonly");
        const req = tx.objectStore(STORE_NAME).get(hash);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
    },
    async getMany(hashes) {
      const map = /* @__PURE__ */ new Map();
      if (hashes.length === 0) return map;
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readonly");
        const store = tx.objectStore(STORE_NAME);
        let completed = 0;
        for (const hash of hashes) {
          const req = store.get(hash);
          req.onsuccess = () => {
            if (req.result) map.set(hash, req.result);
            completed++;
            if (completed === hashes.length) resolve(map);
          };
          req.onerror = () => reject(req.error);
        }
      });
    },
    async put(entry) {
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
        const req = tx.objectStore(STORE_NAME).put(entry);
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
      });
    },
    async putMany(entries) {
      if (entries.length === 0) return;
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
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
        const tx = db.transaction(STORE_NAME, "readwrite");
        const req = tx.objectStore(STORE_NAME).delete(hash);
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
      });
    },
    async clear() {
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
        const req = tx.objectStore(STORE_NAME).clear();
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
      });
    }
  };
}
function openDB(dbName) {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(dbName, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "hash" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
function createMemoryStore() {
  const map = /* @__PURE__ */ new Map();
  return {
    async get(hash) {
      return map.get(hash);
    },
    async getMany(hashes) {
      const result = /* @__PURE__ */ new Map();
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
    }
  };
}
async function ingestURL(url, field) {
  let response;
  try {
    response = await fetch(url);
  } catch (err) {
    throw new SemvecError(
      "FETCH_FAILED",
      `Failed to fetch "${url}": ${err instanceof Error ? err.message : "Network error"}`,
      { url, originalError: err }
    );
  }
  if (!response.ok) {
    throw new SemvecError(
      "FETCH_FAILED",
      `Fetch "${url}" returned HTTP ${response.status}`,
      { url, status: response.status }
    );
  }
  const contentType = response.headers.get("content-type") || "";
  const text = await response.text();
  if (contentType.includes("json") || url.endsWith(".json")) {
    return parseJSON(text, url, field);
  }
  if (contentType.includes("markdown") || url.endsWith(".md")) {
    return [{ text: stripMarkdown(text), source: url }];
  }
  return [{ text, source: url }];
}
function parseJSON(raw, source, field) {
  let data;
  try {
    data = JSON.parse(raw);
  } catch {
    throw new SemvecError(
      "FETCH_FAILED",
      `Failed to parse JSON from "${source}"`,
      { url: source }
    );
  }
  if (Array.isArray(data)) {
    return extractFromArray(data, source, field);
  }
  if (data && typeof data === "object" && !Array.isArray(data)) {
    const obj = data;
    const arrayKeys = Object.keys(obj).filter((k) => Array.isArray(obj[k]));
    if (arrayKeys.length === 1) {
      return extractFromArray(obj[arrayKeys[0]], source, field);
    }
    for (const key of ["data", "items", "results", "entries", "docs", "documents"]) {
      if (Array.isArray(obj[key])) {
        return extractFromArray(obj[key], source, field);
      }
    }
  }
  return [{ text: JSON.stringify(data), source }];
}
function extractFromArray(arr, source, field) {
  const docs = [];
  for (const item of arr) {
    if (typeof item === "string") {
      docs.push({ text: item, source });
    } else if (item && typeof item === "object") {
      const obj = item;
      if (field && typeof obj[field] === "string") {
        docs.push({ text: obj[field], source, meta: obj });
      } else {
        for (const key of ["text", "content", "body", "question", "title", "description"]) {
          if (typeof obj[key] === "string") {
            docs.push({ text: obj[key], source, meta: obj });
            break;
          }
        }
      }
    }
  }
  return docs;
}
function stripMarkdown(md) {
  return md.replace(/^#{1,6}\s+/gm, "").replace(/\*{1,3}([^*]+)\*{1,3}/g, "$1").replace(/_{1,3}([^_]+)_{1,3}/g, "$1").replace(/\[([^\]]+)\]\([^)]+\)/g, "$1").replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1").replace(/```[\s\S]*?```/g, "").replace(/`([^`]+)`/g, "$1").replace(/^>\s+/gm, "").replace(/^[-*_]{3,}\s*$/gm, "").replace(/\n{3,}/g, "\n\n").trim();
}
class Semvec {
  opts;
  rawItems;
  field;
  corpus = [];
  store = null;
  initPromise = null;
  destroyed = false;
  /**
   * Create a Semvec instance from local data.
   * For URLs, use `Semvec.fromURL()` instead.
   *
   * @param data - Array of strings or objects to search over.
   * @param options - Configuration options.
   */
  constructor(data, options = {}) {
    this.rawItems = data;
    this.field = options.field;
    this.opts = {
      model: options.model ?? DEFAULT_MODEL,
      dtype: options.dtype ?? DEFAULT_DTYPE,
      persist: options.persist ?? true,
      ...options
    };
  }
  /**
   * Create a Semvec instance from a URL.
   * Fetches and parses JSON/Markdown/text from the URL.
   *
   * @param url - URL to fetch data from.
   * @param options - Configuration options.
   * @throws SemvecError with code 'FETCH_FAILED' on network error.
   */
  static async fromURL(url, options = {}) {
    const docs = await ingestURL(url, options.field);
    const texts = docs.map((d) => d.text);
    return new Semvec(texts, {
      ...options,
      namespace: options.namespace ?? contentHash(url)
    });
  }
  /**
   * Semantic search over the corpus.
   * Auto-initializes on first call — no .ready needed.
   *
   * @param query - The search query text.
   * @param options - Search options (topK, threshold).
   * @returns Array of results sorted by score descending.
   */
  async search(query, options = {}) {
    this.assertNotDestroyed();
    if (!query.trim()) return [];
    await this.ensureInit();
    if (this.corpus.length === 0) return [];
    const engine = await getEngine(this.opts.model, this.opts.dtype);
    const queryVec = await engine.embed(query);
    const corpusVecs = this.corpus.map((e) => e.embedding);
    const k = options.topK ?? 3;
    const threshold = options.threshold ?? 0;
    const hits = topK(queryVec, corpusVecs, k, threshold);
    return hits.map((hit) => {
      const entry = this.corpus[hit.index];
      return {
        text: entry.text,
        score: Math.round(hit.score * 1e3) / 1e3,
        // 3 decimal places
        index: entry.sourceIndex,
        item: entry.item
      };
    });
  }
  /**
   * Embed a single text using the configured model.
   *
   * @param text - Text to embed.
   * @returns L2-normalized embedding vector.
   */
  async embed(text) {
    this.assertNotDestroyed();
    await this.ensureInit();
    const engine = await getEngine(this.opts.model, this.opts.dtype);
    return engine.embed(text);
  }
  /**
   * Compute cosine similarity between two texts.
   *
   * @returns Similarity score between 0 and 1.
   */
  async similarity(a, b) {
    this.assertNotDestroyed();
    await this.ensureInit();
    const engine = await getEngine(this.opts.model, this.opts.dtype);
    const [vecA, vecB] = await Promise.all([engine.embed(a), engine.embed(b)]);
    return cosineSim(vecA, vecB);
  }
  /**
   * Add a document to the corpus.
   * Returns a content-hash ID that can be used with remove().
   *
   * @param item - String or object (with field) to add.
   * @returns Content-hash ID of the added document.
   */
  async add(item) {
    this.assertNotDestroyed();
    await this.ensureInit();
    const text = this.extractText(item);
    const id = contentHash(text);
    const engine = await getEngine(this.opts.model, this.opts.dtype);
    const embedding = await engine.embed(text);
    const entry = {
      id,
      text,
      embedding,
      sourceIndex: this.rawItems.length,
      chunkIndex: 0,
      item
    };
    this.corpus.push(entry);
    this.rawItems.push(item);
    if (this.store) {
      await this.store.put({
        hash: id,
        text,
        embedding: Array.from(embedding),
        timestamp: Date.now()
      });
    }
    return id;
  }
  /**
   * Add multiple documents to the corpus.
   *
   * @param items - Array of strings or objects to add.
   * @returns Array of content-hash IDs.
   */
  async addMany(items) {
    const ids = [];
    for (const item of items) {
      ids.push(await this.add(item));
    }
    return ids;
  }
  /**
   * Remove a document by its content-hash ID or original array index.
   *
   * @param idOrIndex - Content-hash ID (string) or corpus index (number).
   */
  remove(idOrIndex) {
    this.assertNotDestroyed();
    if (typeof idOrIndex === "string") {
      const idx = this.corpus.findIndex((e) => e.id === idOrIndex);
      if (idx !== -1) {
        this.corpus.splice(idx, 1);
        if (this.store) {
          this.store.delete(idOrIndex).catch(() => {
          });
        }
      }
    } else {
      const idx = this.corpus.findIndex((e) => e.sourceIndex === idOrIndex);
      if (idx !== -1) {
        const entry = this.corpus[idx];
        this.corpus.splice(idx, 1);
        if (this.store) {
          this.store.delete(entry.id).catch(() => {
          });
        }
      }
    }
  }
  /**
   * Destroy this instance — terminates any workers and clears in-memory state.
   * Does NOT clear persisted IndexedDB data.
   */
  destroy() {
    this.destroyed = true;
    this.corpus = [];
    this.store = null;
    this.initPromise = null;
  }
  // ─── Private ──────────────────────────────────────────────
  /** Ensure corpus is initialized (model loaded + texts embedded). */
  async ensureInit() {
    if (!this.initPromise) {
      this.initPromise = this.init();
    }
    return this.initPromise;
  }
  /** Full initialization: create store, extract texts, chunk, embed. */
  async init() {
    const model = this.opts.model;
    const dtype = this.opts.dtype;
    const namespace = this.opts.namespace ?? contentHash(JSON.stringify(this.rawItems.slice(0, 5)));
    if (this.opts.persist) {
      this.store = await createStore(namespace, model);
    }
    const texts = this.rawItems.map((item) => this.extractText(item));
    const chunks = chunkAll(texts);
    if (chunks.length === 0) return;
    const chunkTexts = chunks.map((c) => c.text);
    const hashes = chunkTexts.map((t) => contentHash(t));
    let cached = /* @__PURE__ */ new Map();
    if (this.store) {
      cached = await this.store.getMany(hashes);
    }
    const uncachedIndices = [];
    for (let i = 0; i < hashes.length; i++) {
      if (!cached.has(hashes[i])) {
        uncachedIndices.push(i);
      }
    }
    let newEmbeddings = [];
    if (uncachedIndices.length > 0) {
      const engine = await getEngine(model, dtype, this.opts.onProgress);
      const textsToEmbed = uncachedIndices.map((i) => chunkTexts[i]);
      newEmbeddings = await engine.embedBatch(textsToEmbed, (done, total) => {
        this.opts.onProgress?.({
          phase: "embedding",
          percent: Math.round(done / total * 100),
          detail: `${done}/${total} chunks`
        });
      });
      if (this.store) {
        const entries = uncachedIndices.map((origIdx, newIdx) => ({
          hash: hashes[origIdx],
          text: chunkTexts[origIdx],
          embedding: Array.from(newEmbeddings[newIdx]),
          timestamp: Date.now()
        }));
        await this.store.putMany(entries);
      }
    }
    let newEmbIdx = 0;
    for (let i = 0; i < chunks.length; i++) {
      const hash = hashes[i];
      const cachedEntry = cached.get(hash);
      let embedding;
      if (cachedEntry) {
        embedding = new Float32Array(cachedEntry.embedding);
      } else {
        embedding = newEmbeddings[newEmbIdx++];
      }
      this.corpus.push({
        id: hash,
        text: chunks[i].text,
        embedding,
        sourceIndex: chunks[i].sourceIndex,
        chunkIndex: chunks[i].chunkIndex,
        item: this.rawItems[chunks[i].sourceIndex]
      });
    }
  }
  /** Extract text from a corpus item (string or object with field). */
  extractText(item) {
    if (typeof item === "string") return item;
    if (item && typeof item === "object") {
      const obj = item;
      if (this.field && typeof obj[this.field] === "string") {
        return obj[this.field];
      }
      for (const key of ["text", "content", "body", "question", "title", "description"]) {
        if (typeof obj[key] === "string") {
          return obj[key];
        }
      }
    }
    return String(item);
  }
  /** Throw if instance has been destroyed. */
  assertNotDestroyed() {
    if (this.destroyed) {
      throw new SemvecError("DESTROYED", "This Semvec instance has been destroyed.");
    }
  }
}
export {
  DEFAULT_MODEL as D,
  Semvec as S,
  cosineSim as a,
  DEFAULT_DTYPE as b,
  chunkAll as c,
  SemvecError as d,
  getEngine as g,
  topK as t
};
