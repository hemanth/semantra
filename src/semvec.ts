/**
 * Semvec class — persistent corpus with incremental updates.
 * For simple one-off searches, use the top-level search() function instead.
 *
 * @example
 * ```ts
 * const sv = new Semvec(['doc1', 'doc2', 'doc3']);
 * const results = await sv.search('query'); // auto-inits, no .ready needed
 * ```
 */

import type { SearchResult, SemvecOptions, Chunk } from './types.js';
import { SemvecError } from './errors.js';
import { getEngine, DEFAULT_MODEL, DEFAULT_DTYPE } from './engine.js';
import { topK, cosineSim } from './math.js';
import { chunkAll } from './chunker.js';
import { createStore, contentHash } from './store.js';
import { ingestURL } from './ingest.js';
import type { EmbeddingStore } from './store.js';

/** Internal corpus item with its text, embedding, and metadata. */
interface CorpusEntry {
  id: string; // content hash
  text: string;
  embedding: Float32Array;
  sourceIndex: number;
  chunkIndex: number;
  item: unknown; // original corpus item
}

export class Semvec {
  private readonly opts: Required<
    Pick<SemvecOptions, 'model' | 'dtype' | 'persist'>
  > & SemvecOptions;
  private readonly rawItems: unknown[];
  private readonly field?: string;

  private corpus: CorpusEntry[] = [];
  private store: EmbeddingStore | null = null;
  private initPromise: Promise<void> | null = null;
  private destroyed = false;

  /**
   * Create a Semvec instance from local data.
   * For URLs, use `Semvec.fromURL()` instead.
   *
   * @param data - Array of strings or objects to search over.
   * @param options - Configuration options.
   */
  constructor(data: unknown[], options: SemvecOptions = {}) {
    this.rawItems = data;
    this.field = options.field;
    this.opts = {
      model: options.model ?? DEFAULT_MODEL,
      dtype: options.dtype ?? DEFAULT_DTYPE,
      persist: options.persist ?? true,
      ...options,
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
  static async fromURL(
    url: string,
    options: SemvecOptions & { field?: string } = {},
  ): Promise<Semvec> {
    const docs = await ingestURL(url, options.field);
    const texts = docs.map((d) => d.text);
    return new Semvec(texts, {
      ...options,
      namespace: options.namespace ?? contentHash(url),
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
  async search(
    query: string,
    options: { topK?: number; threshold?: number } = {},
  ): Promise<SearchResult<unknown>[]> {
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
        score: Math.round(hit.score * 1000) / 1000, // 3 decimal places
        index: entry.sourceIndex,
        item: entry.item,
      };
    });
  }

  /**
   * Embed a single text using the configured model.
   *
   * @param text - Text to embed.
   * @returns L2-normalized embedding vector.
   */
  async embed(text: string): Promise<Float32Array> {
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
  async similarity(a: string, b: string): Promise<number> {
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
  async add(item: unknown): Promise<string> {
    this.assertNotDestroyed();
    await this.ensureInit();

    const text = this.extractText(item);
    const id = contentHash(text);
    const engine = await getEngine(this.opts.model, this.opts.dtype);
    const embedding = await engine.embed(text);

    const entry: CorpusEntry = {
      id,
      text,
      embedding,
      sourceIndex: this.rawItems.length,
      chunkIndex: 0,
      item,
    };

    this.corpus.push(entry);
    this.rawItems.push(item);

    // Persist
    if (this.store) {
      await this.store.put({
        hash: id,
        text,
        embedding: Array.from(embedding),
        timestamp: Date.now(),
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
  async addMany(items: unknown[]): Promise<string[]> {
    const ids: string[] = [];
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
  remove(idOrIndex: string | number): void {
    this.assertNotDestroyed();

    if (typeof idOrIndex === 'string') {
      const idx = this.corpus.findIndex((e) => e.id === idOrIndex);
      if (idx !== -1) {
        this.corpus.splice(idx, 1);
        if (this.store) {
          this.store.delete(idOrIndex).catch(() => {}); // best-effort
        }
      }
    } else {
      const idx = this.corpus.findIndex((e) => e.sourceIndex === idOrIndex);
      if (idx !== -1) {
        const entry = this.corpus[idx];
        this.corpus.splice(idx, 1);
        if (this.store) {
          this.store.delete(entry.id).catch(() => {}); // best-effort
        }
      }
    }
  }

  /**
   * Destroy this instance — terminates any workers and clears in-memory state.
   * Does NOT clear persisted IndexedDB data.
   */
  destroy(): void {
    this.destroyed = true;
    this.corpus = [];
    this.store = null;
    this.initPromise = null;
  }

  // ─── Private ──────────────────────────────────────────────

  /** Ensure corpus is initialized (model loaded + texts embedded). */
  private async ensureInit(): Promise<void> {
    if (!this.initPromise) {
      this.initPromise = this.init();
    }
    return this.initPromise;
  }

  /** Full initialization: create store, extract texts, chunk, embed. */
  private async init(): Promise<void> {
    const model = this.opts.model;
    const dtype = this.opts.dtype;
    const namespace = this.opts.namespace ?? contentHash(JSON.stringify(this.rawItems.slice(0, 5)));

    // Create store
    if (this.opts.persist) {
      this.store = await createStore(namespace, model);
    }

    // Extract texts from raw items
    const texts = this.rawItems.map((item) => this.extractText(item));

    // Chunk long texts
    const chunks: Chunk[] = chunkAll(texts);

    if (chunks.length === 0) return;

    // Compute hashes and check cache
    const chunkTexts = chunks.map((c) => c.text);
    const hashes = chunkTexts.map((t) => contentHash(t));

    let cached = new Map<string, { embedding: number[] }>();
    if (this.store) {
      cached = await this.store.getMany(hashes);
    }

    // Find uncached texts
    const uncachedIndices: number[] = [];
    for (let i = 0; i < hashes.length; i++) {
      if (!cached.has(hashes[i])) {
        uncachedIndices.push(i);
      }
    }

    // Embed uncached texts
    let newEmbeddings: Float32Array[] = [];
    if (uncachedIndices.length > 0) {
      const engine = await getEngine(model, dtype, this.opts.onProgress);
      const textsToEmbed = uncachedIndices.map((i) => chunkTexts[i]);

      newEmbeddings = await engine.embedBatch(textsToEmbed, (done, total) => {
        this.opts.onProgress?.({
          phase: 'embedding',
          percent: Math.round((done / total) * 100),
          detail: `${done}/${total} chunks`,
        });
      });

      // Persist new embeddings
      if (this.store) {
        const entries = uncachedIndices.map((origIdx, newIdx) => ({
          hash: hashes[origIdx],
          text: chunkTexts[origIdx],
          embedding: Array.from(newEmbeddings[newIdx]),
          timestamp: Date.now(),
        }));
        await this.store.putMany(entries);
      }
    }

    // Build corpus entries
    let newEmbIdx = 0;
    for (let i = 0; i < chunks.length; i++) {
      const hash = hashes[i];
      const cachedEntry = cached.get(hash);

      let embedding: Float32Array;
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
        item: this.rawItems[chunks[i].sourceIndex],
      });
    }
  }

  /** Extract text from a corpus item (string or object with field). */
  private extractText(item: unknown): string {
    if (typeof item === 'string') return item;

    if (item && typeof item === 'object') {
      const obj = item as Record<string, unknown>;

      // Explicit field
      if (this.field && typeof obj[this.field] === 'string') {
        return obj[this.field] as string;
      }

      // Auto-detect common text fields
      for (const key of ['text', 'content', 'body', 'question', 'title', 'description']) {
        if (typeof obj[key] === 'string') {
          return obj[key] as string;
        }
      }
    }

    return String(item);
  }

  /** Throw if instance has been destroyed. */
  private assertNotDestroyed(): void {
    if (this.destroyed) {
      throw new SemvecError('DESTROYED', 'This Semvec instance has been destroyed.');
    }
  }
}
