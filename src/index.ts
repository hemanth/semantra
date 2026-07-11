/**
 * semantra — Semantic vector search, entirely in the browser.
 *
 * @example
 * ```ts
 * import { search } from 'semantra';
 *
 * const results = await search('forgot password', [
 *   'How to reset your password',
 *   'Track your delivery status',
 *   'Update billing address',
 * ]);
 * // → [{ text: 'How to reset your password', score: 0.88, index: 0 }]
 * ```
 *
 * @packageDocumentation
 */

import type { SearchResult, SearchOptions } from './types.js';
import { getEngine, DEFAULT_MODEL, DEFAULT_DTYPE } from './engine.js';
import { topK, cosineSim } from './math.js';
import { chunkAll } from './chunker.js';

// ─── Corpus embedding cache ────────────────────────────────
// WeakMap keyed by array reference: same array = skip re-embedding.
const corpusCache = new WeakMap<
  object,
  { texts: string[]; embeddings: Float32Array[]; model: string }
>();

/**
 * Semantic search — the main API.
 * One function, one await. Model loads lazily on first call, cached forever.
 *
 * Corpus embeddings are cached by **array reference**: passing the same array
 * skips re-embedding (instant). A new array triggers fresh embedding.
 *
 * @param query - The search query text.
 * @param corpus - Array of strings or objects to search over.
 * @param options - Search options (topK, threshold, field, model).
 * @returns Array of results sorted by score descending.
 *
 * @example
 * ```ts
 * // Strings
 * const results = await search('query', ['doc1', 'doc2', 'doc3']);
 *
 * // Objects with field
 * const results = await search('query', products, { field: 'description' });
 *
 * // Same array = instant (cached)
 * const docs = ['doc1', 'doc2'];
 * await search('query1', docs); // embeds
 * await search('query2', docs); // instant — same ref
 * ```
 */
export async function search<T>(
  query: string,
  corpus: T[],
  options: SearchOptions = {},
): Promise<SearchResult<T>[]> {
  if (!query.trim() || corpus.length === 0) return [];

  const model = options.model ?? DEFAULT_MODEL;
  const field = options.field;
  const k = options.topK ?? 3;
  const threshold = options.threshold ?? 0;

  const engine = await getEngine(model, DEFAULT_DTYPE);

  // Extract texts from corpus items
  const texts = corpus.map((item) => extractText(item, field));

  // Check WeakMap cache — same array ref + same model = reuse embeddings
  let corpusEmbeddings: Float32Array[];
  const cached = corpusCache.get(corpus as object);

  if (cached && cached.model === model && cached.texts.length === texts.length) {
    corpusEmbeddings = cached.embeddings;
  } else {
    // Chunk all texts
    const chunks = chunkAll(texts);
    const chunkTexts = chunks.map((c) => c.text);

    // Embed all chunks
    const chunkEmbeddings = await engine.embedBatch(chunkTexts);

    // For each original text, find its best chunk embedding
    // (if text was short, there's only one chunk)
    corpusEmbeddings = new Array(texts.length);
    const bestScorePerSource = new Float64Array(texts.length).fill(-Infinity);

    // First pass: assign any embedding to each source
    for (let i = 0; i < chunks.length; i++) {
      const srcIdx = chunks[i].sourceIndex;
      if (!corpusEmbeddings[srcIdx]) {
        corpusEmbeddings[srcIdx] = chunkEmbeddings[i];
      }
    }

    // Cache for next call
    corpusCache.set(corpus as object, {
      texts: [...texts],
      embeddings: corpusEmbeddings,
      model,
    });
  }

  // Embed query and find top-K
  const queryVec = await engine.embed(query);
  const hits = topK(queryVec, corpusEmbeddings, k, threshold);

  return hits.map((hit) => ({
    text: texts[hit.index],
    score: Math.round(hit.score * 1000) / 1000,
    index: hit.index,
    item: corpus[hit.index],
  }));
}

/**
 * Compute cosine similarity between two texts.
 * Model loads lazily on first call.
 *
 * @param a - First text.
 * @param b - Second text.
 * @returns Similarity score between -1 and 1 (typically 0–1 for sentence embeddings).
 *
 * @example
 * ```ts
 * const score = await similarity('forgot password', 'reset my password');
 * // → 0.87
 * ```
 */
export async function similarity(a: string, b: string): Promise<number> {
  const engine = await getEngine();
  const [vecA, vecB] = await Promise.all([engine.embed(a), engine.embed(b)]);
  return Math.round(cosineSim(vecA, vecB) * 1000) / 1000;
}

/**
 * Embed a single text into a vector.
 * Power-user API for building custom pipelines.
 *
 * @param text - Text to embed.
 * @param model - Override the default model.
 * @returns L2-normalized Float32Array embedding.
 *
 * @example
 * ```ts
 * const vec = await embed('semantic search is cool');
 * // → Float32Array(384)
 * ```
 */
export async function embed(text: string, model?: string): Promise<Float32Array> {
  const engine = await getEngine(model);
  return engine.embed(text);
}

// ─── Helpers ────────────────────────────────────────────────

/** Extract text from a corpus item (string or object with field). */
function extractText(item: unknown, field?: string): string {
  if (typeof item === 'string') return item;

  if (item && typeof item === 'object') {
    const obj = item as Record<string, unknown>;

    if (field && typeof obj[field] === 'string') {
      return obj[field] as string;
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

// ─── Re-exports ─────────────────────────────────────────────
export { Semvec } from './semvec.js';
export { SemvecError } from './errors.js';
export type {
  SearchResult,
  SearchOptions,
  SemvecOptions,
  Progress,
} from './types.js';
