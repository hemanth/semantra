/** Result from a semantic search operation. */
export interface SearchResult<T = string> {
  /** The matched text (or best-matching chunk for long documents). */
  text: string;
  /** Cosine similarity score between 0 and 1. */
  score: number;
  /** Position in the original corpus array. */
  index: number;
  /** The original item from the corpus (string or object). */
  item: T;
}

/** Options for the search() function and sv.search(). */
export interface SearchOptions {
  /** Maximum number of results to return. @default 3 */
  topK?: number;
  /** Minimum similarity score to include in results. @default 0 */
  threshold?: number;
  /** Which field to embed when corpus items are objects. */
  field?: string;
  /** Override the default embedding model. */
  model?: string;
}

/** Options for the Semvec class constructor. */
export interface SemvecOptions {
  /** Which field to embed when corpus items are objects. */
  field?: string;
  /** HuggingFace model ID or path. @default 'Snowflake/snowflake-arctic-embed-s' */
  model?: string;
  /** Quantization dtype. @default 'q8' */
  dtype?: 'q4' | 'q8' | 'fp16' | 'fp32';
  /** Isolates IndexedDB cache from other instances. Auto-generated if not provided. */
  namespace?: string;
  /** Whether to persist embeddings in IndexedDB. @default true */
  persist?: boolean;
  /** Progress callback for model download and embedding. */
  onProgress?: (progress: Progress) => void;
}

/** Progress info passed to onProgress callback. */
export interface Progress {
  /** Current phase: model download or document embedding. */
  phase: 'model' | 'embedding';
  /** Progress percentage 0–100. */
  percent: number;
  /** Optional detail string, e.g. "23/100 documents". */
  detail?: string;
}

/** Internal document representation after ingestion. */
export interface Document {
  text: string;
  source?: string;
  meta?: Record<string, unknown>;
}

/** Internal chunk representation. */
export interface Chunk {
  text: string;
  sourceIndex: number;
  chunkIndex: number;
}

/** Cache entry stored in IndexedDB. */
export interface CacheEntry {
  hash: string;
  text: string;
  embedding: number[];
  source?: string;
  timestamp: number;
}

/** Options for fromURL static method. */
export interface FromURLOptions extends SemvecOptions {
  /** Which field to extract text from in JSON response objects. */
  field?: string;
}
