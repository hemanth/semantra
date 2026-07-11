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
export declare function embed(text: string, model?: string): Promise<Float32Array>;

/** Progress info passed to onProgress callback. */
export declare interface Progress {
    /** Current phase: model download or document embedding. */
    phase: 'model' | 'embedding';
    /** Progress percentage 0–100. */
    percent: number;
    /** Optional detail string, e.g. "23/100 documents". */
    detail?: string;
}

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
export declare function search<T>(query: string, corpus: T[], options?: SearchOptions): Promise<SearchResult<T>[]>;

/** Options for the search() function and sv.search(). */
export declare interface SearchOptions {
    /** Maximum number of results to return. @default 3 */
    topK?: number;
    /** Minimum similarity score to include in results. @default 0 */
    threshold?: number;
    /** Which field to embed when corpus items are objects. */
    field?: string;
    /** Override the default embedding model. */
    model?: string;
}

/** Result from a semantic search operation. */
export declare interface SearchResult<T = string> {
    /** The matched text (or best-matching chunk for long documents). */
    text: string;
    /** Cosine similarity score between 0 and 1. */
    score: number;
    /** Position in the original corpus array. */
    index: number;
    /** The original item from the corpus (string or object). */
    item: T;
}

export declare class Semvec {
    private readonly opts;
    private readonly rawItems;
    private readonly field?;
    private corpus;
    private store;
    private initPromise;
    private destroyed;
    /**
     * Create a Semvec instance from local data.
     * For URLs, use `Semvec.fromURL()` instead.
     *
     * @param data - Array of strings or objects to search over.
     * @param options - Configuration options.
     */
    constructor(data: unknown[], options?: SemvecOptions);
    /**
     * Create a Semvec instance from a URL.
     * Fetches and parses JSON/Markdown/text from the URL.
     *
     * @param url - URL to fetch data from.
     * @param options - Configuration options.
     * @throws SemvecError with code 'FETCH_FAILED' on network error.
     */
    static fromURL(url: string, options?: SemvecOptions & {
        field?: string;
    }): Promise<Semvec>;
    /**
     * Semantic search over the corpus.
     * Auto-initializes on first call — no .ready needed.
     *
     * @param query - The search query text.
     * @param options - Search options (topK, threshold).
     * @returns Array of results sorted by score descending.
     */
    search(query: string, options?: {
        topK?: number;
        threshold?: number;
    }): Promise<SearchResult<unknown>[]>;
    /**
     * Embed a single text using the configured model.
     *
     * @param text - Text to embed.
     * @returns L2-normalized embedding vector.
     */
    embed(text: string): Promise<Float32Array>;
    /**
     * Compute cosine similarity between two texts.
     *
     * @returns Similarity score between 0 and 1.
     */
    similarity(a: string, b: string): Promise<number>;
    /**
     * Add a document to the corpus.
     * Returns a content-hash ID that can be used with remove().
     *
     * @param item - String or object (with field) to add.
     * @returns Content-hash ID of the added document.
     */
    add(item: unknown): Promise<string>;
    /**
     * Add multiple documents to the corpus.
     *
     * @param items - Array of strings or objects to add.
     * @returns Array of content-hash IDs.
     */
    addMany(items: unknown[]): Promise<string[]>;
    /**
     * Remove a document by its content-hash ID or original array index.
     *
     * @param idOrIndex - Content-hash ID (string) or corpus index (number).
     */
    remove(idOrIndex: string | number): void;
    /**
     * Destroy this instance — terminates any workers and clears in-memory state.
     * Does NOT clear persisted IndexedDB data.
     */
    destroy(): void;
    /** Ensure corpus is initialized (model loaded + texts embedded). */
    private ensureInit;
    /** Full initialization: create store, extract texts, chunk, embed. */
    private init;
    /** Extract text from a corpus item (string or object with field). */
    private extractText;
    /** Throw if instance has been destroyed. */
    private assertNotDestroyed;
}

/**
 * Typed error thrown by semantra operations.
 * Check `error.code` for programmatic handling.
 *
 * @example
 * ```ts
 * try {
 *   await search('query', docs);
 * } catch (e) {
 *   if (e instanceof SemvecError && e.code === 'MODEL_OFFLINE') {
 *     showBanner('Connect to internet to download the search model (one-time, 30 MB)');
 *   }
 * }
 * ```
 */
export declare class SemvecError extends Error {
    readonly code: SemvecErrorCode;
    readonly detail?: unknown;
    constructor(code: SemvecErrorCode, message: string, detail?: unknown);
}

/** Error codes for SemvecError. */
declare type SemvecErrorCode = 'MODEL_OFFLINE' | 'FETCH_FAILED' | 'WASM_UNSUPPORTED' | 'MODEL_LOAD_FAILED' | 'DESTROYED';

/** Options for the Semvec class constructor. */
export declare interface SemvecOptions {
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
export declare function similarity(a: string, b: string): Promise<number>;

export { }
