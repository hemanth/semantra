/** Progress info passed to onProgress callback. */
declare interface Progress {
    /** Current phase: model download or document embedding. */
    phase: 'model' | 'embedding';
    /** Progress percentage 0–100. */
    percent: number;
    /** Optional detail string, e.g. "23/100 documents". */
    detail?: string;
}

/** Result from a semantic search operation. */
declare interface SearchResult<T = string> {
    /** The matched text (or best-matching chunk for long documents). */
    text: string;
    /** Cosine similarity score between 0 and 1. */
    score: number;
    /** Position in the original corpus array. */
    index: number;
    /** The original item from the corpus (string or object). */
    item: T;
}

/** Options for the Semvec class constructor. */
declare interface SemvecOptions {
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
 * React hook for semantic search over a corpus.
 * Handles initialization, debouncing, and cleanup automatically.
 *
 * @param data - Array of strings or objects to search over.
 * @param options - Search and Semvec options.
 * @returns Object with search function, results, loading state, and errors.
 */
export declare function useSearch<T>(data: T[], options?: UseSearchOptions): UseSearchReturn<T>;

/** Options for the useSearch hook. */
declare interface UseSearchOptions extends SemvecOptions {
    /** Debounce delay in ms for search-as-you-type. @default 150 */
    debounce?: number;
    /** Maximum results to return. @default 5 */
    topK?: number;
    /** Minimum similarity score. @default 0 */
    threshold?: number;
}

/** Return value of the useSearch hook. */
declare interface UseSearchReturn<T> {
    /** Trigger a search. Debounced by default. */
    search: (query: string) => void;
    /** Current search results. */
    results: SearchResult<T>[];
    /** Whether a search is in progress. */
    loading: boolean;
    /** Any error from the last operation. */
    error: Error | null;
}

export { }
