/**
 * Embedding engine — wraps @huggingface/transformers for ONNX inference.
 * Lazy singleton per model: first call downloads + initializes, subsequent calls are instant.
 */

import type { Progress } from './types.js';
import { SemvecError } from './errors.js';

/** Default embedding model. */
export const DEFAULT_MODEL = 'Snowflake/snowflake-arctic-embed-s';
export const DEFAULT_DTYPE = 'q8';

/** Engine instance with a loaded model. */
interface Engine {
  embed(text: string): Promise<Float32Array>;
  embedBatch(texts: string[], onProgress?: (done: number, total: number) => void): Promise<Float32Array[]>;
}

/** Cache of initialized engines, keyed by `model:dtype`. */
const engines = new Map<string, Promise<Engine>>();

/** Build the cache key for an engine. */
function engineKey(model: string, dtype: string): string {
  return `${model}:${dtype}`;
}

/**
 * Get or create an engine for the given model + dtype.
 * The first call triggers model download; subsequent calls return the cached engine.
 */
export async function getEngine(
  model: string = DEFAULT_MODEL,
  dtype: string = DEFAULT_DTYPE,
  onProgress?: (progress: Progress) => void,
): Promise<Engine> {
  const key = engineKey(model, dtype);

  if (!engines.has(key)) {
    const initPromise = initEngine(model, dtype, onProgress);
    engines.set(key, initPromise);

    // If init fails, remove from cache so it can be retried
    initPromise.catch(() => engines.delete(key));
  }

  return engines.get(key)!;
}

/**
 * Initialize a new engine by loading the model via Transformers.js.
 */
async function initEngine(
  model: string,
  dtype: string,
  onProgress?: (progress: Progress) => void,
): Promise<Engine> {
  let pipeline: typeof import('@huggingface/transformers').pipeline;

  try {
    const transformers = await import('@huggingface/transformers');
    pipeline = transformers.pipeline;
  } catch {
    throw new SemvecError(
      'WASM_UNSUPPORTED',
      'Failed to load @huggingface/transformers. Ensure the package is installed and WASM is supported in this environment.',
    );
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let extractor: any;

  try {
    extractor = await pipeline('feature-extraction', model, {
      dtype,
      device: 'wasm', // safe default; webgpu can be added later
      progress_callback: (event: Record<string, unknown>) => {
        if (onProgress && event.status === 'progress' && typeof event.progress === 'number') {
          onProgress({
            phase: 'model',
            percent: Math.round(event.progress as number),
            detail: event.file as string | undefined,
          });
        }
      },
    } as Record<string, unknown>);
  } catch (err) {
    // Check if offline
    if (typeof navigator !== 'undefined' && !navigator.onLine) {
      throw new SemvecError(
        'MODEL_OFFLINE',
        `Model "${model}" is not cached and the device is offline. Connect to the internet for the initial download (~30 MB, one-time).`,
        { model },
      );
    }
    throw new SemvecError(
      'MODEL_LOAD_FAILED',
      `Failed to load model "${model}": ${err instanceof Error ? err.message : String(err)}`,
      { model, originalError: err },
    );
  }

  return {
    async embed(text: string): Promise<Float32Array> {
      const output = await extractor(text, {
        pooling: 'mean',
        normalize: true,
      });
      const data = output.tolist()[0] as number[];
      return new Float32Array(data);
    },

    async embedBatch(
      texts: string[],
      onBatchProgress?: (done: number, total: number) => void,
    ): Promise<Float32Array[]> {
      const results: Float32Array[] = [];
      // Process in small batches to allow progress updates
      const BATCH_SIZE = 16;

      for (let i = 0; i < texts.length; i += BATCH_SIZE) {
        const batchTexts = texts.slice(i, i + BATCH_SIZE);

        for (const text of batchTexts) {
          const output = await extractor(text, {
            pooling: 'mean',
            normalize: true,
          });
          const data = output.tolist()[0] as number[];
          results.push(new Float32Array(data));
        }

        if (onBatchProgress) {
          onBatchProgress(Math.min(i + BATCH_SIZE, texts.length), texts.length);
        }
      }

      return results;
    },
  };
}
