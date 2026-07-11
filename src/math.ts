/**
 * Pure math utilities for vector operations.
 * All functions work on Float32Array typed arrays for performance.
 */

/**
 * Cosine similarity between two L2-normalized vectors.
 * Since vectors are normalized, this is just the dot product.
 * Returns 0 for mismatched lengths or empty vectors.
 */
export function cosineSim(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  // Clamp to [-1, 1] to handle floating point drift
  return Math.max(-1, Math.min(1, dot));
}

/**
 * L2-normalize a vector in place and return it.
 * Zero vectors are returned unchanged (no division by zero).
 */
export function l2Normalize(vec: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) {
    norm += vec[i] * vec[i];
  }
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < vec.length; i++) {
      vec[i] /= norm;
    }
  }
  return vec;
}

/**
 * Find the top-K most similar vectors to a query.
 * Returns results sorted by score descending, filtered by threshold.
 */
export function topK(
  query: Float32Array,
  corpus: Float32Array[],
  k: number,
  threshold: number = 0
): { index: number; score: number }[] {
  const scored: { index: number; score: number }[] = [];

  for (let i = 0; i < corpus.length; i++) {
    const score = cosineSim(query, corpus[i]);
    if (score >= threshold) {
      scored.push({ index: i, score });
    }
  }

  // Sort descending by score
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}
