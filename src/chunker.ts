/**
 * Auto-chunking for long texts.
 * Short texts (≤256 estimated tokens) pass through unchanged.
 * Long texts are split on sentence boundaries with overlap.
 */

import type { Chunk } from './types.js';

/** Approximate token count (rough: 1 token ≈ 4 chars for English). */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/** Threshold above which text is auto-chunked (in estimated tokens). */
const AUTO_CHUNK_THRESHOLD = 256;
const CHUNK_SIZE = 200; // tokens (~800 chars)
const CHUNK_OVERLAP = 30; // tokens (~120 chars)
const OVERLAP_CHARS = CHUNK_OVERLAP * 4;
const CHUNK_CHARS = CHUNK_SIZE * 4;

/** Sentence-end pattern: period, exclamation, or question mark followed by whitespace. */
const SENTENCE_END = /[.!?]\s+/g;

/**
 * Split a single text into sentence-boundary-aware chunks.
 * - Short texts (≤256 estimated tokens) → returned as a single chunk.
 * - Long texts → split on sentence boundaries with configurable overlap.
 *
 * @param text - The text to chunk.
 * @param sourceIndex - The index of this text in the original corpus.
 * @returns Array of chunks with source attribution.
 */
export function chunk(text: string, sourceIndex: number): Chunk[] {
  const trimmed = text.trim();
  if (!trimmed) return [];

  // Short text — no chunking needed
  if (estimateTokens(trimmed) <= AUTO_CHUNK_THRESHOLD) {
    return [{ text: trimmed, sourceIndex, chunkIndex: 0 }];
  }

  // Find sentence boundaries
  const sentences: string[] = [];
  let lastEnd = 0;
  let match: RegExpExecArray | null;

  // Reset regex state
  SENTENCE_END.lastIndex = 0;
  while ((match = SENTENCE_END.exec(trimmed)) !== null) {
    sentences.push(trimmed.slice(lastEnd, match.index + match[0].length));
    lastEnd = match.index + match[0].length;
  }
  // Remaining text after last sentence boundary
  if (lastEnd < trimmed.length) {
    sentences.push(trimmed.slice(lastEnd));
  }

  // Group sentences into chunks with overlap
  const chunks: Chunk[] = [];
  let current = '';
  let chunkIndex = 0;

  for (const sentence of sentences) {
    if (current.length + sentence.length > CHUNK_CHARS && current.length > 0) {
      chunks.push({ text: current.trim(), sourceIndex, chunkIndex });
      chunkIndex++;
      // Overlap: keep the tail of the previous chunk
      const overlapStart = Math.max(0, current.length - OVERLAP_CHARS);
      current = current.slice(overlapStart) + sentence;
    } else {
      current += sentence;
    }
  }

  // Final chunk
  if (current.trim()) {
    chunks.push({ text: current.trim(), sourceIndex, chunkIndex });
  }

  return chunks;
}

/**
 * Chunk an array of texts. Short texts pass through, long texts are split.
 *
 * @param texts - Array of raw text strings.
 * @returns Flat array of all chunks with source attribution.
 */
export function chunkAll(texts: string[]): Chunk[] {
  const allChunks: Chunk[] = [];
  for (let i = 0; i < texts.length; i++) {
    allChunks.push(...chunk(texts[i], i));
  }
  return allChunks;
}
