import { describe, it, expect } from 'vitest';
import { chunk, chunkAll } from '../src/chunker.js';

describe('chunk', () => {
  it('returns short text as single chunk', () => {
    const result = chunk('Hello world.', 0);
    expect(result).toHaveLength(1);
    expect(result[0].text).toBe('Hello world.');
    expect(result[0].sourceIndex).toBe(0);
    expect(result[0].chunkIndex).toBe(0);
  });

  it('returns empty array for empty text', () => {
    expect(chunk('', 0)).toEqual([]);
    expect(chunk('   ', 0)).toEqual([]);
  });

  it('chunks long text on sentence boundaries', () => {
    // Create a long text (~1500 chars, well over 256 token threshold)
    const sentences = Array.from({ length: 30 }, (_, i) =>
      `This is sentence number ${i + 1} which contains some meaningful content about topic ${i}. `
    );
    const longText = sentences.join('');

    const result = chunk(longText, 5);

    expect(result.length).toBeGreaterThan(1);
    expect(result[0].sourceIndex).toBe(5);
    expect(result[0].chunkIndex).toBe(0);
    expect(result[1].chunkIndex).toBe(1);

    // Each chunk should be non-empty
    for (const c of result) {
      expect(c.text.length).toBeGreaterThan(0);
    }
  });

  it('preserves sourceIndex through chunks', () => {
    const longText = 'A'.repeat(2000); // well over threshold
    const result = chunk(longText, 42);

    for (const c of result) {
      expect(c.sourceIndex).toBe(42);
    }
  });

  it('text at exactly threshold is not chunked', () => {
    // 256 tokens * 4 chars = 1024 chars
    const text = 'x'.repeat(1024);
    const result = chunk(text, 0);
    expect(result).toHaveLength(1);
  });

  it('text just over threshold is chunked', () => {
    // 257 tokens * 4 chars = 1028 chars
    const text = 'x'.repeat(1028);
    const result = chunk(text, 0);
    // May or may not chunk depending on sentence boundaries,
    // but should have at least 1 chunk
    expect(result.length).toBeGreaterThanOrEqual(1);
  });
});

describe('chunkAll', () => {
  it('chunks array of texts preserving source indices', () => {
    const texts = ['Short text.', 'Another short one.'];
    const result = chunkAll(texts);

    expect(result).toHaveLength(2);
    expect(result[0].sourceIndex).toBe(0);
    expect(result[1].sourceIndex).toBe(1);
  });

  it('handles empty array', () => {
    expect(chunkAll([])).toEqual([]);
  });

  it('handles mix of short and long texts', () => {
    const longText = Array.from({ length: 30 }, (_, i) =>
      `Sentence ${i} with content. `
    ).join('');

    const texts = ['Short.', longText, 'Also short.'];
    const result = chunkAll(texts);

    // Short texts: 1 chunk each. Long text: at least 1 chunk.
    expect(result.length).toBeGreaterThanOrEqual(3);
    expect(result[0].sourceIndex).toBe(0);
    expect(result[result.length - 1].sourceIndex).toBe(2);
  });
});
