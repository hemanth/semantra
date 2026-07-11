import { describe, it, expect } from 'vitest';
import { cosineSim, l2Normalize, topK } from '../src/math.js';

describe('cosineSim', () => {
  it('returns 1 for identical normalized vectors', () => {
    const a = new Float32Array([0.6, 0.8]);
    expect(cosineSim(a, a)).toBeCloseTo(1, 5);
  });

  it('returns 0 for orthogonal vectors', () => {
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([0, 1]);
    expect(cosineSim(a, b)).toBeCloseTo(0, 5);
  });

  it('returns -1 for opposite vectors', () => {
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([-1, 0]);
    expect(cosineSim(a, b)).toBeCloseTo(-1, 5);
  });

  it('returns 0 for mismatched lengths', () => {
    const a = new Float32Array([1, 0, 0]);
    const b = new Float32Array([1, 0]);
    expect(cosineSim(a, b)).toBe(0);
  });

  it('returns 0 for empty vectors', () => {
    const a = new Float32Array([]);
    expect(cosineSim(a, a)).toBe(0);
  });

  it('clamps to [-1, 1] range', () => {
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([1, 0]);
    const score = cosineSim(a, b);
    expect(score).toBeGreaterThanOrEqual(-1);
    expect(score).toBeLessThanOrEqual(1);
  });
});

describe('l2Normalize', () => {
  it('normalizes a vector to unit length', () => {
    const v = l2Normalize(new Float32Array([3, 4]));
    const norm = Math.sqrt(v[0] ** 2 + v[1] ** 2);
    expect(norm).toBeCloseTo(1, 5);
    expect(v[0]).toBeCloseTo(0.6, 5);
    expect(v[1]).toBeCloseTo(0.8, 5);
  });

  it('handles zero vector without NaN', () => {
    const v = l2Normalize(new Float32Array([0, 0, 0]));
    expect(v[0]).toBe(0);
    expect(v[1]).toBe(0);
    expect(v[2]).toBe(0);
  });

  it('normalizes in place', () => {
    const original = new Float32Array([3, 4]);
    const result = l2Normalize(original);
    expect(result).toBe(original); // same reference
  });
});

describe('topK', () => {
  const corpus = [
    l2Normalize(new Float32Array([1, 0])),
    l2Normalize(new Float32Array([0.9, 0.1])),
    l2Normalize(new Float32Array([0, 1])),
    l2Normalize(new Float32Array([-1, 0])),
  ];

  it('returns top-K results sorted by score descending', () => {
    const query = l2Normalize(new Float32Array([1, 0]));
    const results = topK(query, corpus, 2);

    expect(results).toHaveLength(2);
    expect(results[0].index).toBe(0); // most similar
    expect(results[0].score).toBeCloseTo(1, 2);
    expect(results[1].index).toBe(1); // second most
  });

  it('filters by threshold', () => {
    const query = l2Normalize(new Float32Array([1, 0]));
    const results = topK(query, corpus, 10, 0.5);

    // Only indices 0 and 1 should pass threshold 0.5
    expect(results.length).toBeLessThanOrEqual(2);
    for (const r of results) {
      expect(r.score).toBeGreaterThanOrEqual(0.5);
    }
  });

  it('returns empty for empty corpus', () => {
    const query = new Float32Array([1, 0]);
    expect(topK(query, [], 5)).toEqual([]);
  });

  it('returns fewer than K if corpus is smaller', () => {
    const query = l2Normalize(new Float32Array([1, 0]));
    const small = [l2Normalize(new Float32Array([1, 0]))];
    const results = topK(query, small, 5);
    expect(results).toHaveLength(1);
  });
});
