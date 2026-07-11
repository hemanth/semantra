import { describe, it, expect } from 'vitest';
import { SemvecError } from '../src/errors.js';

describe('SemvecError', () => {
  it('creates error with code and message', () => {
    const err = new SemvecError('MODEL_OFFLINE', 'Device is offline');

    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(SemvecError);
    expect(err.code).toBe('MODEL_OFFLINE');
    expect(err.message).toBe('Device is offline');
    expect(err.name).toBe('SemvecError');
  });

  it('includes detail when provided', () => {
    const err = new SemvecError('FETCH_FAILED', 'HTTP 404', { url: 'https://example.com', status: 404 });

    expect(err.detail).toEqual({ url: 'https://example.com', status: 404 });
  });

  it('detail is undefined when not provided', () => {
    const err = new SemvecError('WASM_UNSUPPORTED', 'No WASM');
    expect(err.detail).toBeUndefined();
  });

  it('can be caught with instanceof', () => {
    const err = new SemvecError('DESTROYED', 'Instance destroyed');

    try {
      throw err;
    } catch (e) {
      expect(e instanceof SemvecError).toBe(true);
      if (e instanceof SemvecError) {
        expect(e.code).toBe('DESTROYED');
      }
    }
  });

  it('works with switch on code', () => {
    const err = new SemvecError('MODEL_LOAD_FAILED', 'Bad model');
    let matched = false;

    switch (err.code) {
      case 'MODEL_LOAD_FAILED':
        matched = true;
        break;
    }

    expect(matched).toBe(true);
  });
});
