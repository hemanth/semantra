/** Error codes for SemvecError. */
export type SemvecErrorCode =
  | 'MODEL_OFFLINE'
  | 'FETCH_FAILED'
  | 'WASM_UNSUPPORTED'
  | 'MODEL_LOAD_FAILED'
  | 'DESTROYED';

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
export class SemvecError extends Error {
  readonly code: SemvecErrorCode;
  readonly detail?: unknown;

  constructor(code: SemvecErrorCode, message: string, detail?: unknown) {
    super(message);
    this.name = 'SemvecError';
    this.code = code;
    this.detail = detail;
  }
}
