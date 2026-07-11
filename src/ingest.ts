/**
 * Document ingestion — fetch + parse URLs for Semvec.fromURL().
 * Supports JSON arrays, plain text, and Markdown.
 * Never used for string detection — URLs are always explicit.
 */

import { SemvecError } from './errors.js';
import type { Document } from './types.js';

/**
 * Fetch a URL and parse it into Document objects.
 *
 * @param url - The URL to fetch.
 * @param field - For JSON arrays of objects, which field contains the text to embed.
 * @returns Array of extracted documents.
 * @throws SemvecError with code 'FETCH_FAILED' on network or parse errors.
 */
export async function ingestURL(url: string, field?: string): Promise<Document[]> {
  let response: Response;

  try {
    response = await fetch(url);
  } catch (err) {
    throw new SemvecError(
      'FETCH_FAILED',
      `Failed to fetch "${url}": ${err instanceof Error ? err.message : 'Network error'}`,
      { url, originalError: err },
    );
  }

  if (!response.ok) {
    throw new SemvecError(
      'FETCH_FAILED',
      `Fetch "${url}" returned HTTP ${response.status}`,
      { url, status: response.status },
    );
  }

  const contentType = response.headers.get('content-type') || '';
  const text = await response.text();

  // JSON response
  if (contentType.includes('json') || url.endsWith('.json')) {
    return parseJSON(text, url, field);
  }

  // Markdown — strip to plain text
  if (contentType.includes('markdown') || url.endsWith('.md')) {
    return [{ text: stripMarkdown(text), source: url }];
  }

  // Plain text fallback
  return [{ text, source: url }];
}

/**
 * Parse a JSON response into documents.
 * Handles: JSON array of strings, array of objects (with field extraction),
 * or a single object with an array property.
 */
function parseJSON(raw: string, source: string, field?: string): Document[] {
  let data: unknown;

  try {
    data = JSON.parse(raw);
  } catch {
    throw new SemvecError(
      'FETCH_FAILED',
      `Failed to parse JSON from "${source}"`,
      { url: source },
    );
  }

  // Direct array
  if (Array.isArray(data)) {
    return extractFromArray(data, source, field);
  }

  // Object with a single array property — auto-detect
  if (data && typeof data === 'object' && !Array.isArray(data)) {
    const obj = data as Record<string, unknown>;
    const arrayKeys = Object.keys(obj).filter((k) => Array.isArray(obj[k]));

    if (arrayKeys.length === 1) {
      return extractFromArray(obj[arrayKeys[0]] as unknown[], source, field);
    }

    // Multiple array keys — try common names
    for (const key of ['data', 'items', 'results', 'entries', 'docs', 'documents']) {
      if (Array.isArray(obj[key])) {
        return extractFromArray(obj[key] as unknown[], source, field);
      }
    }
  }

  // Fallback: stringify the whole thing
  return [{ text: JSON.stringify(data), source }];
}

/** Extract documents from a parsed array. */
function extractFromArray(arr: unknown[], source: string, field?: string): Document[] {
  const docs: Document[] = [];

  for (const item of arr) {
    if (typeof item === 'string') {
      docs.push({ text: item, source });
    } else if (item && typeof item === 'object') {
      const obj = item as Record<string, unknown>;

      if (field && typeof obj[field] === 'string') {
        docs.push({ text: obj[field] as string, source, meta: obj });
      } else {
        // Try common text field names
        for (const key of ['text', 'content', 'body', 'question', 'title', 'description']) {
          if (typeof obj[key] === 'string') {
            docs.push({ text: obj[key] as string, source, meta: obj });
            break;
          }
        }
      }
    }
  }

  return docs;
}

/** Strip basic Markdown formatting to plain text. */
function stripMarkdown(md: string): string {
  return md
    // Remove headers
    .replace(/^#{1,6}\s+/gm, '')
    // Remove bold/italic
    .replace(/\*{1,3}([^*]+)\*{1,3}/g, '$1')
    .replace(/_{1,3}([^_]+)_{1,3}/g, '$1')
    // Remove links, keep text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    // Remove images
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')
    // Remove code blocks
    .replace(/```[\s\S]*?```/g, '')
    // Remove inline code
    .replace(/`([^`]+)`/g, '$1')
    // Remove blockquotes
    .replace(/^>\s+/gm, '')
    // Remove horizontal rules
    .replace(/^[-*_]{3,}\s*$/gm, '')
    // Collapse multiple newlines
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}
