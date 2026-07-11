# semantra

[![npm](https://img.shields.io/npm/v/semantra)](https://www.npmjs.com/package/semantra)

**Semantic vector search, entirely in the browser.**

[**Live Demo →**](https://hemanth.github.io/semantra/)

One function. One await. No backend, no API keys, no GPU.

```js
import { search } from 'semantra';

const results = await search('forgot password', [
  'How to reset your password',
  'Track your delivery status',
  'Update billing address',
]);
// → [{ text: 'How to reset your password', score: 0.88, index: 0 }]
```

Model loads silently on first call (~30 MB, one-time), cached forever after. Every subsequent call: **~8 ms**.

## Install

```bash
npm install semantra
```

## API

### `search(query, corpus, options?)` — semantic search

```js
import { search } from 'semantra';

// Strings
const results = await search('query', ['doc1', 'doc2', 'doc3']);

// Objects — specify which field to search
const results = await search('billing', products, { field: 'description' });

// Same array reference = embeddings reused (instant)
const docs = ['doc1', 'doc2', 'doc3'];
await search('query1', docs); // embeds (~500ms)
await search('query2', docs); // instant — same ref

// Options
await search('query', docs, {
  topK: 5,         // max results (default: 3)
  threshold: 0.3,  // min score (default: 0)
  model: 'Snowflake/snowflake-arctic-embed-s', // override model
});
```

### `similarity(a, b)` — compare two texts

```js
import { similarity } from 'semantra';

const score = await similarity('forgot password', 'reset my password');
// → 0.87
```

### `embed(text)` — raw vector (power users)

```js
import { embed } from 'semantra';

const vec = await embed('semantic search');
// → Float32Array(384), L2-normalized
```

### `Semvec` class — persistent corpus

For when you need incremental updates, IndexedDB caching, or progress callbacks:

```js
import { Semvec } from 'semantra';

// From local data
const sv = new Semvec(['doc1', 'doc2', 'doc3']);
const results = await sv.search('query'); // auto-inits

// From a URL (explicit — never guessed from strings)
const sv = await Semvec.fromURL('https://example.com/faq.json', {
  field: 'question',
});

// Incremental updates
const id = await sv.add('New document');
sv.remove(id); // by content-hash ID
sv.remove(2);  // or by index

// Progress callback
const sv = new Semvec(data, {
  onProgress: ({ phase, percent }) => {
    console.log(`${phase}: ${percent}%`);
  },
});

// Cleanup
sv.destroy();
```

### React hook

```tsx
import { useSearch } from 'semantra/react';

function SearchBox() {
  const { search, results, loading } = useSearch(faqDocs);

  return (
    <div>
      <input onChange={e => search(e.target.value)} />
      {loading && <span>Searching...</span>}
      {results.map(r => <p key={r.index}>{r.text} — {r.score}</p>)}
    </div>
  );
}
```

## Models

Default: **Snowflake Arctic-Embed-S** (33M params, ~30 MB). Swap to any HuggingFace ONNX model:

```js
const sv = new Semvec(data, { model: 'Snowflake/snowflake-arctic-embed-m' });
```

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| **Arctic-Embed-S** (default) | ~30 MB | ★★★★ | ~8 ms |
| Arctic-Embed-M | ~110 MB | ★★★★★ | ~20 ms |
| BGE-small-en-v1.5 | ~35 MB | ★★★★ | ~8 ms |
| all-MiniLM-L6-v2 | ~23 MB | ★★★ | ~5 ms |

## Error Handling

```js
import { search, SemvecError } from 'semantra';

try {
  await search('query', docs);
} catch (e) {
  if (e instanceof SemvecError) {
    switch (e.code) {
      case 'MODEL_OFFLINE':
        // Device offline, model not cached yet
        break;
      case 'FETCH_FAILED':
        // URL fetch failed (Semvec.fromURL)
        break;
      case 'WASM_UNSUPPORTED':
        // Browser doesn't support WASM
        break;
    }
  }
}
```

## How It Works

- **Engine**: ONNX Runtime via `@huggingface/transformers` — runs on WASM, auto-upgrades to WebGPU when available
- **Caching**: Embeddings cached in IndexedDB (scoped per model + namespace). Falls back to in-memory in private browsing.
- **Chunking**: Long texts auto-split on sentence boundaries. Short texts pass through unchanged.
- **No DOM coupling**: Works in Web Workers, service workers, and Node.js (with WASM support).

## License

MIT
