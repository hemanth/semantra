import { g as getEngine, D as DEFAULT_MODEL, c as chunkAll, t as topK, a as cosineSim, b as DEFAULT_DTYPE } from "./semvec-iHPOsTC3.js";
import { S, d } from "./semvec-iHPOsTC3.js";
const corpusCache = /* @__PURE__ */ new WeakMap();
async function search(query, corpus, options = {}) {
  if (!query.trim() || corpus.length === 0) return [];
  const model = options.model ?? DEFAULT_MODEL;
  const field = options.field;
  const k = options.topK ?? 3;
  const threshold = options.threshold ?? 0;
  const engine = await getEngine(model, DEFAULT_DTYPE);
  const texts = corpus.map((item) => extractText(item, field));
  let corpusEmbeddings;
  const cached = corpusCache.get(corpus);
  if (cached && cached.model === model && cached.texts.length === texts.length) {
    corpusEmbeddings = cached.embeddings;
  } else {
    const chunks = chunkAll(texts);
    const chunkTexts = chunks.map((c) => c.text);
    const chunkEmbeddings = await engine.embedBatch(chunkTexts);
    corpusEmbeddings = new Array(texts.length);
    new Float64Array(texts.length).fill(-Infinity);
    for (let i = 0; i < chunks.length; i++) {
      const srcIdx = chunks[i].sourceIndex;
      if (!corpusEmbeddings[srcIdx]) {
        corpusEmbeddings[srcIdx] = chunkEmbeddings[i];
      }
    }
    corpusCache.set(corpus, {
      texts: [...texts],
      embeddings: corpusEmbeddings,
      model
    });
  }
  const queryVec = await engine.embed(query);
  const hits = topK(queryVec, corpusEmbeddings, k, threshold);
  return hits.map((hit) => ({
    text: texts[hit.index],
    score: Math.round(hit.score * 1e3) / 1e3,
    index: hit.index,
    item: corpus[hit.index]
  }));
}
async function similarity(a, b) {
  const engine = await getEngine();
  const [vecA, vecB] = await Promise.all([engine.embed(a), engine.embed(b)]);
  return Math.round(cosineSim(vecA, vecB) * 1e3) / 1e3;
}
async function embed(text, model) {
  const engine = await getEngine(model);
  return engine.embed(text);
}
function extractText(item, field) {
  if (typeof item === "string") return item;
  if (item && typeof item === "object") {
    const obj = item;
    if (field && typeof obj[field] === "string") {
      return obj[field];
    }
    for (const key of ["text", "content", "body", "question", "title", "description"]) {
      if (typeof obj[key] === "string") {
        return obj[key];
      }
    }
  }
  return String(item);
}
export {
  S as Semvec,
  d as SemvecError,
  embed,
  search,
  similarity
};
