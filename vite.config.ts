import { defineConfig } from 'vite';
import dts from 'vite-plugin-dts';
import { resolve } from 'path';

export default defineConfig({
  plugins: [dts({ rollupTypes: true })],
  build: {
    lib: {
      entry: {
        index: resolve(__dirname, 'src/index.ts'),
        react: resolve(__dirname, 'src/react.ts'),
      },
      formats: ['es'],
    },
    rollupOptions: {
      external: [
        '@huggingface/transformers',
        'react',
      ],
    },
    target: 'es2022',
    minify: false,
  },
});
