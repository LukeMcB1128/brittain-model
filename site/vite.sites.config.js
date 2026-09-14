import { resolve } from 'node:path';
import react from '@vitejs/plugin-react';
import { defineConfig, loadEnv } from 'vite';
import { brittainDevGateway } from './server/dev-plugin.js';

// This config only rebuilds the current ChatGPT Sites rollback deployment.
// The standalone application uses vite.config.js and server/worker.js.
export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    brittainDevGateway({
      ...loadEnv(mode, resolve(process.cwd(), '..'), ''),
      ...loadEnv(mode, process.cwd(), ''),
    }),
  ],
  base: '/',
}));
