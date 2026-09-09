import { defineConfig, loadEnv } from 'vite'
import { resolve } from 'node:path'
import { brittainDevGateway } from './server/dev-plugin.js'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [react(), brittainDevGateway({ ...loadEnv(mode, resolve(process.cwd(), '..'), ''), ...loadEnv(mode, process.cwd(), '') })],
  // GitHub Pages serves this repo as a PROJECT site, at
  // https://lukemcb1128.github.io/brittain-model/ — not at the domain root, which
  // is the blog (a separate repo). Without this base, every built asset is
  // requested from /assets/... instead of /brittain-model/assets/..., and the
  // page loads as a blank white screen with 404s in the console.
  //
  // `npm run dev` overrides this to '/', so local development is unaffected.
  base: '/brittain-model/',
}))
