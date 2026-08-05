import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // GitHub Pages serves this repo as a PROJECT site, at
  // https://lukemcb1128.github.io/brittain-model/ — not at the domain root, which
  // is the blog (a separate repo). Without this base, every built asset is
  // requested from /assets/... instead of /brittain-model/assets/..., and the
  // page loads as a blank white screen with 404s in the console.
  //
  // `npm run dev` overrides this to '/', so local development is unaffected.
  base: '/brittain-model/',
})
