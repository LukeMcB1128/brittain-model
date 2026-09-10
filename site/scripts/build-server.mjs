import { mkdir, readFile, readdir, writeFile, copyFile } from 'node:fs/promises';
import { join, extname } from 'node:path';
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.png': 'image/png', '.svg': 'image/svg+xml' };
const assets = {};
async function collect(dir, prefix = '') {
  for (const file of await readdir(dir, { withFileTypes: true })) {
    const name = `${prefix}/${file.name}`;
    if (file.isDirectory()) await collect(join(dir, file.name), name);
    else assets[name] = { type: types[extname(file.name)] || 'application/octet-stream', data: (await readFile(join(dir, file.name))).toString('base64') };
  }
}
await collect('dist/client');
await mkdir('dist/server', { recursive: true });
await copyFile('server/gateway.js', 'dist/server/gateway.js');
await copyFile('server/tools.js', 'dist/server/tools.js');
await writeFile('dist/server/assets.js', `export default ${JSON.stringify(assets)};`);
await writeFile('dist/server/index.js', `import { handleApi } from './gateway.js';
import assets from './assets.js';
export default { async fetch(request, env) {
  const path = new URL(request.url).pathname;
  if (path.startsWith('/api/')) return handleApi(request, env);
  if (!['GET', 'HEAD'].includes(request.method)) return new Response('Method not allowed', {status:405});
  const asset = assets[path === '/' ? '/index.html' : path];
  if (!asset) return new Response('Not found', {status:404});
  const bytes = request.method === 'HEAD' ? null : Uint8Array.from(atob(asset.data), c => c.charCodeAt(0));
  return new Response(bytes, {headers:{'Content-Type':asset.type,'Cache-Control':path.startsWith('/assets/')?'public, max-age=31536000, immutable':'no-cache','X-Content-Type-Options':'nosniff'}});
}};`);
