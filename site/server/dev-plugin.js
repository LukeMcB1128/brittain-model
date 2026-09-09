import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { handleApi } from './gateway.js';

export function brittainDevGateway(values) {
  return {
    name: 'brittain-local-gateway',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        if (!req.url?.startsWith('/api/')) return next();
        if (!['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(req.socket.remoteAddress)) {
          res.writeHead(403); res.end('Local development access only.'); return;
        }
        const abort = new AbortController();
        res.on('close', () => { if (!res.writableFinished) abort.abort(); });
        const headers = new Headers(req.headers);
        headers.set('oai-authenticated-user-id', 'local-developer');
        try {
          const request = new Request(`http://${req.headers.host}${req.url}`, {
            method: req.method, headers, signal: abort.signal,
            ...(!['GET', 'HEAD'].includes(req.method) ? { body: Readable.toWeb(req), duplex: 'half' } : {}),
          });
          const response = await handleApi(request, { BRITTAIN4_API_KEY: values.BRITTAIN4_API_KEY || values.BRITTAIN_API_KEY });
          res.writeHead(response.status, Object.fromEntries(response.headers));
          if (response.body) await pipeline(Readable.fromWeb(response.body), res);
          else res.end();
        } catch {
          if (!res.headersSent) res.writeHead(502, { 'Content-Type': 'application/json' });
          if (!res.destroyed) res.end(JSON.stringify({ error: 'The local chat connection ended.' }));
        }
      });
    },
  };
}
