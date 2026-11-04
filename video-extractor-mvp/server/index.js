const http = require('http');
const fs = require('fs');
const path = require('path');
const { Readable } = require('stream');
const { parse, MOBILE_UA } = require('./parser');
const thirdparty = require('./thirdparty');

const PORT = Number(process.env.PORT || 3000);
const PUBLIC_DIR = path.join(__dirname, 'public');

const MIME = {
  '.mp4': 'video/mp4',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.json': 'application/json',
};

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
    'Access-Control-Allow-Origin': '*',
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let raw = '';
    req.on('data', (chunk) => {
      raw += chunk;
      if (raw.length > 1024 * 64) {
        reject(new Error('请求体过大'));
        req.destroy();
      }
    });
    req.on('end', () => resolve(raw));
    req.on('error', reject);
  });
}

function isBlockedHost(hostname) {
  const host = hostname.toLowerCase();
  if (host === 'localhost' || host.endsWith('.local')) return true;
  if (/^127\./.test(host) || /^10\./.test(host) || /^192\.168\./.test(host)) return true;
  if (/^169\.254\./.test(host)) return true;
  return /^172\.(1[6-9]|2\d|3[01])\./.test(host);
}

function formatBytes(bytes) {
  const mb = bytes / 1024 / 1024;
  return mb >= 1 ? mb.toFixed(2) + 'MB' : Math.round(bytes / 1024) + 'KB';
}

async function estimateSize(videoUrl) {
  try {
    const res = await fetch(videoUrl, {
      method: 'HEAD',
      headers: { 'User-Agent': MOBILE_UA },
      signal: AbortSignal.timeout(5000),
    });
    const bytes = Number(res.headers.get('content-length') || 0);
    if (!bytes) return '未知';
    const mb = bytes / 1024 / 1024;
    return mb >= 1 ? mb.toFixed(2) + 'MB' : Math.round(bytes / 1024) + 'KB';
  } catch (e) {
    return '未知';
  }
}

function serveStatic(req, res, pathname) {
  let filePath;
  try {
    filePath = path.normalize(path.join(PUBLIC_DIR, decodeURIComponent(pathname)));
  } catch (e) {
    return sendJson(res, 400, { ok: false, error: '非法路径' });
  }
  if (!filePath.startsWith(PUBLIC_DIR)) {
    return sendJson(res, 403, { ok: false, error: '禁止访问' });
  }

  fs.stat(filePath, (err, stat) => {
    if (err || !stat.isFile()) {
      return sendJson(res, 404, { ok: false, error: '文件不存在' });
    }

    const type = MIME[path.extname(filePath).toLowerCase()] || 'application/octet-stream';
    const total = stat.size;
    const range = req.headers.range;

    if (req.method === 'HEAD') {
      res.writeHead(200, {
        'Content-Type': type,
        'Content-Length': total,
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'public, max-age=3600',
      });
      return res.end();
    }

    if (range) {
      const hit = /bytes=(\d*)-(\d*)/.exec(range);
      const start = hit && hit[1] ? Number(hit[1]) : 0;
      const end = hit && hit[2] ? Number(hit[2]) : total - 1;
      if (start >= total || end >= total || start > end) {
        res.writeHead(416, { 'Content-Range': 'bytes */' + total });
        return res.end();
      }
      res.writeHead(206, {
        'Content-Type': type,
        'Content-Length': end - start + 1,
        'Content-Range': 'bytes ' + start + '-' + end + '/' + total,
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'public, max-age=3600',
      });
      return fs.createReadStream(filePath, { start, end }).pipe(res);
    }

    res.writeHead(200, {
      'Content-Type': type,
      'Content-Length': total,
      'Accept-Ranges': 'bytes',
      'Cache-Control': 'public, max-age=3600',
    });
    fs.createReadStream(filePath).pipe(res);
  });
}

async function handleProxy(req, res, requestUrl) {
  const target = requestUrl.searchParams.get('url');
  if (!target) return sendJson(res, 400, { ok: false, error: '缺少 url 参数' });

  let parsed;
  try {
    parsed = new URL(target);
  } catch (e) {
    return sendJson(res, 400, { ok: false, error: 'url 不合法' });
  }
  if (!/^https?:$/.test(parsed.protocol) || isBlockedHost(parsed.hostname)) {
    return sendJson(res, 403, { ok: false, error: '该地址不允许代理' });
  }

  try {
    const upstream = await fetch(parsed.toString(), {
      headers: {
        'User-Agent': MOBILE_UA,
        Referer: 'https://www.douyin.com/',
      },
      signal: AbortSignal.timeout(20000),
    });
    if (!upstream.ok || !upstream.body) {
      return sendJson(res, 502, { ok: false, error: '源站返回 ' + upstream.status });
    }

    const headers = {
      'Content-Type': upstream.headers.get('content-type') || 'video/mp4',
      'Access-Control-Allow-Origin': '*',
    };
    const length = upstream.headers.get('content-length');
    if (length) headers['Content-Length'] = length;

    res.writeHead(200, headers);
    Readable.fromWeb(upstream.body).pipe(res);
  } catch (e) {
    sendJson(res, 502, { ok: false, error: '代理失败：' + e.message });
  }
}

async function handleParse(req, res) {
  let body;
  try {
    body = JSON.parse((await readBody(req)) || '{}');
  } catch (e) {
    return sendJson(res, 400, { ok: false, error: '请求体不是合法 JSON' });
  }

  const origin = 'http://' + (req.headers.host || '127.0.0.1:' + PORT);

  try {
    const result = await parse(body.text);
    const absolute = (u) => (u && u.startsWith('/') ? origin + u : u);

    result.cover = absolute(result.cover);
    result.videoUrl = absolute(result.videoUrl);
    result.images = (result.images || []).map(absolute);
    result.sources = (result.sources || []).map(absolute);
    result.downloadUrl =
      result.needProxy && result.videoUrl
        ? origin + '/api/proxy?url=' + encodeURIComponent(result.videoUrl)
        : result.videoUrl;
    result.size = result.sizeBytes ? formatBytes(result.sizeBytes) : await estimateSize(result.videoUrl);

    sendJson(res, 200, { ok: true, data: result });
  } catch (e) {
    sendJson(res, 200, { ok: false, error: e.message || '解析失败' });
  }
}

const server = http.createServer(async (req, res) => {
  const requestUrl = new URL(req.url, 'http://' + (req.headers.host || 'localhost'));

  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    });
    return res.end();
  }

  if (requestUrl.pathname === '/health') {
    return sendJson(res, 200, { ok: true, service: 'video-extractor-mvp' });
  }

  if (requestUrl.pathname === '/api/parse' && req.method === 'POST') {
    return handleParse(req, res);
  }

  if (requestUrl.pathname === '/api/proxy' && req.method === 'GET') {
    return handleProxy(req, res, requestUrl);
  }

  if (req.method === 'GET' || req.method === 'HEAD') {
    return serveStatic(req, res, requestUrl.pathname);
  }

  sendJson(res, 405, { ok: false, error: '不支持的方法' });
});

server.listen(PORT, () => {
  console.log('▶ 解析服务已启动: http://127.0.0.1:' + PORT);
  console.log('  健康检查: http://127.0.0.1:' + PORT + '/health');
  console.log(
    '  第三方解析: ' +
      (thirdparty.isEnabled() ? '已启用 -> ' + process.env.PARSE_API_URL : '未配置（PARSE_API_URL 为空）')
  );
});
