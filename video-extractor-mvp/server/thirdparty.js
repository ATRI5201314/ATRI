/**
 * 第三方解析 API 适配层
 *
 * 通过环境变量启用，支持常见两种调用方式：
 *   POST JSON   { "url": "<分享链接>", "text": "<原始分享文案>" }
 *   GET         ?url=<encodeURIComponent(分享链接)>
 *
 * 环境变量：
 *   PARSE_API_URL          接口地址（不填则不启用第三方解析）
 *   PARSE_API_KEY          接口密钥（可选）
 *   PARSE_API_METHOD       POST | GET，默认 POST
 *   PARSE_API_KEY_HEADER   密钥放在请求头时的头名称，默认 Authorization
 *   PARSE_API_KEY_PARAM    密钥放 querystring 时的参数名，如 key（不填则走 header）
 *   PARSE_API_TIMEOUT      超时毫秒，默认 25000
 *   PARSE_API_NEED_PROXY   设为 1 时，所有结果走本机 /api/proxy 下载（防盗链用）
 *   PARSE_API_SEND_TEXT    设为 1 时请求体额外带上原始分享文案（默认只发 url）
 *   PARSE_API_USERNAME     账号密码登录用（与 PARSE_API_KEY 二选一）
 *   PARSE_API_PASSWORD
 *   PARSE_API_DEBUG        设为 1 时打印原始响应，便于对字段
 *
 * 返回字段做了多路径兼容（data / result / data.data 等嵌套，video / url / play 等命名），
 * 如果你的服务商字段特殊，只需在下面的 VIDEO_KEYS / IMAGE_KEYS 里补一个 key 即可。
 */

const ENDPOINT = process.env.PARSE_API_URL || '';
const API_KEY = process.env.PARSE_API_KEY || '';
const API_USER = process.env.PARSE_API_USERNAME || '';
const API_PASS = process.env.PARSE_API_PASSWORD || '';
const METHOD = (process.env.PARSE_API_METHOD || 'POST').toUpperCase();
const KEY_HEADER = process.env.PARSE_API_KEY_HEADER || 'Authorization';
const KEY_PARAM = process.env.PARSE_API_KEY_PARAM || '';
const URL_PARAM = process.env.PARSE_API_URL_PARAM || 'url';
const TIMEOUT = Number(process.env.PARSE_API_TIMEOUT || 25000);
const NEED_PROXY = process.env.PARSE_API_NEED_PROXY === '1';
const SEND_TEXT = process.env.PARSE_API_SEND_TEXT === '1';
const DEBUG = process.env.PARSE_API_DEBUG === '1';

const VIDEO_KEYS = [
  'video', 'videos', 'video_url', 'videoUrl', 'video_url_hd', 'videoUrlHd',
  'url', 'urls', 'play', 'play_url', 'playUrl', 'download_url', 'downloadUrl',
  'mp4', 'src', 'origin_url', 'originUrl', 'no_watermark_url', 'wmfree_url',
];

const IMAGE_KEYS = ['images', 'image', 'pics', 'pic', 'imgs', 'photos', 'image_urls', 'imageUrls'];

const COVER_KEYS = ['covers', 'cover', 'cover_url', 'coverUrl', 'poster', 'thumb', 'thumbnail', 'pic'];

const TITLE_KEYS = ['title', 'desc', 'description', 'text', 'name', 'caption'];

const MESSAGE_KEYS = ['msg', 'message', 'error', 'errmsg', 'error_msg', 'retmsg'];

function isEnabled() {
  return !!ENDPOINT;
}

/** 兼容 a.b[0] 这类路径取值 */
function getValue(obj, path) {
  return path.split('.').reduce((acc, key) => {
    if (acc == null) return undefined;
    const index = key.match(/^(\w+)\[(\d+)\]$/);
    if (index) {
      const list = acc[index[1]];
      return Array.isArray(list) ? list[Number(index[2])] : undefined;
    }
    return acc[key];
  }, obj);
}

function isHttpUrl(value) {
  return typeof value === 'string' && /^https?:\/\//i.test(value.trim());
}

/** 字符串 / 数组 / 对象里的地址统一取出一个可用 URL */
function pickUrl(value) {
  if (!value) return '';
  if (typeof value === 'string') return isHttpUrl(value) ? value.trim() : '';
  if (Array.isArray(value)) {
    for (const item of value) {
      const url = pickUrl(item);
      if (url) return url;
    }
    return '';
  }
  if (typeof value === 'object') {
    return pickUrl(value.url || value.play || value.src || value.download_url || value.url_list || value.urls);
  }
  return '';
}

function pickText(value) {
  if (!value) return '';
  if (typeof value === 'string') return value.trim();
  if (typeof value === 'number') return String(value);
  if (Array.isArray(value)) return pickText(value[0]);
  if (typeof value === 'object') return pickText(value.text || value.title || value.desc || value.message || value.msg);
  return '';
}

function toUrlList(value) {
  if (!value) return [];
  if (typeof value === 'string') return isHttpUrl(value) ? [value.trim()] : [];
  if (Array.isArray(value)) {
    return value.map((item) => pickUrl(item)).filter(Boolean);
  }
  if (typeof value === 'object') {
    const url = pickUrl(value);
    return url ? [url] : [];
  }
  return [];
}

/** 收集所有可能承载数据的对象，兼容各种嵌套写法 */
function collectRoots(body) {
  const roots = [];
  const seen = new Set();

  const push = (value) => {
    if (!value) return;
    if (typeof value === 'string') {
      if (isHttpUrl(value)) push({ url: value });
      return;
    }
    if (Array.isArray(value)) {
      value.slice(0, 3).forEach(push);
      return;
    }
    if (typeof value !== 'object' || seen.has(value)) return;
    seen.add(value);
    roots.push(value);
  };

  push(body);
  [
    'data', 'result', 'list', 'video', 'media',
    'data.data', 'data.result', 'data.list', 'data.media', 'result.data',
  ].forEach((path) => push(getValue(body, path)));

  return roots;
}

/** 有些接口会把入参 url 原样吐回来，要过滤掉，否则会把分享页当视频 */
function isEcho(candidate, shareUrl) {
  if (!candidate || !shareUrl) return false;
  const strip = (u) => u.replace(/^https?:\/\//i, '').replace(/\/+$/, '').toLowerCase();
  return strip(candidate) === strip(shareUrl);
}

function absolutize(url) {
  if (!url) return '';
  if (isHttpUrl(url)) return url;
  if (url.startsWith('/')) {
    try {
      return new URL(url, ENDPOINT).toString();
    } catch (e) {
      return url;
    }
  }
  return '';
}

function extractMessage(body) {
  const roots = collectRoots(body);
  for (const root of roots) {
    for (const key of MESSAGE_KEYS) {
      const text = pickText(root[key]);
      if (text) return text;
    }
  }
  return '';
}

/** 把第三方响应映射成内部统一结构 */
function normalize(body, shareUrl, platform) {
  const roots = collectRoots(body);

  let videoRaw = '';
  for (const root of roots) {
    for (const key of VIDEO_KEYS) {
      const candidate = pickUrl(root[key]);
      if (candidate && !isEcho(candidate, shareUrl)) {
        videoRaw = candidate;
        break;
      }
    }
    if (videoRaw) break;
  }

  let images = [];
  for (const root of roots) {
    for (const key of IMAGE_KEYS) {
      const list = toUrlList(root[key]).filter((u) => !isEcho(u, shareUrl));
      if (list.length) {
        images = list;
        break;
      }
    }
    if (images.length) break;
  }

  let title = '';
  let cover = '';
  for (const root of roots) {
    if (!title) {
      for (const key of TITLE_KEYS) {
        const text = pickText(root[key]);
        if (text && text.length <= 200) {
          title = text;
          break;
        }
      }
    }
    if (!cover) {
      for (const key of COVER_KEYS) {
        const url = pickUrl(root[key]);
        if (url) {
          cover = url;
          break;
        }
      }
    }
    if (title && cover) break;
  }

  const videoUrl = absolutize(videoRaw);
  if (!videoUrl && !images.length) return null;

  // 部分接口直接给出体积和分辨率，能省掉一次 HEAD 探测
  let sizeBytes = 0;
  let height = 0;
  for (const root of roots) {
    const bytes = Number(root.size_bytes || (root.video && root.video.size_bytes) || 0);
    if (bytes > 0) sizeBytes = bytes;
    const h = Number((root.video && root.video.height) || root.height || 0);
    if (h > 0) height = h;
    if (sizeBytes && height) break;
  }

  return {
    platform,
    title: title || '未命名作品',
    cover: absolutize(cover) || images[0] || '',
    videoUrl,
    images: images.map(absolutize).filter(Boolean),
    quality: height ? height + 'P' : '高清',
    sizeBytes,
    sources: videoUrl ? [videoUrl] : [],
    needProxy: NEED_PROXY,
    via: 'thirdparty',
  };
}

let sessionCookie = '';
let sessionAt = 0;

/** 账号密码登录换取 Cookie 会话（部分自托管/演示实例用这种方式鉴权） */
async function ensureSession() {
  if (!API_USER || !API_PASS) return '';
  if (sessionCookie && Date.now() - sessionAt < 30 * 60 * 1000) return sessionCookie;

  const loginPath = process.env.PARSE_API_LOGIN_PATH || '/api/v1/auth/login';
  const loginUrl = new URL(loginPath, ENDPOINT).toString();
  const res = await fetch(loginUrl, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ username: API_USER, password: API_PASS }),
    signal: AbortSignal.timeout(TIMEOUT),
  });
  if (!res.ok) throw new Error('解析服务登录失败（' + res.status + '）');

  const raw = res.headers.getSetCookie
    ? res.headers.getSetCookie()
    : [res.headers.get('set-cookie') || ''];
  sessionCookie = raw
    .filter(Boolean)
    .map((item) => item.split(';')[0])
    .join('; ');
  sessionAt = Date.now();
  if (DEBUG) console.log('[thirdparty] 登录成功, cookie=' + sessionCookie.slice(0, 40));
  return sessionCookie;
}

async function callApi(shareUrl, rawText, retried) {
  const headers = { 'content-type': 'application/json' };

  let target = ENDPOINT;
  if (API_KEY && KEY_PARAM) {
    const u = new URL(ENDPOINT);
    u.searchParams.set(KEY_PARAM, API_KEY);
    target = u.toString();
  } else if (API_KEY) {
    headers[KEY_HEADER] = API_KEY;
  }

  const cookie = await ensureSession();
  if (cookie) headers.cookie = cookie;

  let response;
  if (METHOD === 'GET') {
    const u = new URL(target);
    u.searchParams.set(URL_PARAM, shareUrl);
    response = await fetch(u.toString(), { headers, signal: AbortSignal.timeout(TIMEOUT) });
  } else {
    response = await fetch(target, {
      method: 'POST',
      headers,
      body: JSON.stringify(SEND_TEXT ? { url: shareUrl, text: rawText } : { url: shareUrl }),
      signal: AbortSignal.timeout(TIMEOUT),
    });
  }

  // 会话过期：清掉缓存重登一次
  if ((response.status === 401 || response.status === 403) && cookie && !retried) {
    sessionCookie = '';
    return callApi(shareUrl, rawText, true);
  }

  const text = await response.text();
  if (DEBUG) console.log('[thirdparty] status=' + response.status + ' body=' + text.slice(0, 800));

  let body;
  try {
    body = JSON.parse(text);
  } catch (e) {
    throw new Error('解析服务返回了非 JSON 内容');
  }

  if (!response.ok) {
    throw new Error(extractMessage(body) || '解析服务暂时不可用，请稍后重试');
  }
  return body;
}

/**
 * @returns {Promise<object|null>} 解析成功返回统一结构；失败抛错
 */
async function parseViaThirdParty({ shareUrl, text, platform }) {
  if (!isEnabled()) return null;

  const body = await callApi(shareUrl, text);
  const result = normalize(body, shareUrl, platform);
  if (!result) {
    throw new Error(extractMessage(body) || '解析服务没有返回可保存的内容');
  }
  return result;
}

module.exports = {
  isEnabled,
  parseViaThirdParty,
  // 便于单测/调试
  normalize,
  collectRoots,
};
