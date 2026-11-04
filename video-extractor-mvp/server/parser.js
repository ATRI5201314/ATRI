const MOBILE_UA =
  'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 ' +
  '(KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1';

const SHARE_URL_RE = /https?:\/\/[^\s"'<>，。！？、）】]+/i;

/** 从分享文案里抠出第一个链接 */
function extractShareUrl(text) {
  if (!text) return null;
  const matched = String(text).match(SHARE_URL_RE);
  if (!matched) return null;
  return matched[0].replace(/[),.;]+$/, '');
}

function detectPlatform(rawUrl) {
  let host = '';
  try {
    host = new URL(rawUrl).hostname.toLowerCase();
  } catch (e) {
    return 'unknown';
  }
  if (/(douyin|iesdouyin)\./.test(host) || host.endsWith('douyin.com')) return 'douyin';
  if (/(kuaishou|gifshow|chenzhongtech)\./.test(host) || host.endsWith('kuaishou.com')) return 'kuaishou';
  if (/(xiaohongshu|xhslink)\./.test(host) || host.endsWith('xiaohongshu.com')) return 'xiaohongshu';
  if (/(bilibili|b23)\./.test(host) || host.endsWith('bilibili.com')) return 'bilibili';
  return 'unknown';
}

async function fetchWithTimeout(url, options = {}, ms = 8000) {
  const res = await fetch(url, {
    ...options,
    headers: { 'User-Agent': MOBILE_UA, ...(options.headers || {}) },
    signal: AbortSignal.timeout(ms),
  });
  return res;
}

/** 短链 302 跟随，拿到真实分享页地址 */
async function resolveRedirect(shareUrl, maxHops = 5) {
  let current = shareUrl;
  for (let hop = 0; hop < maxHops; hop += 1) {
    const res = await fetchWithTimeout(current, { redirect: 'manual' });
    if (res.status >= 300 && res.status < 400) {
      const location = res.headers.get('location');
      if (res.body) res.body.cancel().catch(() => {});
      if (!location) return current;
      current = new URL(location, current).toString();
      continue;
    }
    if (res.body) res.body.cancel().catch(() => {});
    return current;
  }
  return current;
}

/**
 * 抖音适配层（尽力而为）。
 * 说明：抖音公开的 iteminfo 接口已停止返回数据，分享页视频数据由前端动态加载，
 * 正式接口需要 a_bogus / X-Bogus 签名和 Cookie，无法靠简单请求稳定拿到。
 * 所以这里的策略是：先试旧接口 -> 再试分享页 og:video -> 都失败就返回明确错误。
 * 想稳定商用，需要在此处接入自建签名服务或购买第三方解析 API。
 */
async function parseDouyin(shareUrl) {
  const finalUrl = await resolveRedirect(shareUrl);
  const idMatch =
    finalUrl.match(/\/video\/(\d+)/) ||
    finalUrl.match(/modal_id=(\d+)/) ||
    shareUrl.match(/\/video\/(\d+)/);
  const awemeId = idMatch && idMatch[1];
  if (!awemeId) {
    throw new Error('没能从抖音链接里识别出作品 ID，请确认复制的是完整分享链接');
  }

  try {
    const api =
      'https://www.iesdouyin.com/web/api/v2/aweme/iteminfo/?item_ids=' + awemeId;
    const res = await fetchWithTimeout(api, {
      headers: { Referer: 'https://www.douyin.com/' },
    });
    if (!res.ok) throw new Error('抖音详情接口返回 ' + res.status);

    const raw = await res.text();
    let json = null;
    try {
      json = JSON.parse(raw);
    } catch (e) {
      json = null;
    }

    const item = json && json.item_list && json.item_list[0];
    if (item) {
      const playList = (item.video && item.video.play_addr && item.video.play_addr.url_list) || [];
      const playUrl = playList[0] || '';
      if (!playUrl && !(item.images || []).length) {
        throw new Error('作品里没有可用的视频地址');
      }
      return {
        platform: 'douyin',
        title: item.desc || '未命名作品',
        cover:
          (item.video && item.video.cover && item.video.cover.url_list && item.video.cover.url_list[0]) || '',
        videoUrl: playUrl.replace('/playwm/', '/play/'),
        images: (item.images || []).map((img) => (img.url_list || [])[0]).filter(Boolean),
        quality: /1080/.test(playUrl) ? '1080P' : '高清',
        sources: playList.map((u) => u.replace('/playwm/', '/play/')),
        needProxy: true,
      };
    }
  } catch (e) {
    console.warn('[douyin] iteminfo 失败:', e.message);
  }

  // 兜底：直接解析分享页里的 og:video
  try {
    return await parseGeneric(finalUrl);
  } catch (e) {
    console.warn('[douyin] 分享页兜底失败:', e.message);
    throw new Error('这个链接暂时获取不到视频，请稍后重试或换一条链接');
  }
}

/** 通用兜底：抓分享页的 og:video / og:image */
async function parseGeneric(shareUrl) {
  const res = await fetchWithTimeout(shareUrl, { redirect: 'follow' });
  if (!res.ok) throw new Error('分享页返回 ' + res.status);
  const html = await res.text();

  const og = (name) => {
    const re = new RegExp(
      '<meta[^>]+(?:property|name)=["\']' + name + '["\'][^>]+content=["\']([^"\']+)["\']',
      'i'
    );
    const hit = html.match(re);
    return hit ? hit[1] : '';
  };

  const videoUrl = og('og:video') || og('og:video:url') || og('twitter:player:stream');
  if (!videoUrl) {
    throw new Error('这个平台暂时解析失败，分享页里没有公开的视频地址');
  }

  return {
    platform: detectPlatform(shareUrl),
    title: og('og:title') || '未命名作品',
    cover: og('og:image') || '',
    videoUrl,
    images: [],
    quality: '未知',
    sources: [videoUrl],
    needProxy: true,
  };
}

/** 本地演示数据：不依赖外网，保证整条链路能跑通 */
function demoResult() {
  return {
    platform: 'demo',
    title: '演示作品 · 本地可跑通的完整流程（非真实平台内容）',
    cover: '/demo-cover.jpg',
    videoUrl: '/demo.mp4',
    images: ['/demo-img-1.jpg', '/demo-img-2.jpg'],
    quality: '720P',
    sources: ['/demo.mp4'],
    needProxy: false,
    demo: true,
  };
}

/**
 * 统一入口
 * @param {string} text 分享文案或纯链接
 */
async function parse(text) {
  const raw = String(text || '').trim();
  if (!raw) throw new Error('内容为空');

  // 演示口令：输入 demo 或 ---- 即可体验本地流程
  if (/^demo$/i.test(raw) || /example\.com\/demo/i.test(raw)) {
    return demoResult();
  }

  const shareUrl = extractShareUrl(raw);
  if (!shareUrl) {
    throw new Error('没有识别到链接，请把整段分享文案粘进来');
  }

  const platform = detectPlatform(shareUrl);
  let result;

  if (platform === 'douyin') {
    result = await parseDouyin(shareUrl);
  } else {
    result = await parseGeneric(shareUrl);
  }

  if (!result.videoUrl && !result.images.length) {
    throw new Error('没有解析到可保存的内容');
  }
  return result;
}

module.exports = {
  parse,
  extractShareUrl,
  detectPlatform,
  MOBILE_UA,
};
