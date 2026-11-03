/**
 * 激励视频广告封装。
 * adUnitId 为空时返回 null，页面走「无广告直接保存」的降级路径，方便本地调试。
 */
function createRewardedAd(adUnitId) {
  if (!adUnitId || !wx.createRewardedVideoAd) return null;

  const ad = wx.createRewardedVideoAd({ adUnitId });
  ad.onError((err) => {
    console.warn('[rewarded-ad] error', err);
  });
  return ad;
}

/**
 * 展示激励视频，resolve({ isEnded, skipped })
 * isEnded=false 表示用户提前关掉，不应该放行
 */
function showRewardedAd(ad) {
  return new Promise((resolve) => {
    if (!ad) return resolve({ isEnded: true, skipped: true });

    const onClose = (res) => {
      ad.offClose(onClose);
      resolve({ isEnded: !!(res && res.isEnded), skipped: false });
    };
    ad.onClose(onClose);

    ad.show().catch(() => {
      ad.load()
        .then(() => ad.show())
        .catch(() => {
          ad.offClose(onClose);
          resolve({ isEnded: true, skipped: true });
        });
    });
  });
}

module.exports = { createRewardedAd, showRewardedAd };
