const config = require('../../config');
const { createRewardedAd, showRewardedAd } = require('../../utils/ad');

const UNLOCK_KEY = 'adUnlockUntil';
const UNLOCK_MS = 24 * 60 * 60 * 1000; // 观看一次广告解锁 24 小时

Page({
  data: {
    inputText: '',
    loading: false,
    result: null,
    activeTab: 'video',
    bannerAdUnitId: config.bannerAdUnitId,
    rewardedAdUnitId: config.rewardedAdUnitId,
    adFree: false,
    adFreeText: '',
  },

  onLoad() {
    this.rewardedAd = createRewardedAd(config.rewardedAdUnitId);
    if (this.rewardedAd) {
      this.rewardedAd.load().catch(() => {});
    }

    this.refreshAdFree();
    // 每分钟刷新剩余时间，页面销毁时清理
    this.adFreeTimer = setInterval(() => this.refreshAdFree(), 60000);
  },

  onShow() {
    this.refreshAdFree();
  },

  onUnload() {
    if (this.adFreeTimer) clearInterval(this.adFreeTimer);
  },

  refreshAdFree() {
    const until = Number(wx.getStorageSync(UNLOCK_KEY) || 0);
    const left = until - Date.now();

    if (left <= 0) {
      if (this.data.adFree) this.setData({ adFree: false, adFreeText: '' });
      return;
    }

    const hours = Math.floor(left / 3600000);
    const minutes = Math.floor((left % 3600000) / 60000);
    const text = hours > 0 ? hours + ' 小时 ' + minutes + ' 分' : minutes + ' 分钟';
    this.setData({ adFree: true, adFreeText: text });
  },

  onInput(e) {
    this.setData({ inputText: e.detail.value });
  },

  onPaste() {
    wx.getClipboardData({
      success: (res) => this.setData({ inputText: res.data || '' }),
      fail: () => wx.showToast({ title: '读取剪贴板失败', icon: 'none' }),
    });
  },

  onClear() {
    this.setData({ inputText: '', result: null });
  },

  onDemo() {
    this.setData({ inputText: 'demo', result: null }, () => this.onStart());
  },

  onStart() {
    if (this.data.loading) return;
    const text = this.data.inputText.trim();
    if (!text) {
      wx.showToast({ title: '请先粘贴分享链接', icon: 'none' });
      return;
    }

    this.setData({ loading: true });
    wx.request({
      url: config.baseUrl + '/api/parse',
      method: 'POST',
      header: { 'content-type': 'application/json' },
      data: { text },
      timeout: 45000,
      success: (res) => {
        const body = res.data || {};
        if (!body.ok) {
          wx.showToast({ title: body.error || '获取失败，请稍后重试', icon: 'none' });
          return;
        }
        this.setData({
          result: body.data,
          activeTab: body.data.videoUrl ? 'video' : 'images',
        });
      },
      fail: (err) => {
        console.warn(err);
        wx.showToast({ title: '网络异常，请稍后重试', icon: 'none' });
      },
      complete: () => this.setData({ loading: false }),
    });
  },

  onTab(e) {
    this.setData({ activeTab: e.currentTarget.dataset.tab });
  },

  onPreview(e) {
    const current = e.currentTarget.dataset.url;
    wx.previewImage({
      current,
      urls: (this.data.result && this.data.result.images) || [],
    });
  },

  onCopy() {
    const result = this.data.result;
    if (!result) return;
    wx.setClipboardData({ data: result.videoUrl || result.title || '' });
  },

  onCopyTitle() {
    const result = this.data.result;
    if (!result) return;
    wx.setClipboardData({ data: result.title || '' });
  },

  /**
   * 免广告判断：24 小时内直接放行；
   * 否则播一段激励视频，完整看完才解锁并放行。
   */
  ensureAdUnlocked() {
    if (this.data.adFree) return Promise.resolve(true);
    if (!this.rewardedAd) return Promise.resolve(true);

    return showRewardedAd(this.rewardedAd).then((res) => {
      if (!res.isEnded) {
        wx.showToast({ title: '看完整段广告才能解锁', icon: 'none' });
        return false;
      }
      wx.setStorageSync(UNLOCK_KEY, Date.now() + UNLOCK_MS);
      this.refreshAdFree();
      wx.showToast({ title: '已解锁 24 小时免广告', icon: 'success' });
      return true;
    });
  },

  onSaveVideo() {
    const result = this.data.result;
    if (!result || !result.videoUrl) return;

    this.ensureAdUnlocked().then((unlocked) => {
      if (!unlocked) return;

      wx.showLoading({ title: '下载中', mask: true });
      wx.downloadFile({
        url: result.downloadUrl || result.videoUrl,
        success: (res) => {
          wx.hideLoading();
          if (res.statusCode !== 200) {
            wx.showToast({ title: '下载失败，请重试', icon: 'none' });
            return;
          }
          this.saveToAlbum(res.tempFilePath, 'video');
        },
        fail: (err) => {
          wx.hideLoading();
          console.warn(err);
          wx.showToast({ title: '下载失败，请重试', icon: 'none' });
        },
      });
    });
  },

  onSaveImages() {
    const images = (this.data.result && this.data.result.images) || [];
    if (!images.length) return;

    this.ensureAdUnlocked().then((unlocked) => {
      if (!unlocked) return;

      let saved = 0;
      wx.showLoading({ title: '0/' + images.length, mask: true });

      const step = (index) => {
        if (index >= images.length) {
          wx.hideLoading();
          wx.showToast({ title: '已保存 ' + saved + ' 张', icon: 'none' });
          return;
        }
        wx.downloadFile({
          url: images[index],
          success: (res) => {
            wx.saveImageToPhotosAlbum({
              filePath: res.tempFilePath,
              success: () => {
                saved += 1;
              },
              complete: () => {
                wx.showLoading({ title: index + 1 + '/' + images.length, mask: true });
                step(index + 1);
              },
            });
          },
          fail: () => step(index + 1),
        });
      };

      step(0);
    });
  },

  saveToAlbum(filePath, type) {
    const save = type === 'video' ? wx.saveVideoToPhotosAlbum : wx.saveImageToPhotosAlbum;
    save({
      filePath,
      success: () => wx.showToast({ title: '已保存到相册', icon: 'success' }),
      fail: (err) => {
        console.warn(err);
        if (/auth|authorize/i.test(err.errMsg || '')) {
          wx.showModal({
            title: '需要相册权限',
            content: '请在设置里打开「保存到相册」后重试',
            confirmText: '去设置',
            success: (res) => {
              if (res.confirm) wx.openSetting();
            },
          });
          return;
        }
        wx.showToast({ title: '保存失败', icon: 'none' });
      },
    });
  },
});
