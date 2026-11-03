const config = require('../../config');
const { createRewardedAd, showRewardedAd } = require('../../utils/ad');

Page({
  data: {
    inputText: '',
    loading: false,
    result: null,
    activeTab: 'video',
    bannerAdUnitId: config.bannerAdUnitId,
    rewardedAdUnitId: config.rewardedAdUnitId,
  },

  onLoad() {
    this.rewardedAd = createRewardedAd(config.rewardedAdUnitId);
    if (this.rewardedAd) {
      this.rewardedAd.load().catch(() => {});
    }
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
      timeout: 20000,
      success: (res) => {
        const body = res.data || {};
        if (!body.ok) {
          wx.showToast({ title: body.error || '解析失败', icon: 'none' });
          return;
        }
        this.setData({
          result: body.data,
          activeTab: body.data.videoUrl ? 'video' : 'images',
        });
      },
      fail: (err) => {
        console.warn(err);
        wx.showToast({ title: '连不上解析服务，检查 config.baseUrl', icon: 'none' });
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

  ensureAdUnlocked() {
    return showRewardedAd(this.rewardedAd).then((res) => {
      if (res.isEnded) return true;
      wx.showToast({ title: '完整观看广告后才能保存', icon: 'none' });
      return false;
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
            wx.showToast({ title: '下载失败 ' + res.statusCode, icon: 'none' });
            return;
          }
          this.saveToAlbum(res.tempFilePath, 'video');
        },
        fail: (err) => {
          wx.hideLoading();
          console.warn(err);
          wx.showToast({ title: '下载失败，检查域名白名单', icon: 'none' });
        },
      });
    });
  },

  onSaveImages() {
    const images = (this.data.result && this.data.result.images) || [];
    if (!images.length) return;

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
