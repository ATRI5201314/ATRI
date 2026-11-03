// 后端地址：开发者工具里用 127.0.0.1；真机调试改成电脑的局域网 IP（如 http://192.168.1.8:3000）
// 正式上线必须换成已备案的 HTTPS 域名，并在小程序后台配置 request / downloadFile 合法域名
module.exports = {
  baseUrl: 'http://127.0.0.1:3000',

  // 开通「流量主」后，把后台生成的广告位 ID 填进来；留空则只显示占位框
  bannerAdUnitId: '',
  rewardedAdUnitId: '',
};
