# 第三方组件、数据与许可证

## 代码

本仓库的代码（Python、JavaScript、HTML/CSS、构建脚本）全部由本项目编写，**没有拷贝或内嵌第三方源代码**。
网页端不使用任何第三方 JavaScript 库：神经网络推理（`common/nn.js`）、网络结构图（`common/archviz.js`）、手写板、图表都是自己写的纯 JavaScript，
不需要联网，也不从 CDN 加载任何东西。界面使用设备自带的系统字体，仓库里不含字体文件。
网络结构图的画法参考了 Brendan Bycroft 的 LLM Visualization（https://bbycroft.net/llm）的思路，没有使用它的代码或素材。

## 数据：MNIST 手写数字

| 项目 | 内容 |
|---|---|
| 来源 | MNIST database of handwritten digits |
| 版权 | © Yann LeCun、Corinna Cortes（MNIST 本身是 NIST 数据集的派生作品） |
| 许可证 | Creative Commons Attribution-Share Alike 3.0（CC BY-SA 3.0） |
| 许可证依据 | TensorFlow / Keras 的 MNIST 数据集文档注明的条款；作者未能直接访问 MNIST 官方主页核对，如有出入以官方为准 |
| 本仓库如何使用 | 训练脚本运行时通过 torchvision 自动下载，原始数据**不随仓库分发** |
| 随仓库再分发的部分 | 网页里内嵌的部分 MNIST 测试集图片，以及由它们派生的数据（还原图、2 维编码坐标等），位于 `play/assets/gen/` 和 `results/` |

上表最后一行的数据**按 CC BY-SA 3.0 再分发**（署名：MNIST © Yann LeCun、Corinna Cortes；相同方式共享），
不适用本仓库文档的 CC BY 4.0。网页的页脚注明了出处。模型权重由本项目在 MNIST 上训练得到。

## Python 依赖（requirements.txt，不随仓库分发，由使用者自行安装）

版本为作者测试时的版本；许可证取自各包发布的元数据。

| 包 | 测试版本 | 许可证 | 用途 |
|---|---|---|---|
| torch | 2.11.0 | BSD-3-Clause | 训练 |
| torchvision | 0.26.0 | BSD-3-Clause | 下载与读取 MNIST |
| numpy | 2.5.2 | BSD-3-Clause（及其内含组件的 0BSD / MIT / Zlib / CC0-1.0） | 数值计算 |
| scikit-learn | 1.9.1 | BSD-3-Clause | 最近邻、逻辑回归探针 |
| mini-racer | 0.14.1 | ISC | 测试：用 V8 运行网页端推理，与 PyTorch 比对 |
| websocket-client | 1.9.2 | Apache-2.0 | 浏览器验收（DevTools 协议） |
| pytest | 9.1.1 | MIT | 测试 |

浏览器验收（`webapp/browser_check.py`）只驱动本机已安装的 Chrome / Edge，不下载任何浏览器。
