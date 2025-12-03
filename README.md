# 🎨 IC-Light: 光照可控的图像生成模型训练

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/yourusername/ic-light.svg?style=social)](https://github.com/yourusername/ic-light)

**IC-Light** 是一个基于扩散模型的光照可控图像生成/编辑项目，允许用户通过文本描述和环境光照图来控制生成图像的光照效果。

> 🔥 **核心特性**: 多条件控制 • 高质量生成 • 光照一致性 • 易于训练

---

## 📖 目录
- [✨ 特性](#特性)
- [🎯 演示效果](#演示效果)
- [🚀 快速开始](#快速开始)
  - [环境安装](#环境安装)
  - [数据准备](#数据准备)
  - [开始训练](#开始训练)
  - [模型推理](#模型推理)
- [📊 训练细节](#训练细节)
- [🏗️ 项目结构](#项目结构)
- [🤝 贡献指南](#贡献指南)
- [📄 许可证](#许可证)
- [🙏 致谢](#致谢)
- [📞 联系方式](#联系方式)

---

## ✨ 特性

### 🎨 **核心功能**
- **光照条件控制**: 通过环境光照图精确控制生成图像的光照
- **文本条件生成**: 支持文本描述引导图像生成
- **图像编辑**: 在指定区域进行光照编辑，保持背景不变
- **高质量输出**: 基于Stable Diffusion，生成512×512高分辨率图像

### ⚡ **技术亮点**
- **多条件融合**: 13通道UNet输入，融合噪声、参考图、背景和掩码
- **光照编码器**: 专用环境光照编码网络
- **高效训练**: 支持梯度检查点、混合精度训练
- **模块化设计**: 易于扩展和自定义

### 🛠️ **易用性**
- 简单的训练接口
- 详细的错误提示和日志
- 预训练模型支持
- 完整的示例和文档

---

## 🎯 演示效果

### **光照编辑示例**
| 输入图像 | 目标光照 | 生成结果 |
|:--------:|:--------:|:--------:|
| ![输入](docs/images/input.jpg) | ![光照](docs/images/light_env.jpg) | ![输出](docs/images/output.jpg) |

### **不同光照条件生成**
