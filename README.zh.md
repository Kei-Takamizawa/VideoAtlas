# VideoAtlas

[English](README.md) | [日本語](README.ja.md) | **简体中文** | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Bahasa Indonesia](README.id.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | [Português](README.pt.md)

**仅支持 Windows 的本地视频库。** 浏览和播放视频，并根据视频中出现的人物查找片段。您可以在应用中检查并修正人脸匹配建议。

## 功能

- 浏览并播放所选文件夹中的视频
- 按候选人物查找相关视频和人脸出现时间
- 暂停或继续分析，并手动分配、拆分、合并或排除人脸匹配
- 将索引保存在电脑上，不移动原始视频

自动分组依据视觉相似度，只是建议，可能出错，请检查并修正。人脸匹配准确度尚未测量。处理在本地进行；首次设置会下载所需软件和模型。

## Windows 安装

需要 64 位 Windows、Python 3.12 和 PowerShell。在项目文件夹运行：

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

默认使用 CPU。使用 NVIDIA CUDA 加速请运行 `.\Scripts\setup_windows.cmd -Gpu`。TensorRT 需要单独安装，详见[Windows 设置说明](docs/WINDOWS_SETUP.md)。Windows 实现尚未进行运行验证：未运行测试、启动应用或分析真实视频。

## 隐私与模型

索引和生成图像保存在 Windows 本地应用数据文件夹中。应用不会上传视频或人脸数据。安装时会下载固定版本的 FACE01 日本人脸模型和 MediaPipe 人脸关键点模型，并校验 SHA-256。FACE01 模型有独立使用条款，请在使用前查看。项目的 [MIT 许可证](LICENSE)不适用于第三方模型和依赖项。

## 项目状态

Windows 实现尚未进行运行验证：未运行测试、启动应用或分析真实视频。此 Windows 专用版本不支持此前的 macOS 应用和安装方式。
