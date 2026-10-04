# VideoAtlas

[English](README.md) | [日本語](README.ja.md) | **简体中文** | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Bahasa Indonesia](README.id.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | [Português](README.pt.md)

VideoAtlas 是一款适用于 Windows 的本地应用，可整理视频并按出镜人物查找片段。支持 MP4、MOV、AVI、MKV、M4V 和 WebM。应用使用视频内容哈希；仅重命名文件不会触发重新分析。

## 功能

- 为每段视频收集多个面部样本，并追踪其中的人物
- 谨慎匹配出现在不同视频中的人物
- 在审核队列中选择“同一人”“不同人”或“稍后处理”（快捷键 S、D、L）
- 手动编辑人物名称、合并、拆分、排除及代表图
- 记录已审核的人物对，以评估匹配效果

视频间自动合并默认关闭。已评估少量提供的真实视频，但当前阈值仍会漏掉许多同一人物匹配，真实口罩条件下的准确率尚未确认。不会根据文件名识别人。测试结果和上半脸模式的限制见[实现状态](docs/IMPLEMENTATION_STATUS.md)。

## Windows 快速开始

需要 64 位 Windows、Python 3.12 和 PowerShell。在项目文件夹运行：

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

默认使用 CPU。安装程序会下载并校验官方模型。应用提供英语和日语界面。请参阅 [Windows 设置指南](docs/WINDOWS_SETUP.md)。

原视频不会移动，索引保存在电脑上。应用不会上传视频或人脸数据。模型和软件各自受其许可条款约束。
