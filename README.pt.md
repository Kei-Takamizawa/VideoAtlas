# VideoAtlas

[English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Bahasa Indonesia](README.id.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | **Português**

**Biblioteca de vídeos local somente para Windows.** Navegue e reproduza vídeos e encontre cenas pelas pessoas que aparecem neles. Você pode revisar e corrigir as sugestões de rostos no aplicativo.

## O que você pode fazer

- Navegar e reproduzir vídeos das pastas escolhidas
- Encontrar pessoas candidatas, vídeos relacionados e horários em que rostos aparecem
- Pausar e retomar a análise e atribuir, dividir, combinar ou excluir correspondências manualmente
- Manter o índice no PC sem mover os vídeos originais

Os grupos automáticos são sugestões baseadas em semelhança visual e podem estar errados. Revise e corrija os resultados. A precisão da correspondência facial não foi medida. O processamento é local; a primeira configuração baixa os programas e modelos necessários.

## Configuração no Windows

É necessário Windows de 64 bits, Python 3.12 e PowerShell. Na pasta do projeto, execute:

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

A configuração padrão usa CPU. Para aceleração NVIDIA CUDA, execute `.\Scripts\setup_windows.cmd -Gpu`. O TensorRT precisa ser instalado separadamente; consulte os [detalhes de configuração para Windows](docs/WINDOWS_SETUP.md). A implementação para Windows não foi testada em execução: não foram executados testes, iniciado o aplicativo ou analisados vídeos reais.

## Privacidade e modelos

O índice e as imagens geradas ficam nos dados locais de aplicativos do Windows. O app não envia vídeos nem dados faciais. A configuração baixa os modelos fixados FACE01 para rostos japoneses e MediaPipe Face Landmarker e verifica seus hashes SHA-256. O modelo FACE01 tem termos próprios; leia-os antes de usar. A [licença MIT](LICENSE) do projeto não cobre modelos nem dependências de terceiros.

## Estado do projeto

A implementação para Windows não foi testada em execução: não foram executados testes, iniciado o aplicativo ou analisados vídeos reais. Esta versão exclusiva para Windows não oferece suporte ao aplicativo e à configuração anteriores para macOS.
