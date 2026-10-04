# VideoAtlas

[English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Bahasa Indonesia](README.id.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | **Português**

O VideoAtlas é um aplicativo local para Windows que organiza vídeos e ajuda a encontrar cenas pelas pessoas que aparecem neles. Aceita MP4, MOV, AVI, MKV, M4V e WebM. Os vídeos são identificados pelo hash do conteúdo; apenas renomear um arquivo não inicia uma nova análise.

## O que você pode fazer

- Reunir várias amostras de rostos por vídeo e acompanhar as pessoas em cada um
- Comparar com cautela pessoas que aparecem em vídeos diferentes
- Revisar sugestões e escolher “mesma pessoa”, “pessoas diferentes” ou “mais tarde” (teclas S, D e L)
- Editar nomes, mesclar ou dividir grupos, excluir pessoas e escolher uma imagem representativa
- Salvar os pares revisados para avaliar os resultados

A união automática entre vídeos está desativada. Uma pequena coleção real foi avaliada, mas os limiares atuais perdem muitas correspondências. A precisão com máscaras reais ainda não foi verificada. Nomes de arquivos não identificam pessoas. Consulte os resultados e os limites do modo da parte superior do rosto no [estado da implementação](docs/IMPLEMENTATION_STATUS.md).

## Começar no Windows

É necessário Windows de 64 bits, Python 3.12 e PowerShell. Na pasta do projeto, execute:

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

A configuração padrão usa CPU e baixa modelos oficiais verificando seus hashes. A interface está disponível em inglês e japonês. Consulte o [guia de configuração do Windows](docs/WINDOWS_SETUP.md). Os vídeos originais ficam onde estão e o índice é salvo no PC. O aplicativo não envia vídeos nem dados faciais. Modelos e softwares têm termos de uso próprios.
