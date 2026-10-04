# VideoAtlas

[English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh.md) | [हिन्दी](README.hi.md) | **Español** | [العربية](README.ar.md) | [Français](README.fr.md) | [Bahasa Indonesia](README.id.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | [Português](README.pt.md)

VideoAtlas es una aplicación local para Windows que organiza vídeos y permite encontrar escenas según las personas que aparecen. Admite MP4, MOV, AVI, MKV, M4V y WebM. Identifica cada vídeo por el hash de su contenido, así que cambiar solo su nombre no provoca otro análisis.

## Qué puedes hacer

- Recopilar varias muestras de rostros por vídeo y seguir a las personas dentro de cada uno
- Comparar con cautela a quienes aparecen en vídeos distintos
- Revisar sugerencias y elegir «misma persona», «personas distintas» o «más tarde» (teclas S, D y L)
- Editar nombres, combinar o separar grupos, excluir personas y elegir una imagen representativa
- Guardar las decisiones revisadas para evaluar las coincidencias

La unión automática entre videos está desactivada. Se evaluó una pequeña colección real, pero los umbrales actuales omiten muchas coincidencias. La precisión con mascarillas reales sigue sin verificarse. Los nombres de archivo no se usan para identificar personas. Consulte los resultados y los límites del modo de cara superior en el [estado de implementación](docs/IMPLEMENTATION_STATUS.md).

## Primeros pasos en Windows

Necesitas Windows de 64 bits, Python 3.12 y PowerShell. Desde la carpeta del proyecto, ejecuta:

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

La configuración predeterminada usa CPU y descarga modelos oficiales verificando sus sumas de comprobación. La interfaz está en inglés y japonés. Consulta la [guía de configuración de Windows](docs/WINDOWS_SETUP.md). Los vídeos originales no se mueven y el índice se guarda en el PC. La aplicación no sube vídeos ni datos faciales. Modelos y software tienen sus propias condiciones de uso.
