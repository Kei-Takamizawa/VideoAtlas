# VideoAtlas

[English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh.md) | [हिन्दी](README.hi.md) | **Español** | [العربية](README.ar.md) | [Français](README.fr.md) | [Bahasa Indonesia](README.id.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | [Português](README.pt.md)

**Biblioteca local de vídeos solo para Windows.** Explora y reproduce vídeos, y encuentra escenas según las personas que aparecen. Puedes revisar y corregir las sugerencias faciales dentro de la aplicación.

## Qué puedes hacer

- Explorar y reproducir vídeos de las carpetas que elijas
- Buscar personas candidatas, vídeos relacionados y momentos en que aparecen
- Pausar y reanudar el análisis, y asignar, dividir, combinar o excluir coincidencias manualmente
- Guardar el índice en tu PC sin mover los vídeos originales

Los grupos automáticos son sugerencias basadas en similitud visual y pueden ser incorrectos; revísalos y corrígelos. No se ha medido la precisión del reconocimiento facial. El procesamiento es local; la primera instalación descarga el software y los modelos necesarios.

## Configuración en Windows

Necesitas Windows de 64 bits, Python 3.12 y PowerShell. Desde la carpeta del proyecto, ejecuta:

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

La configuración predeterminada usa la CPU. Para aceleración CUDA de NVIDIA, ejecuta `.\Scripts\setup_windows.cmd -Gpu`. TensorRT requiere una instalación separada; consulta [los detalles de configuración para Windows](docs/WINDOWS_SETUP.md). La implementación para Windows aún no se ha probado en ejecución: no se han ejecutado pruebas, iniciado la aplicación ni analizado vídeos reales.

## Privacidad y modelos

El índice y las imágenes generadas se guardan en los datos locales de aplicaciones de Windows. La aplicación no sube vídeos ni datos faciales. La instalación descarga los modelos fijados FACE01 de rostros japoneses y MediaPipe Face Landmarker, y comprueba sus hashes SHA-256. El modelo FACE01 tiene términos independientes; revísalos antes de usarlo. La [licencia MIT](LICENSE) del proyecto no cubre modelos ni dependencias de terceros.

## Estado del proyecto

La implementación para Windows aún no se ha probado en ejecución: no se han ejecutado pruebas, iniciado la aplicación ni analizado vídeos reales. Esta versión solo para Windows no admite la aplicación ni la configuración anteriores para macOS.
