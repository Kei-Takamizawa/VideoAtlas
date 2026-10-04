# VideoAtlas

[English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | **Français** | [Bahasa Indonesia](README.id.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | [Português](README.pt.md)

VideoAtlas est une application locale pour Windows qui organise vos vidéos et aide à retrouver des scènes selon les personnes qui y apparaissent. Elle prend en charge MP4, MOV, AVI, MKV, M4V et WebM. L’identification repose sur l’empreinte du contenu : un simple changement de nom ne relance pas l’analyse.

## Fonctions

- Recueillir plusieurs échantillons de visages par vidéo et suivre les personnes dans chacune
- Comparer prudemment les personnes présentes dans différentes vidéos
- Examiner les suggestions et choisir « même personne », « personnes différentes » ou « plus tard » (touches S, D, L)
- Modifier les noms, fusionner ou séparer des groupes, exclure des personnes et choisir une image représentative
- Enregistrer les paires examinées pour évaluer les correspondances

La fusion automatique entre vidéos est désactivée. Une petite collection réelle a été évaluée, mais les seuils actuels manquent de nombreuses correspondances. La précision avec de vrais masques reste inconnue. Les noms de fichiers ne servent pas à identifier les personnes. Consultez les résultats et les limites du mode visage supérieur dans l’[état de réalisation](docs/IMPLEMENTATION_STATUS.md).

## Démarrer sous Windows

Il faut Windows 64 bits, Python 3.12 et PowerShell. Dans le dossier du projet, exécutez :

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

La configuration standard utilise le CPU et télécharge les modèles officiels en vérifiant leur somme de contrôle. L’interface est en anglais et en japonais. Consultez le [guide de configuration Windows](docs/WINDOWS_SETUP.md). Les vidéos originales restent à leur emplacement et l’index est conservé sur le PC. L’application n’envoie ni vidéos ni données faciales. Modèles et logiciels ont leurs propres conditions d’utilisation.
