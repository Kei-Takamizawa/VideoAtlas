# VideoAtlas

[English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | **Français** | [Bahasa Indonesia](README.id.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | [Português](README.pt.md)

**Bibliothèque vidéo locale réservée à Windows.** Parcourez et lisez vos vidéos, puis retrouvez des scènes à partir des personnes qui y apparaissent. Vous pouvez vérifier et corriger les suggestions de visages dans l’application.

## Fonctions

- Parcourir et lire les vidéos des dossiers choisis
- Rechercher des personnes candidates, des vidéos associées et les moments où des visages apparaissent
- Mettre l’analyse en pause puis la reprendre, et attribuer, séparer, fusionner ou exclure les correspondances manuellement
- Garder l’index sur votre PC sans déplacer les vidéos originales

Les groupes automatiques sont des suggestions fondées sur la similarité visuelle et peuvent être erronés. Vérifiez-les et corrigez-les. La précision de reconnaissance n’a pas été mesurée. Le traitement est local ; la première installation télécharge les logiciels et modèles nécessaires.

## Installation Windows

Il faut Windows 64 bits, Python 3.12 et PowerShell. Dans le dossier du projet, exécutez :

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

La configuration standard utilise le CPU. Pour l’accélération CUDA NVIDIA, lancez `.\Scripts\setup_windows.cmd -Gpu`. TensorRT s’installe séparément ; consultez les [détails d’installation Windows](docs/WINDOWS_SETUP.md). L’implémentation Windows n’a pas été testée à l’exécution : aucun test, lancement de l’application ou analyse de vidéos réelles n’a été effectué.

## Confidentialité et modèles

L’index et les images créées sont enregistrés dans les données locales d’applications Windows. L’application n’envoie pas de vidéos ni de données faciales. L’installation télécharge les modèles épinglés FACE01 pour visages japonais et MediaPipe Face Landmarker, puis vérifie leurs empreintes SHA-256. Le modèle FACE01 a ses propres conditions ; consultez-les avant utilisation. La [licence MIT](LICENSE) du projet ne couvre pas les modèles et dépendances tiers.

## État du projet

L’implémentation Windows n’a pas été testée à l’exécution : aucun test, lancement de l’application ou analyse de vidéos réelles n’a été effectué. Cette version réservée à Windows ne prend pas en charge l’ancienne application et l’installation macOS.
