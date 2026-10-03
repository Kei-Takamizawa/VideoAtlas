# VideoAtlas

[English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Bahasa Indonesia](README.id.md) | **한국어** | [Русский](README.ru.md) | [Português](README.pt.md)

**Windows 전용 로컬 동영상 라이브러리입니다.** 동영상을 둘러보고 재생한 뒤, 영상에 나온 사람을 기준으로 장면을 찾을 수 있습니다. 얼굴 매칭 제안을 앱에서 확인하고 수정할 수 있습니다.

## 할 수 있는 일

- 선택한 폴더의 동영상을 찾아보고 재생
- 후보 인물, 관련 동영상, 얼굴이 나타난 시점 검색
- 분석을 일시 중지하고 다시 시작하며 얼굴 매칭을 직접 지정·분할·병합·제외
- 원본 동영상을 옮기지 않고 PC에 라이브러리 색인을 저장

자동 그룹은 시각적 유사성을 바탕으로 한 제안이며 틀릴 수 있으므로 확인하고 수정하세요. 얼굴 매칭 정확도는 측정되지 않았습니다. 처리는 로컬에서 이루어지며 최초 설정 시 필요한 소프트웨어와 모델을 다운로드합니다.

## Windows 설정

64비트 Windows, Python 3.12, PowerShell이 필요합니다. 프로젝트 폴더에서 실행하세요.

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

기본 설정은 CPU를 사용합니다. NVIDIA CUDA 가속을 사용하려면 `.\Scripts\setup_windows.cmd -Gpu`를 실행하세요. TensorRT는 별도로 설치해야 합니다. [Windows 설정 상세](docs/WINDOWS_SETUP.md)를 참고하세요. Windows 구현은 실행 테스트를 거치지 않았습니다. 테스트, 앱 실행, 실제 동영상 분석을 하지 않았습니다.

## 개인정보와 모델

색인과 생성된 이미지는 Windows 로컬 앱 데이터에 저장됩니다. 앱은 동영상이나 얼굴 데이터를 업로드하지 않습니다. 설정 과정에서 고정된 FACE01 일본인 얼굴 모델과 MediaPipe 얼굴 랜드마크 모델을 다운로드하고 SHA-256을 확인합니다. FACE01 모델에는 별도의 이용 약관이 있으니 사용 전에 확인하세요. 프로젝트의 [MIT 라이선스](LICENSE)는 제3자 모델이나 의존성에 적용되지 않습니다.

## 프로젝트 상태

Windows 구현은 실행 테스트를 거치지 않았습니다. 테스트, 앱 실행, 실제 동영상 분석을 하지 않았습니다. 이 Windows 전용 릴리스는 이전 macOS 앱과 설치 방식을 지원하지 않습니다.
