# VideoAtlas

[English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Bahasa Indonesia](README.id.md) | **한국어** | [Русский](README.ru.md) | [Português](README.pt.md)

VideoAtlas는 동영상을 정리하고 등장인물을 기준으로 장면을 찾는 Windows용 로컬 앱입니다. MP4, MOV, AVI, MKV, M4V, WebM을 지원합니다. 동영상 내용의 해시로 파일을 식별하므로 이름만 바꿔도 다시 분석하지 않습니다.

## 할 수 있는 일

- 동영상마다 여러 얼굴 샘플을 모으고 그 안의 인물을 추적
- 서로 다른 동영상에 등장하는 인물을 신중하게 비교
- 검토 대기열에서 ‘같은 사람’, ‘다른 사람’, ‘나중에’를 선택 (S, D, L 키)
- 이름 변경, 그룹 병합·분할, 인물 제외, 대표 이미지 선택
- 검토한 인물 쌍을 저장해 매칭 결과 평가

영상 간 자동 병합은 기본적으로 꺼져 있습니다. 제공된 소규모 실제 영상으로 평가했지만 현재 기준은 같은 사람의 많은 일치를 놓칩니다. 실제 마스크 착용 시 정확도는 아직 확인되지 않았습니다. 파일명으로 사람을 판별하지 않습니다. 결과와 얼굴 윗부분 모드의 한계는 [구현 상태](docs/IMPLEMENTATION_STATUS.md)를 참고하세요.

## Windows에서 시작하기

64비트 Windows, Python 3.12, PowerShell이 필요합니다. 프로젝트 폴더에서 실행하세요.

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

기본 설정은 CPU를 사용하며 체크섬을 확인한 공식 모델을 다운로드합니다. 앱 인터페이스는 영어와 일본어를 제공합니다. [Windows 설정 안내](docs/WINDOWS_SETUP.md)를 참고하세요. 원본 동영상은 이동하지 않고 색인은 PC에 저장됩니다. 앱은 동영상이나 얼굴 데이터를 업로드하지 않습니다. 모델과 소프트웨어에는 별도의 사용 조건이 있습니다.
