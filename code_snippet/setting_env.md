# 개발 환경 설정

## 가상환경 설정

```bash
# Ubuntu apt 저장소 업데이트
sudo apt-get update

# 실습에 필요한 필수 라이브러리 설치
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev lzma liblzma-dev jq

# Pyenv 설치
curl https://pyenv.run | bash   

# Ubuntu 환경에서 Pyenv를 사용할 수 있도록 환경 설정
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

# 설치 가능한 파이썬 버전 목록 출력
pyenv install --list

# Pyenv를 이용해서 원하는 버전의 파이썬 설치 (Python 3.9.16 설치)
pyenv install 3.9.16

# 설치된 파이썬 버전 확인
pyenv versions 

# PDM 설치 
curl -sSL https://pdm-project.org/install-pdm.py | python3 -

# PDM을 이용해서 프로젝트 환경 초기화
pdm init 

# PDM으로 라이브러리 설치할 때 버전 정보를 미리 알아야 할 때
pip index versions {LIBRARY_NAME}

# PDM을 이용해서 pandas 2.1.4 버전 이상의 버전을 추가할 때
pdm add “pandas>=2.1.4”

# PDM을 이용해서 black 최신 버전을 설치하고, 개발 환경에서만 사용하도록 할 때
pdm add -dG dev black 

# PDM 가상환경으로 접근하려면
source .venv/bin/activate
```

## 설치 라이브러리 목록

```yaml
dependencies = [
    "bentoml>=1.2.6",
    "scikit-learn>=1.3.2",
    "pandas>=2.1.4",
    "numpy>=1.26.4",
    "mlflow>=2.11.1",
    "category-encoders>=2.6.3",
    "pydantic>=2.6.3",
    "deepchecks>=0.18.1",
    "joblib>=1.3.2",
    "rich>=13.7.1",
    "seaborn>=0.13.2",
]

[project.optional-dependencies]
dev = [
    "matplotlib>=3.8.3",
    "pre-commit>=3.6.2",
    "black>=24.2.0",
    "isort>=5.13.2",
    "mypy>=1.9.0",
    "jupyter>=1.0.0",
]
```

## 프리커밋 설정

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: check-ast
      - id: check-case-conflict
      - id: check-docstring-first
      - id: check-executables-have-shebangs
      - id: check-json
      - id: check-yaml
      - id: debug-statements
      - id: detect-private-key
      - id: end-of-file-fixer
      - id: fix-byte-order-marker
      - id: trailing-whitespace
      - id: mixed-line-ending
  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black", "-l80"]
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        args: [-l 80]
```