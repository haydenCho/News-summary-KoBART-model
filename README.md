# News-summary-KoBART-model
**2023 DSWU 소프트웨어 전공 강의 [딥러닝응용] 기말 과제 레파지토리**

<br/>

## 프로젝트 개요
### 1. 프로젝트 주제 및 목표
- **주제: KoBART 모델 사용을 사용한 뉴스 요약 모델 학습**
- 바쁜 일상 속에서 모든 기사를 자세하게 읽는 것이 어려운 사용자를 위해 뉴스 데이터에 특화된 문서 요약 서비스를 개발한다.
- 자연어처리(NLP) 분야에서의 Seq-to-Seq 모델 활용 능력을 키우고, KoBART를 기반으로 한국어 요약 모델을 효과적으로 미세 조정(Fine-tuning)한다.

<br/>

### 2. 개발환경 및 기간
#### 개발 환경
- GPU: Nvidia RTX 3060
- CUDA 버전: 11.8
- 언어: Python
- 도구: Visual Studio Code, Google Colab, Hugging Face Transformers, PyTorch, PyTorch Lightning, Streamlit

<br/>

#### 개발 기간
2023.11.19 ~ 2023.12.19 (약 1개월)

<br/>

### 3. 데이터셋
- 데이콘(DACON) 과 AIHub에서 제공하는 문서 요약 데이터셋
- [AIHub-문서요약 텍스트](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97)
    - 기사별로 이뤄진 json 파일 형태의 데이터셋을 제공

<br/>

#### 데이터 선별 과정
1. 데이터셋(JSON)에서 기사 본문과 요약문 추출
2. Pandas를 사용하여 .tsv 형태로 변환 → 메모리 효율적
3. 병합한 데이터셋 중에서 원활한 학습을 위해 대략 13만 개 데이터를 학습에 사용

<br/>

### 4. 기대되는 결과 및 기여
- 뉴스 자동 요약 모델을 통해 긴 뉴스 기사의 핵심 정보를 빠르게 파악 가능
- 한글 문서 요약에 특화된 KoBART 모델을 Fine-tuning한 실습 경험 제공
- 교육 및 연구 목적에서 KoBART 기반 요약 모델 실험 및 확장 가능성 제공

<br/>

## 모델
### BART (Bidirectional and Auto-Regressive Transformer)
- BART는 BERT와 GPT의 장점을 결합한 형태로, 인코더-디코더 구조를 가지는 표준 Transformer 아키텍처를 따름
- **Encoder**는 BERT처럼 입력 문장을 **양방향**으로 이해하고, **Decoder**는 GPT처럼 **왼쪽에서 오른쪽**으로 문장을 생성
- 이로 인해 BART는 단순 언어 모델이 아니라, 문서 요약, 기계 번역, 문장 생성 등 다양한 자연어 생성 태스크에서 강력한 성능을 보임

<br/>

#### 기존 BERT의 MLM과 BART의 사전 학습 차이
- BERT의 Masked Language Modeling (MLM) <br/>
  → 문장에서 일부 단어를 [MASK]로 가리고 이를 예측하게 하는 방식
- BART의 사전 학습 방식은 더 다양하고 강력 <br/>
  → 입력 문장을 완전히 섞거나, 문장의 일부분을 삭제하거나, 순서를 바꾸는 등 다양한 노이즈를 추가한 후 <br/>
  → 원래의 문장을 복원하는 방식으로 학습 <br/>
  → 이 과정을 통해 BART는 문맥, 순서, 의미를 더 깊이 이해하게 되고, 요약 태스크에 최적화 <br/>
✔️ 즉, BART는 입력 문장의 복원 능력을 바탕으로, 요약이나 번역, 대화 생성 등에 뛰어난 성능을 보임

<br/>

### KoBART
- KoBART는 SKT에서 공개한 한국어 특화 BART 모델
- 한국어 말뭉치로 사전학습된 BART 구조이기 때문에 추가적인 번역이나 언어 변환 없이도 한국어 데이터셋에 바로 적용 가능하며, 기계 번역, 문서 요약 등 자연어 생성(NLG) 태스크에서 좋은 성능을 보임

<br/>

#### 프로젝트에서의 적용 흐름
- HuggingFace에서 사전 학습된 facebook/bart-base 및 gogamza/kobart-summarization 모델 로드
- 뉴스 기사 요약 데이터셋으로 미세 조정(Fine-tuning) 수행
- 테스트셋으로 성능 평가 및 비교 (ROUGE Score 기준)

<br/>

### 학습 파라미터
- 에폭
- 배치
- 학습률

<br/>

## 모델 평가
### 정확도 평가 방식
- **ROUGE Score 사용**
    - ROUGE-1: 단어 단위 겹침
    - ROUGE-2: 문맥/순서 반영된 2-gram 기반 정확도
- **한국어의 특성 보완**
    - 기존 ROUGE는 조사/어순 무시 → 정확도 신뢰 어려움<br/>
    → *mecab 형태소 분석기 적용*으로 보다 정확한 평가

<br/>

### 성능 비교 및 결과
- 기존 KoBART 대비 정확도 향상
    - 특히 ROUGE-2에서 약 40% 향상 → 문맥 기반 요약 품질 개선
    
- 시각화
    - 학습 전후 ROUGE Score 그래프 비교
    <img src="https://github.com/user-attachments/assets/9983457f-bbca-45af-8f60-fb0a86b37e60" width="500px">

<br/>

### 코드 설명
- train.py: 학습 코드
- requirements.txt: 프로젝트에 필요한 라이브러리와 버전 리스트 (모델 로드나 학습시 사용)
- mecabGogamza.ipynb: 기존 모델 정확도
- plusMecab.ipynb: 최종 모델 정확도
- infer.py: 모델 사용 코드

<br/>

## Reference
- [조건희, 성창민, 김은총, 정윤경.(2022).인과 추론 문제 해결에 적합한 자연어 처리 모델 BART 분석.한국정보과학회 학술발표논문집,(),359-361.](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE11113334)
- [wikidocs](https://wikidocs.net/book/8056)
- [참고 블로그](https://velog.io/@noop00/%ED%95%9C%EA%B5%AD%EC%96%B4-%ED%85%8D%EC%8A%A4%ED%8A%B8-%EC%9A%94%EC%95%BD-AI-%EB%AA%A8%EB%8D%B8%EB%A1%9C-%EA%B0%80%EC%9D%B4%EB%93%9C-%EC%8A%A4%ED%81%AC%EB%A6%BD%ED%8A%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0)
