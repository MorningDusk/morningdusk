---
{"publish":true,"aliases":"SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation","created":"2025-06-22","modified":"2025-07-31T16:25:17.113+09:00","tags":["computer_vision","ai","article","6d"],"cssclasses":""}
---

# 1. Introduction
## 1.1. 연구 배경과 문제 정의
기존의 **인스턴스 수준 6D 자세 추정**<sub>Instance-level 6D pose estimation</sub>은 미리 분류된 훈련 이미지들이 필요했기 때문에 모델이 특정 객체에만 특화될 수밖에 없었다. 본 적 없는 객체를 탐지하기 위해 **카테고리 수준 6D 자세 추정**<sub>category-level 6D pose estimation</sub>으로 발전했지만, 여전히 특정 카테고리에만 특화되어 있어 한계가 존재했다. 이러한 한계를 극복하기 위해 완전히 새로운 객체와 그 자세를 탐지하는 **Zero-shot 6D 객체 자세 추정**<sub>Zero-shot 6D object pose estimation</sub>을 연구하게 되었다.
![[Image/Pasted image 20250623215340.png|600]]
## 1.2. SAM의 활용과 연구 동기
최근 **Segment Anything Model**<sub>SAM</sub>은 뛰어난 Zero-shot 분할 성능을 보여준다. 점, 박스, 텍스트 또는 마스크를 입력하면 훈련되지 않은 새로운 객체도 정확히 인식하고 분할할 수 있다. 더 정확한 자세 추정을 위해 SAM의 강력한 능력과 zero-shot 6D 기술을 결합하여 SAM-6D를 개발했다.
SAM-6D는 **인스턴스 분할 모델**<sub>Instance Segmentation Model; ISM</sub>과 **자세 추정 모델**<sub>Pose Estimation Model; PEM</sub>을 함께 사용하는 2단계 시스템이다.
# 2. Related Work
## 2.1. Segment Anything
**Segment Anything**<sub>SA</sub>는 다양한 프롬프트를 사용하는 분할 기술로, 점, 박스, 텍스트, 마스크 등을 입력받아 해당하는 마스크를 생성하는 데 집중한다. 이를 기반으로한 SAM은 이미지 인코더, 프롬프트 인코더, 마스크 디코더 3개의 주요 구성요소로 이루어져 있다. 
SAM은 의료 이미지, 위장된 객체, 투명한 객체 등 다양한 어려운 상황에서도 뛰어난 제로샷 성능을 보여주었다. 효율성을 위해 **FastSAM**은 SAM의 무거운 시각 트랜스포머<sub>visual transformer</sub> 대신 가벼운 컨볼루션 네트워크를 사용한다.
## 2.2. 새로운 객체의 자세 추정
### 이미지 매칭 기반 방법
이 그룹들의 방법들은 객체 후보들을 미리 렌더링된 템플릿과 비교하여 가장 잘 매칭되는 자세를 찾는 방식을 사용한다. Gen6D, OVE6D, GigaPose는 **이미지 매칭**<sub>image matching</sub>을 통해 먼저 시점 회전을 선택한 다음 평면 내 회전을 추정하여 최종 자세를 계산한다.
### 특징 매칭 기반 방법
이 그룹의 방법들은 후보들의 2D 픽셀이나 3D 점들을 특징 공간에서 객체 표면과 정렬하여 **대응 관계**<sub>correspondance</sub>를 구축한 후 객체 자세를 계산한다. OnePose는 2D-3D 대응을 위해 픽셀 특징을 점 특징과 매칭하고, ZeroPose는 기하학적 구조를 통해 3D-3D 매칭을 실형한다.
# 3. Methodology of SAM-6D
## 3.1. 인스턴스 분할 모델
SAM-6D는 ISM을 사용하여 새로운 객체 $O$의 인스턴스들을 분할한다. 복잡한 장면의 RGB 이미지 $I$가 주여졌을 때, SAM의 뛰어난 zero-shot 능력을 활용하여 모든 가능한 **객체 후보**<sub>object candidate</sub> $M$을 생성한다.
### Segment Anything Model 기본 원리
RGB 이미지 I가 주어졌을 때, SAM은 점, 박스, 텍스트, 마스크 등의 다양한 **프롬프트**<sub>prompt</sub> P<sub>r</sub>을 받아 분할 결과를 생성한다. SAM은 이미지 인코더 $\Phi_{Image}$, 프롬프트 인코더  $\Phi_{Prompt}$, 마스크 디코더Ψ$\Psi_{Mask}$ 세 개의 모듈로 구성된다:
$$
M,C=\Psi_{Mask}(\Phi_{Image}(I),\Phi_{Prompt}(P_r))
$$
여기서 $M$과 $C$는 각각 예측된 마스크 후보들과 **신뢰도 점수**<sub>confidence score</sub>를 의미한다.
### 객체 매칭 점수
후보 $M$이 생성되면, 각 후보 $m \in M$에 대해 지정된 객체 O와의 매칭 정도를 평가하는 **객체 매칭 점수**<sub>object matching score</sub> s<sub>m</sub>을 계산한다. 이 점수는 **의미적 매칭**<sub>semantic matching</sub>, **외관 매칭**<sub>appearance matching</sub>, **기하학적 매칭**<sub>geometric matching</sub>이라는 세 가지 요소로 구성된다.
먼저 객체 $O$에 대해 $SE(3)$ 공간에서 NT개의 자세를 샘플링하여 **템플릿**<sub>template</sub> $\{T_k\}^{N_T}_{k=1}$을 렌더링하고, 이를 DINOv2의 ViT 백본에 입력하여 각 템플릿의 **클래스 임베딩**<sub>class embedding</sub> $f^{cls}_{T_k}$와 **패치 임베딩**<sub>patching embedding</sub> $\{f^{patch}_{T_k,i}\}^{N^{patch}_{T_k}}_{i=1}$을 추출한다. 각 후보 $m$에 대해서도 동일한 과정을 거쳐 클래스 임베딩 $f^{cls}_{I_m}$과 패치 임베딩 $\{f^{patch}_{Im,j}\}^{N^{patch}_{I_m}}_{j=1}$을 얻는다.
#### 의미적 매칭 점수
의미적 매칭 점수는 클래스 임베딩 간의 **코사인 유사도**<sub>cosine similarity</sub>를 통해 계산한다:
$$
s_{sem} = \frac{1}{K} \sum_{i=1}^{K} \text{top-K}\left\{\frac{\langle f^{cls}_{I_m}, f^{cls}_{T_k} \rangle}{\|f^{cls}_{I_m}\| \cdot \|f^{cls}_{T_k}\|} : k = 1, 2, \ldots, N_T\right\}
$$
가장 높은 점수를 받은 템플릿을 **최적 매칭 템플릿**<sub>best matching template</sub> $T_{best}$로 선정한다.
#### 외관 매칭 점수
외관 매칭 점수는 후보와 최적 템플릿 간의 패치 임베딩 유사도로 계산된다:
$$
s_{appe} = \frac{1}{N^{patch}_{I_m}} \sum_{j=1}^{N^{patch}_{I_m}} \max_{i \in \{1,\ldots,N^{patch}_{T_{best}}\}} \frac{\langle f^{patch}_{I_m,j}, f^{patch}_{T_{best},i} \rangle}{\|f^{patch}_{I_m,j}\| \cdot \|f^{patch}_{T_{best},i}\|}
$$
### 기하학적 매칭 점수
기하학적 매칭 점수는 최적 템플릿의 회전 정보와 후보의 평균 위치를 사용해 객체를 변환한 후, 투영된 **경계 상자**<sub>bounding box</sub> $B_o$와 후보의 경계 상자 $B_m$ 간 **IoU**<sub>Intersection over Union</sub>로 계산된다:
$$
s_{geo}=\frac{B_m \cap B_o}{B_m \cup B_o}
$$
가림 현상을 고려하기 위해 가시 비율 $r_{\text{vis}}$​을 계산하여 기하학적 점수의 신뢰도를 평가한다:
$$
r_{\text{vis}} = \frac{1}{N^{\text{patch}}_{T_{\text{best}}}} \sum_{i=1}^{N^{\text{patch}}_{T_{\text{best}}}} r_{\text{vis},i}
$$
여기서 $r_{\text{vis},i}$​는 패치별 가시성을 나타내는 이진 값이고 $r_{vis}$는 **가시 비율**<sub>visibility ratio</sub>이다.
최종 **객체 매칭 점수**는 세 점수를 가중 평균하여 계산된다:
$$
s_m=\frac{s_{sem}+s_{appe}+r_{vis} \cdot s_{geo}}{1+1+r_{vis}}
$$
## 3.2. 자세 추정 모델
PEM은 ISM에서 식별된 각 객체 후보에 대해 정확한 6D 자세를 예측한다. 후보의 **점 집합**<sub>point set</sub> $\mathcal{P}_m \in \mathbb{R}^{N_m \times 3}$과 목표 객체의 점 집합 $\mathcal{P}_o \in \mathbb{R}^{N_o \times 3}$ 간의 **부분-부분 점 매칭**<sub>partial-to-partial point matching</sub> 문제로 접근한다.
### 배경 토큰의 혁신적 설계
각 점 집합의 특징에 학습 가능한 **배경 토큰**<sub>background token</sub>을 추가한다. **어텐션 매트릭스**<sub>attention matrix</sub>는 다음과 같이 계산된다:
$$
\mathcal{A} = [\mathbf{f}^{\text{bg}}_m, \mathbf{F}_m] \times [\mathbf{f}^{\text{bg}}_o, \mathbf{F}_o]^T \in \mathbb{R}^{(N_m+1) \times (N_o+1)}
$$
**소프트 할당 매트릭스**<sub>soft assignment matrix</sub>는 행과 열 방향으로 각각 소프트맥스를 적용하여 얻는다:
$$
\tilde{\mathcal{A}} = \text{Softmax}_{\text{row}}(\mathcal{A}/\tau) \cdot \text{Softmax}_{\text{col}}(\mathcal{A}/\tau) \quad
$$
여기서 $\tau$는 **온도 매개변수**<sub>temperature parameter</sub>이다.
### 거친 점 매칭
**희소한 점 집합**<sub>sparse point set</sub>을 사용하여 초기 자세를 추정한다. $T^c$개의 **기하학적 변환기**<sub>geometric transformer</sub>를 거쳐 향상된 특징을 얻고, 매칭 확률을 바탕으로 6,000개의 자세 가설을 생성한다.
$$
s_{hyp}​=\frac{N^c_m}{\sum_{p_m^c​\in P_m^c}​\text{​min}_{p_o^c​∈P_o^c}​​∣∣R_{hyp}^T​(p_o^c​−t_{hyp}​)−p_m^c​∣∣2}​​​
$$
가장 높은 점수를 받은 가설을 초기 자세 $\mathbf{R}_{\text{init}},\mathbf{t}_{\text{init}}$​로 선택한다.
### 정밀한 점 매칭
**조밀한 점 집합**<sub>dense point set</sub>을 사용한다. **Sparse-to-Dense Point Transformer**<sub>SDPT</sub>를 적용하여 조밀한 대응 관계를 모델링한다. SDPT는 희소 점들 간의 기하학적 변환기 처리 결과를 **선형 교차 어텐션**<sub>linear cross attention</sub>을 통해 조밀한 특징으로 전파한다.
### 훈련 목적 함수 
**InfoNCE 손실**<sub>InfoNCE loss</sub>을 사용하여 어텐션 매트릭스를 지도학습한다:
$$
\mathcal{L} = \text{CE}(\mathcal{A}[1:, :], \hat{\mathcal{Y}}_m) + \text{CE}(\mathcal{A}[:, 1:]^T, \hat{\mathcal{Y}}_o) \quad
$$
전체 최적화 목표는 거친 매칭과 정밀한 매칭의 모든 변환기 레이어에 대한 손실의 합이다:
$$
\min \sum_{l=1}^{T^c} \mathcal{L}^c_l + \sum_{l=1}^{T^f} \mathcal{L}^f_l
$$
![[Image/Pasted image 20250626152537.png]]
# Experiments
## 4.1. 데이터셋 및 평가 환경
SAM-6D의 성능을 종합적으로 평가하기 위해 **BOP 벤치마크**<sub>Benchmark for 6D Object Pose Estimation</sub>의 7개 핵심 데이터셋을 사용했다. **YCB-V 데이터셋**은 일상생활에서 흔히 볼 수 있는 21개의 객체들로 구성되어 있으며, **LM-O 데이터셋**은 가림 현상이 있는 어려운 장면들을 포함한다. **T-LESS 데이터셋**은 질감이 없거나 매우 단순한 산업용 부품들로 구성되어 있다.
## 4.2. 인스턴스 분할 성능
ISM의 성능을 평가하기 위해 IoU 임계값 0.50부터 0.95까지에서의 **평균 정밀도**<sub>mean Average Precision; mAP</sub>를 측정했다.
개별 매칭 점수의 기여도 분석 결과:
- 의미적 매칭만: SAM 기반 44.0 mAP
- 의미적 + 외관적: SAM 기반 45.0 mAP
- 의미적 + 기하학적: SAM 기반 46.7 mAP
- 전체 조합: SAM 기반 **48.1 mAP**
## 4.3. 자세 추정 성능
자세 추정 성능은 VSD, MSSD, MSPD의 평균 **Average Recall**<sub>AR</sub>로 측정했다.
### 전체 성능 비교
- SAM-6D (SAM): **70.4 AR**
- SAM-6D (FastSAM): 66.2 AR
- GigaPose: 57.9 AR
- ZeroPose: 58.4 AR
- MegaPose: 57.2 AR
### 데이터셋 별 상세 성능 (SAM-6D with SAM)
- TUD-L: **90.4 AR** (최고 성능)
- YCB-V: 84.5 AR
- HB: 77.6 AR
- LM-O: 69.9 AR
![[Image/Pasted image 20250626152701.png]]
## 4.4. 핵심 기술 요소들의 기여도 분석
### 배경 토큰 vs 최적 수송 비교
- 배경 토큰: 84.5 AR, 1.36초/이미지
- **최적 수송**<sub>optimal transport</sub>: 81.4 AR, 4.31초/이미지 (3배 느림)
### 2단계 매칭 전략 효과
- 거친 매칭만: 77.6 AR
- 정밀한 매칭만: 40.2 AR(초기 자세 없이는 크게 저하)
- 두 단계 결합: **84.5 AR**
### 계산 효율성 분석
처리 속도 분석 결과:
- FastSAM 기반: 1.43초/이미지
- SAM 기반: 4.37초/이미지
- MegaPose: > 10초/이미지
# 5. Conclusion
## 5.1. 주요 기술적 성과
SAM-6D는 6D 객체 자세 추정 분야에서 여러 중요한 돌파구를 달성했다. 가장 핵심적인 성과는 zero-shot 방식으로 새로운 객체의 6D 자세를 추정하는 통합 프레임워크를 구축했다는 것이다.
### 구체적인 성과
- **정확도**: 70.4 AR로 20.5% 향상
- **속도**: 1.36초로 68.4% 개선 (배경 토큰 사용 시)
- **일반화**: zero-shot으로 즉시 적용 가능
## 5.2. 기술적 혁신 요약
이 연구의 기술적 혁신은 여러 측면에서 중요하다. **배경 토큰 기반 점 매칭**<sub>background token-based point matching</sub>은 기존의 **최적 수송 방식**<sub>optimal transport approach</sub>보다 3배 빠른 속도를 제공하면서도 더 높은 정확도를 달성했다.
**2단계 매칭 전략**<sub>two-stage matching strategy</sub>은 거친 매칭에서 정밀한 매칭으로 이어지는 **계층적 접근법**<sub>hierarchical approach</sub>으로 효과적인 자세 추정을 가능하게 했다. **Sparse-to-Dense Point Transformer**는 희소 점들 간의 기하학적 관계를 조밀한 특징으로 전파하여 더 정확한 **대응 관계**<sub>correspondence</sub>를 구축했다.
**삼중 매칭 점수**<sub>triple matching score</sub> 체계는 의미적, 외관적, 기하학적 정보를 통합하여 강건한 객체 식별을 가능하게 했다. 이런 여러 기술적 구성 요소들이 조화롭게 결합되어 전체적인 성능 향상을 이뤄냈다.
## 5.3. 향후 연구 방향
현재 SAM-6D는 단일 강체 물체만 처리할 수 있어서 복잡한 **관절 물체**<sub>articulated object</sub>나 **변형 가능한 물체**<sub>deformable object</sub>는 다룰 수 없다는 한계가 있다. 또한 **동적 환경**<sub>dynamic environment</sub>에서의 **시간적 일관성**<sub>temporal consistency</sub> 유지와 **다중 객체**<sub>multi-object</sub> 상황에서의 처리 능력 개선이 필요하다.
향후 연구에서는 **실시간 성능**<sub>real-time performance</sub> 최적화를 통해 로봇 응용에서의 실용성을 높이는 것이 중요하다. 특히 **에지 디바이스**<sub>edge device</sub>에서도 동작할 수 있는 경량화된 모델 개발이 필요하다.
또한 **불확실성 추정**<sub>uncertainty estimation</sub>을 통해 자세 예측의 신뢰도를 정량화하고, **적응적 학습**<sub>adaptive learning</sub>을 통해 새로운 환경에 점진적으로 적응할 수 있는 시스템 구축이 중요한 연구 방향이다.
**다중 모달리티 융합**<sub>multi-modality fusion</sub>을 통해 RGB-D 정보 외에 **열화상**<sub>thermal</sub>, **촉각**<sub>tactile</sub> 정보 등을 통합하여 더 강건한 자세 추정을 수행하는 것도 흥미로운 연구 주제이다.
## 5.4. 미래 전망
SAM-6D가 제시한 방향성은 **기반 모델**<sub>foundation model</sub> 접근법의 강력함을 입증하며, 6D 객체 자세 추정 분야에서 **패러다임 전환**<sub>paradigm shift</sub>을 제시했다. 앞으로 이 연구가 제시한 방향성을 따라 더욱 발전된 AI 시스템들이 개발될 것으로 기대된다.
특히 **대규모 언어 모델**<sub>Large Language Model</sub>과 **비전 모델**<sub>vision model</sub>의 융합, **멀티모달 학습**<sub>multimodal learning</sub>의 발전과 함께 더욱 지능적이고 범용적인 6D 자세 추정 시스템이 구현될 것이다. 이러한 발전은 궁극적으로 인간과 로봇이 공존하는 지능형 환경 구축에 핵심적인 역할을 할 것으로 전망된다.