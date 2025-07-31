---
{"publish":true,"created":"2025-07-28","modified":"2025-07-31T16:24:33.345+09:00","tags":["6d","ai","computer_vision","deep_learning","article"],"cssclasses":""}
---

# 1. Introduction
## 1.1. 연구 배경과 문제 정의
6D 자세 추정<sub>6D pose estimation</sub>은 물체의 3차원 위치 $(x,y,z)$와 회전 $(roll, pitch, yaw)$을 모두 정확히 찾아내는 컴퓨터 비전의 핵심 기술이다. 이 기술은 로봇이 물건을 정확히 조작하거나 증강현실AR에서 가상 물체를 현실 세계에 정확히 배치할 때 반드시 필요한 기반 기술이다. 하지만 기존의 6D 자세 추정 방법들은 근본적으로 세 가지 큰 문제점을 안고 있었다.
### 기존 방법들의 주요 한계점
- **인스턴스 수준 방법**<sub>instance-level method</sub>: 학습에서 본 특정 물체만 인식 가능, 새로운 물체에는 적용 불가
- **카테고리 수준 방법**<sub>category-level method</sub>: 미리 정의된 카테고리 내에서만 작동, 새로운 형태 등장 시 어려움
- **설정 분리 문제**: 모델 기반<sub>model-based</sub>과 모델 프리<sub>model-free</sub> 설정이 완전히 분리되어 유연성 부족
## 1.2. 기존 연구의 한계점 심화 분석
기존 연구들을 자세히 살펴보면 크게 세 가지 접근법으로 나눌 수 있다. **모델 기반 방법**들은 PVNet, DPOD, PoseCNN 등이 대표적인데, 높은 정확도를 보이지만 새로운 물체마다 정교한 3D CAD 모델이나 세밀한 질감<sub>texture</sub> 정보가 포함된 메시<sub>mesh</sub>가 반드시 필요하다는 비현실적인 요구사항을 만들어낸다.
**모델 프리** 방법들은 Gen6D, OnePose, ZeroPose 등이 대표적이며, 3D 모델 대신 여러 각도에서 찍은 참조 이미지들이나 동영상 시퀀스가 필요하다. 이들은 어느 정도 일반화 능력을 보이지만, 여전히 다중 시점<sub>multi-view</sub>이나 비디오 시퀀스<sub>video sequence</sub>가 필요하다는 제약이 있다.
최근의 **Oryon**과 같은 방법은 언어적 가이드를 활용해서 진전을 보였지만, 폐색<sub>occlusion</sub>이 발생하거나 참조 이미지와 타겟 이미지 사이에 겹치는 부분이 적을 때, 또는 물체의 질감이 부족하거나 조명 조건이 크게 다를 때 성능이 급격히 떨어지는 근본적인 한계를 가지고 있다.
## 1.3. Any6D의 핵심 혁신과 기여점
Any6D는 이러한 기존 방법들의 한계를 극복하기 위해 혁신적인 접근 방식을 제안한다. 핵심 아이디어는 "단 하나의 RGB-D 앵커 이미지만 있으면 완전히 새로운 물체도 다양한 환경에서 정확한 자세를 찾아낼 수 있다"는 것이다.
### Any6D의 주요 기여점
- **완전한 모델 프리 프레임워크**: 3D CAD 모델이나 다중 시점 이미지 없이 작동하는 최초의 방법
- **단일 RGB-D 이미지**: 정말 단 한 장의 이미지만으로 새로운 물체의 자세 추정 가능
- **결합 객체 정렬**<sub>joint object alignment</sub>: 2D와 3D 정보를 동시에 활용하는 혁신적인 정렬 기법
- **렌더링 비교 전략**<sub>render-and-compare strategy</sub>: 도전적인 시나리오에서도 강건한<sub>robust</sub> 성능 보장
![[3. Resource/Image/Pasted image 20250729114000.png]]
# 2. Related Works
## 2.1. 인스턴스 수준 6D 자세 추정
인스턴스 수준 6D 자세 추정 분야에서는 PVNet, DPOD, PoseCNN과 같은 대표적인 방법들이 개발되어 왔다. PVNet은 픽셀 단위 투표<sub>pixel-wise voting</sub>
를 통해 키포인트<sub>keypoint</sub>를 찾고 이를 바탕으로 자세를 계산하는 방식을 사용하며, DPOD는 조밀한 대응관계<sub>dense correspondence</sub>를 활용한다. PoseCNN은 합성곱 신경망<sub>convolutional neural network</sub>을 사용해서 직접적으로 자세를 회귀<sub>regression</sub>하는 접근법을 취한다.
이들은 모두 뛰어난 정확도를 보이지만, 학습할 때 본 특정한 물체들만 인식할 수 있으며, 완전히 새로운 물체가 나타나면 처음부터 다시 학습해야 한다는 치명적인 한계를 가지고 있다. 예를 들어, 특정 브랜드의 특정 모델 '컵'으로 학습한 모델은 다른 브랜드나 다른 디자인의 컵에 대해서는 전혀 작동하지 않는다.
## 2.2. 카테고리 수준 6D 자세 추정
카테고리 수준 접근법은 인스턴스 수준의 한계를 어느정도 극복하기 위해 등장했다. NOCS<sub>Normalized Object Coordinate Space</sub>, SPD<sub>Shape Prior Deformation</sub>, FS-Net<sub>Few-Shot learning Network</sub> 등이 대표적인 방법들이다. 이들은 물체들을 '컵', '병', '신발', '카메라' 같은 카테고리로 분류하고, 각 카테고리 내에서 다양한 인스턴스들에 대해 일반화된 자세 추정을 수행한다.
그러나 이러한 방법들도 미리 정의된 카테고리에서 벗어나는 물체들에 대해서는 역시 무용지물이 되며, 각 카테고리에 대한 대량의 주석<sub>annotation</sub>이 된 데이터가 필요하다는 근본적인 제약을 가지고 있다.
## 2.3. 모델 프리 접근법
모델 프리 접근법들은 3D CAD 모델이나 사전 정의된 카테고리 없이 자세 추정을 수행하려는 시도들이다. Gen6D, Oryon, GEDI, OnePose 등이 이 분야의 대표적인 연구들이다.
이들 방법 중에서 Oryon은 특히 주목할 만한데, 단일 RGB-D 참조 이미지와 언어적 설명을 사용해서 완전히 다른 환경에서도 자세 추정을 수행할 수 있다는 점에서 Any6D와 유사한 목표를 가지고 있다. 하지만 Oryon도 폐색이 발생하거나 참조와 타겟 이미지 사이에 겹치는 영역이 최소한일 때, 또는 물체의 질감이 부족할 때 성능이 급격히 저하되는 중요한 한계점들을 가지고 있다.
## 2.4. Any6D의 차별점과 위치
Any6D는 기존의 모든 방법들과 구별되는 독특한 특징들을 가지고 있다. 가장 중요한 차별점은 **정말 단 한 장의 RGB-D 이미지**만으로 완전히 새로운 물체의 자세를 추정할 수 있다는 것이다. 이는 다중 시점이나 비디오 시퀀스, 언어적 설명, 또는 사전 3D 모델 등 어떤 추가적인 정보도 필요로 하지 않는다.
![[3. Resource/Image/Pasted image 20250729114103.png]]
# 3. Method
## 3.1. 전체 프레임워크 개요와 핵심 철학
Any6D의 전체 프레임워크는 세 개의 주요 단계로 구성되어 있으며, 각 단계가 유기적으로 연결되어 최종적인 자세 추정을 달성한다. 핵심 철학은 **사람이 물체를 인식하는 방식을 모방하자**는 것이다. 사람은 한 번 본 물체를 다른 각도에서 봐도, 조명이 달라도, 일부가 가려져도 쉽게 인식할 것이다.
### 프레임워크 구성 단계
1. **형태 복원**<sub>Shape Reconstruction</sub>: 앵커 이미지로부터 물체의 정규화된<sub>normalized</sub> 3D 형태 $O_N$ 생성
2. **객체 정렬**<sub>Object Alignment</sub>: 2D-3D 결합 정렬을 통해 정확한 자세와 실제 척도<sub>metric scale</sub> 추정
3. **자세 추정**<sub>Pose Estimation</sub>: 쿼리 이미지에서 복원된 실제 척도 객체 형태 $O_M$을 활용한 최종 자세 추정
## 3.2. 결합 객체 정렬의 수학적 기초
결합 객체 정렬은 Any6D의 가장 중요한 기술적 기여 중 하나이다. 이 방법의 핵심 아이디어는 **2D 이미지 정보**와 **3D 점군<sub>point cloud</sub> 데이터**를 동시에 활용해서 정렬을 수행한다는 것이다. 수학적으로 이는 다음과 같이 공식화할 수 있다.
$$
E_{total}=\lambda_1 \cdot E_{2D}+\lambda_2 \cdot E_{3D} + \lambda_3 \cdot E_{scale}
$$
여기서 $E_{2D}$는 2D 특징 매칭 오차, $E_{3D}$는 3D 기하학적 정렬 오차, $E_{scale}$은 실제 척도 추정 오차를 나타낸다. 이러한 결합 최적화<sub>joint optimization</sub> 과정은 반복적<sub>iterative</sub>으로 수행되며, 각 반복에서 2D와 3D 정렬 오차를 동시에 최소화하는 방향으로 매개변수들이 업데이트된다.
## 3.3. 렌더링-비교 전략의 세부 구현
렌더링-비교 전략은 Any6D가 다양한 도전적인 시나리오에서 강건한 성능을 보이는 핵심 이유 중 하나이다. 이 전략은 **다중 자세 가설 생성**<sub>multiple pose hypothesis generation</sub>, **렌더링**<sub>rendering</sub>, **비교**<sub>comparison</sub>의 3단계로 구성된다.
### 전략 구성 요소
- **가설 생성**<sub>hypothesis generation</sub>: 초기 정렬 결과 기반 $N$개 자세 변형<sub>variation</sub> 생성
- **렌더링 과정**<sub>rendering process</sub>: 각 가설에 대해 합성 RGB-D 이미지 생성
- **비교 및 선택**<sub>comparison and selection</sub>: 광도 유사성<sub>photometric similarity</sub>, 깊이 일관성<sub>depth consistency</sub>, 기하학적 정렬<sub>geometric alignment</sub>을 종합 평가하여 최적 자세 선택
최종적으로 가장 높은 점수를 가진 가설이 선택되며, 이는 폐색이나 극한 시점 변화<sub>extreme viewpoint change</sub>가 있는 상황에서도 올바른 자세를 찾아낼 수 있게 해준다.
## 3.4. 실제 척도 추정의 정밀한 구현
실제 척도 추정은 Any6D가 실제 응용에서 유용하게 사용될 수 있게 하는 중요한 기능이다. 많은 기존 방법들이 상대적 크기나 정규화된 척도만을 제공했던 것과 달리, Any6D는 실제 물리적 크기를 미터 단위로 정확히 추정한다.
### 척도 추정 과정
- **다중 영역 척도 추정**<sub>multi-region scale estimation</sub>: 물체의 여러 영역에서 독립적으로 척도 추정
- **강건한 척도 융합**<sub>robust scale fusion</sub>: RANSAC 기반 강건한 추정으로 이상치<sub>outlier</sub> 제거
- **신뢰도 추정**<sub>confidence estimation</sub>: 척도 추정의 신뢰도 계산으로 불확실한 경우 감지
![[3. Resource/Image/Pasted image 20250729114117.png]]
# 4. Experiments
## 4.1. 실험 설계와 데이터셋 구성
Any6D의 성능을 종합적으로 평가하기 위해 총 5개의 데이터셋에서 광범위한 실험을 수행했다. 각 데이터셋은 서로 다른 특성과 난이도를 가지고 있어 방법론의 다양한 측면을 평가할 수 있게 한다.
### 평가 데이터셋
- **Real275**: 실제 환경 275개 시퀀스, 다양한 조명과 배경 잡음<sub>background clutter</sub> 포함
- **Toyota-Light**: Toyota 산업용 데이터셋, 제조업 환경의 부품 인식 평가
- **HO3D**: 사람 손-물체 상호작용, 손에 의한 폐색과 동적 조작 포함
- **YCBInEOAT**: 일상용품 데이터셋, 대칭성이나 질감 부족 물체들로 강건성 테스트
- **LM-O**: 심각한 폐색 시나리오, 물체 상당 부분이 가려진 상황
## 4.2. 평가 지표와 주요 실험 결과
실험에서는 6D 자세 추정 분야에서 표준적으로 사용되는 여러 평가 지표를 활용했다. ADD는 추정된 자세로 변환한 3D 모델 점들과 실측값<sub>ground truth</sub> 자세로 변환한 점들 사이의 평균 거리를 측정하고, ADD-S는 대칭 물체를 위한 버전이다. 5cm 5° 지표는 실제 응용에서 중요한 실용적 정확도<sub>practical accuracy</sub>를 측정한다.
### 전체 성능 비교 결과
- **REAL275**: Any6D 78.4% ADD-S vs. Oryon 62.1% (16.3%p 향상)
- **HO3D**: Any6D 85.2 ADD vs. 기존 최고 67.8% (17.4%p 향상)
- **YCBInEOAT**: Any6D 91.7% ADD-S vs. GEDI 76.3% (15.4%p 향상)
- **LM-O**: Any6D 83.6% ADD(-S) vs. Gen6D 71.2% (12.4%p 향상)
- **Toyota-Light**: Any6D 89.3% ADD vs. ZeroPose 74.5% (14.8%p 향상)
### 폐색 강건성 분석
- 폐색 < 30%: 92.3 ADD
- 폐색 30-50%: 87.1 ADD
- 폐색 50-70%: 82.6 ADD
- 폐색 > 70%: 74.2% ADD
![[3. Resource/Image/Pasted image 20250729114217.png]]
## 4.3. 기여도 분석
각 구성요소의 기여도를 정확히 파악하기 위해 체계적인 제거 연구를 수행했다. 결합 정렬 구성요소를 제거한 실험에서는 전반적인 성능이 평균 17.3% 감소했으며, 특히 도전적인 시나리오에서의 성능 저하가 더욱 두드러졌다.
### 구성요소별 기여도 분석
- 전체 방법<sub>Full Method</sub>: 85.2% ADD-S
- 결합 정렬 없이<sub>Without joint alignment</sub>: 67.9% ADD-S (-17.3%p)
- 2D 전용 정렬<sub>2D-only alignment</sub>: 71.4 ADD-S(-13.8%p)
- 3D 전용 정렬<sub>3D-only alignment</sub>: 69.8% ADD-S (-15.4%p)
### 렌더링-비교 전략 효과
- 다중 가설<sub>Multiple hypothesis</sub> (N=8): 85.2% ADD-S
- 단일 가설<sub>Single hypothesis</sub>: 76.8% ADD-S (-8.4%p)
- 무작위 선택<sub>Random selection</sub>: 68.3% ADD-S (-16.9%p)
### 가설 개수별 성능 변화
- N=3: 81.7% ADD-S / N=5: 83.9% ADD-S / N=8: 85.2% ADD-S
- N=12: 85.4% ADD-S (미미한 개선<sub>marginal improvement</sub>) / N=16: 85.3% ADD-S (계산 부담<sub>computational overhead</sub>)
## 4.4. 실제 로봇 시나리오에서의 검증
실제 로봇 시스템에서의 적용 가능성을 검증하기 위해 **Universal Robots UR5** DOF 로봇팔을 사용한 종합적인 조작 실험을 수행했다. 실험에서는 사람의 손이나 로봇 팔 자체가 물체를 가리는 현실적인 상황들을 의도적으로 포함시켰다.
### 로봇 실험 성능 결과
- **파지 성공률**<sub>Grasping Success Rate</sub>: 알려진 물체 94.2% vs. 새로운 물체 with Any6D 87.6%
- **동적 폐색 대응**<sub>Dynamic Occlusion</sub>: 정적 장면 89.3% → 심한 폐색 82.1%
- **조명 조건 변화**: 제어된 조명 88.4% → 산업용 조명 84.2%
### 교차 환경 성능
- 동일 환경<sub>Same environment</sub>: 89.4% ADD-S
- 다른 조명<sub>Different lighting</sub>: 85.7% ADD-S
- 다른 배경<sub>Different background</sub>: 83.2% ADD-S
- 완전히 다른 장면<sub>Completely different scene</sub>: 79.6% ADD-S
## 4.5. 계산 효율성과 실시간 성능
처리 시간 분석을 Intel i9-12900K CPU와 NVIDIA RTX 4090 GPU 환경에서 수행했다. 이는 실제 로봇 응용에서 요구되는 준 실시간 수준에 근접한 성능이다.
### 처리 시간 분석
- 형태 복원<sub>Shape reconstruction</sub>: 0.32초
- 객체 정렬<sub>Object alignment</sub>: 0.78초
- 자세 추정<sub>Pose estimation</sub>: 0.41초
- **객체당 총 시간<sub>Total per object</sub>: 1.51초**
### 메모리 사용량
- 형태 복원 모델<sub>Shape reconstruction model</sub>: 2.1GB GPU 메모리
- 정렬 네트워크<sub>Alignment network</sub>: 1.4GB GPU 메모리
- 렌더링 버퍼<sub>Rendering buffers</sub>: 0.8GB GPU 메모리
- **총 최대 사용량<sub>Total peak usage</sub>: 4.3GB GPU 메모리**
# 5. Conclusion
## 5.1. 주요 기술적 성과와 혁신
Any6D는 6D 자세 추정 분야에서 여러 중요한 돌파구를 달성했다. 가장 핵심적인 성과는 완전한 모델 자유 프레임워크의 개발이다. 이는 3D CAD 모델이나 다중 시점 이미지, 그리고 복잡한 사전 설정 없이도 작동하는 최초 방법으로, 실제 응용에서의 접근성을 크게 향상시켰다.
### 핵심 기술적 성과
- **완전한 모델 자유 프레임워크**: 진정한 "즉시 사용<sub>plug-and-play</sub>" 방식의 6D 자세 추정 실현
- **단일 RGB-D 이미지 기반**: 모바일 로봇이나 휴대용 AR 기기에서도 적용 가능
- **결합 객체 정렬<sub>joint object alignment</sub>**: 2D와 3D 정보 동시 활용으로 17.3%p 성능 향상 달성
- **렌더링-비교 전략<sub>render-and-compare strategy</sub>**: 도전적인 시나리오에서 8.4%p 추가 성능 향상
결합 객체 정렬 기법은 기존의 단일 양식 방법들보다 훨씬 강건하고 정확한 결과를 제공하며, 렌더링-비교 전략을 통한 다중 가설 검증은 폐색이나 극한 시점 변화가 있는 상황에서 기존 방법들이 실패하는 경우를 효과적으로 해결했다.
## 5.2. 실용적 임팩트와 산업적 의미
Any6D의 실용적 임팩트는 여러 분야에 걸쳐 광범위하게 나타날 것으로 예상된다. 로봇 공학 분야에서는 로봇이 처음 보는 물체도 바로 조작할 수 있게 함으로써 유연성과 적응성을 크게 향상시킬 것이다. 특히 산업 4.0과 대량 맞춤화 트렌드에 매우 적합하다.
### 분야별 임팩트
- **로봇 공학<sub>Robotics</sub>**: 새로운 물체 즉시 대응, 87.6% 파지 성공률로 실용화 수준 달성
- **증강현실/가상현실<sub>AR/VR</sub>**: 별도 스캔닝 없는 즉시 추적, AR 쇼핑과 가상 체험<sub>virtual try-on</sub> 혁신
- **제조업<sub>Manufacturing</sub>**: 새로운 부품 도입 시 별도 학습 없는 즉시 적용, 생산 라인production line 유연성 향상
- **의료<sub>Healthcare</sub>**: 최소 침습 수술<sub>minimally invasive surgery</sub>에서 수술 도구 실시간 추적
실제 로봇 실험에서 87.6%의 파지 성공률을 달성한 것은 실용화 가능한 수준의 성능임을 보여주며, Toyota-Light 데이터셋에서 89.3%의 성능을 달성한 것은 실제 제조업 환경에서도 충분히 활용 가능함을 시사한다.
## 5.3. 한계점과 기술적 도전
Any6D가 뛰어난 성능을 보임에도 불구하고 여전히 몇 가지 한계점이 존재한다. 투명하거나 높은 반사성을 가진 물체들에 대해서는 RGB-D 센서의 근본적인 한계로 인해 여전히 어려움이 있다. 매우 작은 물체들에 대해서도 깊이 정보의 해상도나 정확도가 물체의 크기에 비해 상대적으로 부족할 때 정확한 형태 복원이나 자세 추정이 어려워진다.
### 현재 한계점들
- **투명/반사 물체**: RGB-D 센서의 근본적 한계로 형태 복원 어려움
- **매우 작은 물체**: 서브 밀리미터<sub>sub-millimeter</sub> 정확도 요구되는 정밀 제조<sub>precision manufacturing</sub>에서 제약
- **처리 시간<sub>Processing time</sub>**: 1.51초로 고속 조작<sub>high-speed manipulation</sub>이나 빠른 움직임 물체 추적<sub>fast-moving object tracking</sub>에 제약
- **극한 조명<sub>Extreme lighting</sub>**: 매우 어둡거나 밝은 환경, 극적인 그림자<sub>dramatic shadow</sub>에서 RGB 정보 품질 저하
## 5.4. 향후 연구 방향과 기술 발전
향후 연구 방향으로는 여러 가지 유망한 영역들이 있다. 형태 개선 방면에서는 **신경 복사장**<sub>Neural Radiance Fields; NeRF</sub>나 3D 가우시안 스플래팅<sub>3D Gaussian Splatting</sub>같은 최신 3D 표현<sub>representation</sub> 기법들을 통합하는 것이 흥미로운 방향이다.
### 주요 연구 방향들
- **형태 개선<sub>Shape refinement</sub>**: NeRF, 3D 가우시안 스플래팅 등 최신 3D 표현 기법 통합
- **다중 객체 시나리오<sub>Multi-object scenario</sub>**: 객체 검출/분할 통합, 공간 관계<sub>spatial relationship</sub> 고려
- **시간적 일관성<sub>Temporal consistency</sub>**: 칼만 필터<sub>Kalman filter</sub>, 파티클 필터<sub>particle filter</sub> 등 고전적 추적 기법과의 융합
- **모바일 배치<sub>Mobile deployment</sub>**: 지식 증류<sub>Knowledge distillation</sub>, 양자화<sub>quantization</sub>, 가지치기<sub>pruning</sub>을 통한 경량화
- **기반 모델 통합<sub>Foundation model integration</sub>**: 대규모 비전-언어 모델<sub>Large Vision-Language Models(VLMs)</sub>과의 융합
다중 객체 시나리오는 특히 중요한 연구 방향이다. 현재 Any6D는 주로 단일 객체에 초점을 맞추고 있지만, 실제 환경에서는 여러 물체가 함께 있는 경우가 많다. 객체 검출과 분할을 통합하고, 다중 객체들 간의 공간 관계를 고려하는 방법이 필요하다.
## 5.5. 패러다임 전환과 미래 전환
Any6D가 제시한 방향성은 **기반 모델 접근법**<sub>foundation model approach</sub>의 강력함을 입증하며, 6D 객체 자세 추정 분야에서 **패러다임 전환**<sub>paradigm shift</sub>을 제시했다. 기존의 작업 특화적<sub>task-specific</sub>하고 도메인 제한적<sub>domain-limited</sub>한 접근법에서 벗어나, **범용 목적**<sub>general-purpose</sub>이고 **광범위한 적용 가능**<sub>broadly applicable</sub>한 방향으로의 전환을 보여준다.
이러한 변화는 영상 처리 분야 전반에서 일어나고 있는 큰 흐름과 일치한다. GPT나 CLIP 같은 기반 모델들이 자연어 처리와 비전 분야에서 큰 성공을 거둔 것처럼, 6D 자세 추정에서도 **사전 학습된 일반 모델**<sub>pre-trained general model</sub>이 작업 특화 미세 조정<sub>task-specific fine-tuning</sub>보다 더 효과적일 수 있음을 보여준다.
### 미래 전망
- **체화된 인공지능<sub>Embodied AI</sub>**: 로봇의 인간 수준 시각 이해 달성에 핵심 구성 요소
- **범용 목적 로봇<sub>General Purpose Robot</sub>**: 새로운 환경에서 새로운 물체 즉시 인식/조작 능력
- **스마트 환경<sub>Smart Environment</sub>**: 스마트 홈, 지능형 공장, 자율 서비스 로봇의 필수 기술
- **삶의 질 향상**: 인간의 삶의 질 향상과 산업 생산성 혁신에 직접 기여
궁극적으로 이러한 발전은 **인간과 로봇이 공존하는 지능형 환경** 구축에 핵심적인 역할을 할 것으로 전망된다. Any6D가 보여준 "단일 이미지로 새로운 물체를 즉시 인식"하는 능력은 미래의 지능형 시스템들에서 필수적인 기술이 될 것이다.