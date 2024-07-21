# Numpy로 구현한 간단한 신경망 
이 프로젝트는 Numpy를 사용하여 간단한 신경망을 구현하고, XOR 문제를 해결하는 예제입니다. 

## 개요 

이 프로젝트는 다음과 같은 목표를 가지고 있습니다:

1. Numpy를 사용하여 신경망의 기본 개념을 이해합니다.
 
## 수학적 배경 

### 활성화 함수 

활성화 함수는 뉴런의 출력값을 결정하는 데 사용됩니다. 여기서는 시그모이드 함수(sigmoid function)를 사용합니다.
 
- 시그모이드 함수:

$$
 \sigma(x) = \frac{1}{1 + e^{-x}} 
$$
 
- 시그모이드 함수의 도함수:

$$
 \sigma'(x) = \sigma(x) \cdot (1 - \sigma(x)) 
$$

### 손실 함수 

손실 함수는 예측값과 실제값 간의 차이를 측정합니다. 여기서는 평균 제곱 오차(Mean Squared Error, MSE)를 사용합니다.
 
- 평균 제곱 오차:

$$
 \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 
$$

### 순전파 

순전파(Feedforward) 과정에서는 입력값이 네트워크를 통해 전달되어 출력값을 생성합니다.
 
- 은닉층 활성화:

$$
 h = \sigma(X \cdot W_{ih} + b_h) 
$$
 
- 출력층 활성화:

$$
 \hat{y} = \sigma(h \cdot W_{ho} + b_o) 
$$

### 역전파 

역전파(Backpropagation) 과정에서는 손실 함수를 최소화하기 위해 가중치와 바이어스를 업데이트합니다.
 
- 출력층 오차와 델타:

$$
 \delta_o = (y - \hat{y}) \cdot \sigma'(\hat{y}) 
$$
 
- 은닉층 오차와 델타:

$$
 \delta_h = (\delta_o \cdot W_{ho}^T) \cdot \sigma'(h) 
$$
 
- 가중치와 바이어스 업데이트:

$$
 W_{ho} += h^T \cdot \delta_o \cdot \eta 
$$

$$
 W_{ih} += X^T \cdot \delta_h \cdot \eta 
$$

$$
 b_o += \sum \delta_o \cdot \eta 
$$

$$
 b_h += \sum \delta_h \cdot \eta 
$$


## 실행 방법 

1. 필요한 라이브러리를 설치합니다.


```bash
pip install  numpy
```

```bash
python xor_neural_network.py
```

## 결과 

신경망이 XOR 문제를 해결하기 위해 학습하고, 학습 진행 상황을 시각적으로 확인할 수 있습니다. 학습이 완료되면, 예측된 출력값을 확인할 수 있습니다.
