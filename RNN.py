import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# 시퀀스 길이, 입력 차원 및 출력 차원 정의
sequence_length = 10
input_dim = 5
output_dim = 1

# RNN 모델 생성
model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(sequence_length, input_dim)))
model.add(Dense(units=output_dim))

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')

# 입력 데이터 준비
# 여기서는 임의의 입력 데이터로 대체될 수 있습니다.
import numpy as np
X = np.random.random((100, sequence_length, input_dim))
y = np.random.random((100, output_dim))

# 모델 학습
model.fit(X, y, epochs=10, batch_size=16)
