import numpy as np

class PerceptronModel:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand(1)
    
    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def train(self, inputs, outputs, learning_rate=0.1, epochs=20):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                total_input = np.dot(inputs[i], self.weights) + self.bias
                prediction = self.step_function(total_input)
                error = outputs[i] - prediction
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error
    
    def predict(self, input_data):
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)

class AndModel(PerceptronModel):
    def __init__(self):
        super().__init__(2)
    
    def train(self):
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([0, 0, 0, 1])
        super().train(inputs, outputs)

class OrModel(PerceptronModel):
    def __init__(self):
        super().__init__(2)
    
    def train(self):
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([0, 1, 1, 1])
        super().train(inputs, outputs)

class NotModel(PerceptronModel):
    def __init__(self):
        super().__init__(1)
    
    def train(self):
        inputs = np.array([[0], [1]])
        outputs = np.array([1, 0])
        super().train(inputs, outputs)

class XorModel:
    def __init__(self):
        # AND, OR, NOT 퍼셉트론 모델을 사용하여 XOR 구현
        self.and_model = AndModel()  # AND 연산을 담당할 모델
        self.or_model = OrModel()    # OR 연산을 담당할 모델
        self.not_model = NotModel()  # NOT 연산을 담당할 모델
    
    def train(self):
        # 각각의 논리 연산 모델을 학습
        self.and_model.train()
        self.or_model.train()
        self.not_model.train()
    
    def predict(self, input_data):
        # OR 연산 수행
        or_result = self.or_model.predict(input_data)  
        
        # AND 연산 수행
        and_result = self.and_model.predict(input_data)  
        
        # AND 결과를 NOT 연산 수행 (AND 결과를 반전)
        not_and_result = self.not_model.predict([and_result])  
        
        # 최종 XOR 연산: OR 결과와 NOT(AND 결과)를 AND 연산
        return self.and_model.predict([or_result, not_and_result])

# 미리 학습된 모델 저장
models = {
    "AND": AndModel(),
    "OR": OrModel(),
    "NOT": NotModel(),
    "XOR": XorModel()
}
for model in models.values():
    model.train()