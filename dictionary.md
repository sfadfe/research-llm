# dictionary

이 리서치에서 쓰는 용어. 대화, 문서, 실행 이름은 이 용어를 따른다.

| 용어 | 뜻 | 코드에서의 이름 |
|---|---|---|
| DeltaNet | 기준이 되는 방식. Gated DeltaNet의 α, β를 현재 토큰 x_t만으로 정한다. | `gate_mode = "linear"` |
| okkie-attention | 내 방식. 기준 모델은 DeltaNet이고, 상태 갱신식은 DeltaNet과 같다. pooling: 문맥 경계에서 구간을 나누고, 구간마다 key/value를 pooling해 뭉탱이 하나를 만들어 저장한다. 저장량은 구간 수에 비례하고, 토큰별 KV 원본은 저장하지 않는다. 유사도: 현재 토큰과 각 뭉탱이의 유사도를 구한다. 읽기: 유사도가 낮은 구간은 뭉탱이를 KV 한 쌍으로 그대로 쓰고, 높은 구간은 복원 레이어(nn.Linear)로 KV m쌍으로 펼쳐 쓰고, 나머지 구간은 복원을 건너뛴다. 복원은 원본과 같을 필요가 없다. 수명 게이트: head마다 뭉탱이 m_h를 두고 w_h = sigmoid(τ·cos(c_t, m_h) + b). α_h ← α_h^w_h, β_h ← w_h·β_h, 출력에도 w_h를 곱한다. w_h = 0이면 그 head의 상태는 감쇠도 쓰기도 없이 보존된다. 미정: 경계 감지 방법, 뭉탱이 크기, m, 높은 유사도의 기준, 읽은 결과를 넣는 위치. | — |
| 뭉탱이 | 같은 문맥이 이어지는 구간의 key/value를 pooling해 만든 벡터 하나. 문맥이 바뀌면 새 뭉탱이가 시작된다. (예: AAABBBACCC → a1, b1, a2, c1) | — |
| plain | DeltaNet을 적용한 모델. | 실행 이름 `plain` (이전 표기: A) |
| okkie-test-117m | 이전 okkie-attention(과거 [k; v]를 감쇠 가중 pooling해 게이트를 정함, `gate_mode="pool"`)으로 학습한 117M 모델. | 실행 이름 `okkie-transformer-117M` (이전 표기: B) |
