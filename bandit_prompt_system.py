import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import transformers
import torch

# 0. LLaMA 파이프라인 초기화

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

llama_pipe = transformers.pipeline(
    "text-generation",
    model=MODEL_ID,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# 1. 프롬프트 ARM 정의

@dataclass
class PromptArmConfig:
    """
    각 프롬프트 전략(arm)에 대한 메타 정보.
    feature_vector는 이 전략의 성향을 나타내는 임베딩(스타일/목적 가중치 등).
    """
    name: str
    description: str
    feature_vector: np.ndarray  # shape = (d,)

# 2. Logistic Bandit (arm별 w, H + UCB)

class LogisticPromptBandit:
    """
    프롬프트 전략 선택용 logistic bandit.
    - 각 arm별로 logistic 모델 w_a, H_a 유지
    - UCB 스타일로 exploration (mean + beta * uncertainty)
    """
    def __init__(
        self,
        arms: List[PromptArmConfig],
        lambda_reg: float = 1.0,
        eta: float = 0.1,
        beta: float = 1.0,
    ):
        self.arms = arms
        self.n_arms = len(arms)
        self.d = arms[0].feature_vector.shape[0]

        # arm별 parameter, Hessian 근사
        self.w = np.zeros((self.n_arms, self.d))  # w_a
        self.H = [lambda_reg * np.eye(self.d) for _ in range(self.n_arms)]

        self.eta = eta   # step size
        self.beta = beta # exploration 강도

    def _sigmoid(self, z: float) -> float:
        return 1.0 / (1.0 + np.exp(-z))

    def select_arm(self) -> int:
        """
        현재까지의 w, H를 기반으로 어느 프롬프트 전략 arm을 쓸지 선택.
        score_a = (w_a^T x_a) + beta * sqrt( x_a^T H_a^{-1} x_a )
        """
        scores = []
        for a_idx, arm in enumerate(self.arms):
            x = arm.feature_vector  # (d,)
            mean = float(self.w[a_idx].dot(x))

            invH = np.linalg.inv(self.H[a_idx])
            var = float(x.T @ invH @ x)
            ucb = self.beta * np.sqrt(max(var, 1e-12))

            scores.append(mean + ucb)

        chosen = int(np.argmax(scores))
        return chosen

    def update(self, arm_idx: int, reward: float):
        """
        bandit reward(0~1 범위)를 받아 online logistic regression update.
        reward는 verifier에서 나온 스칼라 점수.
        """
        arm = self.arms[arm_idx]
        x = arm.feature_vector
        z = float(self.w[arm_idx].dot(x))
        p = self._sigmoid(z)  # 현재 model이 보는 "성공 확률"

        # logistic loss gradient: -(y - p) * x
        grad = -(reward - p) * x

        # Hessian 근사 업데이트: H_a += x x^T
        self.H[arm_idx] += np.outer(x, x)

        # Online Newton-style step: w <- w - eta * H^{-1} grad
        invH = np.linalg.inv(self.H[arm_idx])
        step = self.eta * (invH @ grad)
        self.w[arm_idx] -= step

    def explain_current_strategy(self) -> List[str]:
        """
        각 arm에 대해 현재 bandit가 보는 "예상 성공도"를 텍스트로 정리.
        """
        lines = []
        for a_idx, arm in enumerate(self.arms):
            x = arm.feature_vector
            z = float(self.w[a_idx].dot(x))
            p = self._sigmoid(z)
            lines.append(
                f"[{arm.name}] "
                f"예상 성공도 ≈ {p:.3f} | "
                f"feature={np.round(x, 2)} | "
                f"설명: {arm.description}"
            )
        return lines

# 3. LLM Generator (Meta-Llama-3 기반)


class LlamaChatGenerator:
    """
    Meta-Llama-3-8B-Instruct HF pipeline을 감싸는 generator 래퍼.
    system_prompt + user_prompt를 받아 답변을 생성.
    """
    def __init__(self, pipe):
        self.pipe = pipe
        self.tokenizer = pipe.tokenizer

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        out = outputs[0]["generated_text"]

        # HF 버전에 따라 구조가 다를 수 있어 방어적으로 처리
        if isinstance(out, list):
            for m in reversed(out):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    return m.get("content", "").strip()
            return str(out)
        else:
            return str(out).strip()


def build_prompt_from_arm(
    base_instruction: str,
    complaint_text: str,
    arm: PromptArmConfig,
) -> str:
    """
    arm의 전략 설명을 user 프롬프트에 녹여서 스타일/방향을 제어.
    - base_instruction: system 쪽에 들어가는 역할/전략 기본 설명
    - 여기서 반환하는 건 user_prompt에 들어갈 텍스트
    """
    return (
        f"아래 민원에 대해 답변을 작성하되, 다음 전략 프로필을 따르세요.\n\n"
        f"[전략 이름]\n{arm.name}\n\n"
        f"[전략 설명]\n{arm.description}\n\n"
        f"[민원 내용]\n{complaint_text}\n\n"
        f"[추가 지침]\n"
        f"- 관련 법령/정책과의 일치성을 확인하고, 처리 가능/불가능을 명확히 구분하세요.\n"
        f"- 민원인이 이해하기 쉬운 구조로 답변하고, 필요한 경우 대안/절차를 제시하세요.\n"
        f"- 과도한 약속이나 확정적인 표현은 피하고, '담당 부서의 최종 판단'이 필요함을 밝혀주세요.\n"
    )

# 4. Verifier: LLM 기반 점수 → reward


@dataclass
class VerificationScores:
    """각 품질 지표 스코어 (0~1)."""
    resolution_likelihood: float    # 실제로 해결/처리가 될 가능성
    policy_legal_alignment: float   # 정책/법령·과거 답변과의 일치도
    explanation_clarity: float      # 구조/논리/명료성
    empathy_tone: float             # 민원인 친화적 톤, 갈등 완화
    risk_safety: float              # 기관 입장에서의 안전성 (높을수록 안전)


class LlamaVerifier:
    """
    질문+답변을 평가하고 JSON 형태의 점수를 받는 verifier.
    """
    def __init__(self, pipe):
        self.pipe = pipe
        self.tokenizer = pipe.tokenizer

    def _call_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipe(
            messages,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=False,
            temperature=0.0,
        )

        out = outputs[0]["generated_text"]
        if isinstance(out, list):
            text = ""
            for m in reversed(out):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    text = m.get("content", "")
                    break
            if not text:
                text = str(out)
        else:
            text = str(out)

        # 첫 번째 '{' 이후를 JSON으로 가정
        try:
            start = text.find("{")
            if start == -1:
                raise ValueError("No JSON object found in output")
            json_str = text[start:]
            data = json.loads(json_str)
            return data
        except Exception as e:
            print("[Verifier] JSON parse error:", e)
            print("[Verifier] RAW OUTPUT:", text[:300])
            # 실패 시 모두 0.0
            return {
                "resolution_likelihood": 0.0,
                "policy_legal_alignment": 0.0,
                "explanation_clarity": 0.0,
                "empathy_tone": 0.0,
                "risk_safety": 0.0,
            }

    def evaluate(
        self,
        complaint_text: str,
        answer_text: str,
        extra_context: str = "",
    ) -> VerificationScores:
        system_prompt = (
            "You are an expert evaluator for administrative complaint answers.\n"
            "You must rate the quality of the ANSWER on several criteria between 0.0 and 1.0.\n"
            "Respond ONLY with a single JSON object. No explanations, no prose.\n"
            "Keys: resolution_likelihood, policy_legal_alignment, "
            "explanation_clarity, empathy_tone, risk_safety."
        )

        user_prompt = f"""
[COMPLAINT]
{complaint_text}

[ANSWER]
{answer_text}

[CONTEXT]
{extra_context}

Return JSON like:
{{
  "resolution_likelihood": 0.85,
  "policy_legal_alignment": 0.8,
  "explanation_clarity": 0.9,
  "empathy_tone": 0.7,
  "risk_safety": 0.95
}}
        """.strip()

        data = self._call_json(system_prompt, user_prompt)

        def to_float(key: str) -> float:
            v = float(data.get(key, 0.0))
            return max(0.0, min(1.0, v))

        return VerificationScores(
            resolution_likelihood=to_float("resolution_likelihood"),
            policy_legal_alignment=to_float("policy_legal_alignment"),
            explanation_clarity=to_float("explanation_clarity"),
            empathy_tone=to_float("empathy_tone"),
            risk_safety=to_float("risk_safety"),
        )


def aggregate_reward(
    scores: VerificationScores,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    reward 구조:
      r = w1 * 해결 가능성
        + w2 * 정책/법률 일치
        + w3 * 명료성
        + w4 * 공감
        + w5 * 안전성
    """
    if weights is None:
        weights = {
            "resolution_likelihood": 0.35,
            "policy_legal_alignment": 0.25,
            "explanation_clarity": 0.20,
            "empathy_tone": 0.10,
            "risk_safety": 0.10,
        }

    s = (
        scores.resolution_likelihood * weights["resolution_likelihood"]
        + scores.policy_legal_alignment * weights["policy_legal_alignment"]
        + scores.explanation_clarity * weights["explanation_clarity"]
        + scores.empathy_tone * weights["empathy_tone"]
        + scores.risk_safety * weights["risk_safety"]
    )

    return float(max(0.0, min(1.0, s)))

# 5. 전체 엔진: bandit → generator → verifier → bandit update

class PromptBanditEngine:
    """
    - bandit이 프롬프트 전략 선택
    - LLM이 답변 생성
    - verifier가 품질 측정 → reward 산출
    - bandit 업데이트
    """
    def __init__(
        self,
        arms: List[PromptArmConfig],
        base_instruction: str,
        llm: LlamaChatGenerator,
        verifier: LlamaVerifier,
    ):
        self.bandit = LogisticPromptBandit(arms=arms)
        self.base_instruction = base_instruction
        self.llm = llm
        self.verifier = verifier

    def step(
        self,
        complaint_text: str,
        reward_weights: Optional[Dict[str, float]] = None,
        extra_context: str = "",
    ) -> Dict[str, Any]:
        """
        민원 1건에 대해:
          1) bandit으로 전략 arm 선택
          2) 해당 전략으로 user_prompt 구성 + LLM 답변
          3) verifier로 평가 + reward
          4) bandit update
          5) 결과/로그 반환
        """
        # 1. arm 선택
        arm_idx = self.bandit.select_arm()
        arm = self.bandit.arms[arm_idx]

        # 2. prompt 구성 + LLM 답변
        user_prompt = build_prompt_from_arm(
            base_instruction=self.base_instruction,
            complaint_text=complaint_text,
            arm=arm,
        )
        answer = self.llm.generate(
            system_prompt=self.base_instruction,
            user_prompt=user_prompt,
        )

        # 3. verifier 평가
        scores = self.verifier.evaluate(
            complaint_text=complaint_text,
            answer_text=answer,
            extra_context=extra_context,
        )
        reward = aggregate_reward(scores, weights=reward_weights)

        # 4. bandit 업데이트
        self.bandit.update(arm_idx=arm_idx, reward=reward)

        # 5. 현재 전략 방향성 요약
        strategy_view = self.bandit.explain_current_strategy()

        return {
            "chosen_arm_idx": arm_idx,
            "chosen_arm_name": arm.name,
            "chosen_arm_description": arm.description,
            "user_prompt": user_prompt,
            "answer": answer,
            "verification_scores": scores,
            "reward": reward,
            "strategy_view": strategy_view,
        }


# 6. to be decided

def build_default_arms() -> List[PromptArmConfig]:
    """
      [법률/정책 중시 정도, 해결책 제안 정도, 공감/톤, 단호함(제한 강조), 근거 제시 강조]
    """
    return [
        PromptArmConfig(
            name="법률-정책 최우선",
            description="법령, 조례, 내부지침과의 일치성을 최우선으로 하고 책임 범위를 분명히 하는 보수적 답변.",
            feature_vector=np.array([0.9, 0.6, 0.4, 0.8, 0.7]),
        ),
        PromptArmConfig(
            name="민원인 공감형",
            description="민원인의 감정과 상황을 충분히 공감하고, 이해하기 쉬운 언어로 절차와 한계를 설명하는 답변.",
            feature_vector=np.array([0.6, 0.7, 0.9, 0.3, 0.5]),
        ),
        PromptArmConfig(
            name="해결책 제안형",
            description="현실적으로 가능한 대안과 행동 옵션을 최대한 많이 제시하는 해결 중심 답변.",
            feature_vector=np.array([0.7, 0.9, 0.7, 0.4, 0.8]),
        ),
    ]


def main():
    arms = build_default_arms()
    base_instruction = (
        "당신은 한국 공공기관의 민원 처리 보조 AI입니다. "
        "사실과 법령에 기반해 답변하고, 정책과 조례를 임의로 확대 해석하지 마세요. "
        "항상 처리 가능 범위와 한계를 분명히 설명하고, "
        "답변 마지막에는 '실제 처리 여부는 담당 부서의 최종 판단에 따릅니다.'라고 명시하세요."
    )

    generator = LlamaChatGenerator(llama_pipe)
    verifier = LlamaVerifier(llama_pipe)

    engine = PromptBanditEngine(
        arms=arms,
        base_instruction=base_instruction,
        llm=generator,
        verifier=verifier,
    )

    sample_complaints = [
        "우리 동네 공원 옆 도로에서 밤마다 오토바이 소음이 심해서 잠을 잘 수 없습니다. "
        "어디에 민원을 넣어야 하고, 구청에서 어떤 조치를 할 수 있는지 알고 싶습니다.",

        "아파트 단지 앞 불법주차 차량 때문에 시야가 가려져 사고 위험이 큽니다. "
        "단속 기준과 신고 방법, 그리고 실제로 조치가 이루어지는 절차를 알고 싶습니다.",
    ]

    for t, c in enumerate(sample_complaints, start=1):
        print("\n" + "=" * 80)
        print(f"[라운드 {t}] 민원:")
        print(c)

        result = engine.step(
            complaint_text=c,
            reward_weights=None,   # 가중치 조절 필요
            extra_context="",      # 법령 RAG context?
        )

        print("\n[선택된 프롬프트 전략]")
        print(f"- 이름: {result['chosen_arm_name']}")
        print(f"- 설명: {result['chosen_arm_description']}")

        print("\n[생성된 답변 (앞부분만)]")
        print(result["answer"][:600])

        scores = result["verification_scores"]
        print("\n[Verifier Scores]")
        print(f"- resolution_likelihood:  {scores.resolution_likelihood:.3f}")
        print(f"- policy_legal_alignment: {scores.policy_legal_alignment:.3f}")
        print(f"- explanation_clarity:    {scores.explanation_clarity:.3f}")
        print(f"- empathy_tone:           {scores.empathy_tone:.3f}")
        print(f"- risk_safety:            {scores.risk_safety:.3f}")
        print(f"=> aggregated reward:     {result['reward']:.3f}")

        print("\n[현재 전략 방향성 요약]")
        for line in result["strategy_view"]:
            print(" ", line)


if __name__ == "__main__":
    main()