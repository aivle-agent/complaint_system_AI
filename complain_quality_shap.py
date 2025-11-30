"""
민원 질문/답변 데이터를 기반으로:
1) Meta-Llama-3-8B-Instruct로 질문/답변 퀄리티 변수 스코어링
2) qa_similarity, topic_match + q_* 변수들로 답변 퀄리티 예측 모델 학습 (RandomForestRegressor)
3) SHAP으로 각 질문 변수(q_*)의 기여도(global + local) 계산
4) 각 민원별로 SHAP + 점수를 기반으로 LLM이 교정 가이드라인 생성
5) 교정 가이드를 반영해서 질문 리라이팅
"""

import json
import numpy as np
import torch
import transformers
from typing import List, Dict, Any

# ML & SHAP
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap

# 0. LLaMA-3 파이프라인 로딩

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

pipe = transformers.pipeline(
    "text-generation",
    model=MODEL_ID,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    max_new_tokens=512,
)
tokenizer = pipe.tokenizer


# 1. 공통 LLM 호출 유틸

def call_llm_json(messages: List[Dict[str, str]], temperature: float = 0.2) -> Dict[str, Any] | None:
    """
    messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
    LLM에게 JSON만 출력하도록 요청하고, 마지막 { ... } 부분을 파싱해서 dict로 반환.
    실패 시 None.
    """
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    out = pipe(
        messages,
        eos_token_id=terminators,
        do_sample=False,
        temperature=temperature,
    )[0]["generated_text"]

    try:
        json_str_start = out.rfind("{")
        if json_str_start == -1:
            raise ValueError("No JSON object found in LLM output.")
        json_str = out[json_str_start:]
        return json.loads(json_str)
    except Exception as e:
        print("[WARN] JSON parse error:", e)
        print("[RAW OUTPUT]", out[:500], "...")
        return None

# 2. 질문 퀄리티 스코어링

def score_question_quality(question_text: str, topic: str | None = None) -> Dict[str, float]:
    """
    질문 퀄리티 변수 C_t:
    - clarity: 시간/장소/대상/피해/요구 slot 충실도 (0~1)
    - specific: 숫자/기간/횟수 등 구체성 (0~1)
    - fact_ratio: 사실 vs 감정 비율 (0~1)
    - noise: 잡음/불필요 정보 비중 (0~1, 높을수록 나쁨)
    - focus: 쟁점 집중도 (1.0 = 한 가지 이슈, 0.0 = 여러 이슈 뒤섞임)
    - law_situation_sim: 전형적인 법/행정 처리 케이스와 의미적으로 유사한 정도 (0~1)
    """
    system_prompt = """You are an expert policy maker and law expert that rates the QUALITY of a citizen complaint QUESTION.
You must respond ONLY in strict JSON with floating-point scores between 0 and 1.

Definitions:
- clarity: how clearly who/when/where/what happened/what is requested are specified.
- specific: how concrete the description is (numbers, dates, durations, counts).
- fact_ratio: proportion of factual descriptions vs emotions/insults.
- noise: proportion of irrelevant or off-topic content (higher = worse).
- focus: whether the complaint is about a single main issue (1.0) or many mixed issues (0.0).
- law_situation_sim: how much this situation resembles typical legal/administrative cases,
  NOT whether legal articles are explicitly cited.
"""

    user_content = f"""
[TOPIC]: {topic}
[QUESTION]:
{question_text}

Return ONLY JSON like:
{{
  "clarity": 0.0,
  "specific": 0.0,
  "fact_ratio": 0.0,
  "noise": 0.0,
  "focus": 0.0,
  "law_situation_sim": 0.0
}}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    res = call_llm_json(messages)
    if res is None:
        res = {
            "clarity": 0.0,
            "specific": 0.0,
            "fact_ratio": 0.0,
            "noise": 1.0,
            "focus": 0.0,
            "law_situation_sim": 0.0,
        }
    # float 캐스팅
    return {k: float(v) for k, v in res.items()}

# 3. 답변 퀄리티 스코어링

def score_answer_quality(question_text: str, answer_text: str, law_context: str = "") -> Dict[str, float]:
    """
    답변 퀄리티 R_t:
    - law_align: 법/정책 정합성
    - proc_clarity: 절차 안내 명확도
    - helpful: 실질적 도움 정도
    - tone: 톤/공감/갈등 완화
    - responsible: 책임질 수 있는 표현 정도
    """
    system_prompt = """You are an expert policy maker and law expert that rates the QUALITY of an administrative ANSWER
to a citizen complaint. Respond ONLY in JSON with scores 0.0–1.0.

- law_align: answer is consistent with the legal/policy context (not obviously wrong/ can be found in legal database).
- proc_clarity: clear procedural guidance (what future action is to be taken, for how long is it to be taken, etc).
- helpful: how much this answer helps the citizen understand what to do next.
- tone: politeness, empathy, de-escalation.
- responsible: institutionally safe and responsible wording (no over-promising, 
  proper limits and disclaimers).
"""

    user_content = f"""
[QUESTION]:
{question_text}

[ANSWER]:
{answer_text}

[LEGAL/POLICY CONTEXT SNIPPET]:
{law_context[:1500]}

Return ONLY JSON like:
{{
  "law_align": 0.0,
  "proc_clarity": 0.0,
  "helpful": 0.0,
  "tone": 0.0,
  "responsible": 0.0
}}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    res = call_llm_json(messages)
    if res is None:
        res = {
            "law_align": 0.0,
            "proc_clarity": 0.0,
            "helpful": 0.0,
            "tone": 0.0,
            "responsible": 0.0,
        }
    return {k: float(v) for k, v in res.items()}

# 4. 질문-답변 유사도 / 주제 일치도 

def qa_similarity(q: str, a: str) -> float:
    """
    - 공통 단어 비율 기반 Jaccard 유사도 (0~1)
    to be sentence embedding-based similarity later.
    """
    q_tokens = set(q.split())
    a_tokens = set(a.split())
    if not q_tokens or not a_tokens:
        return 0.0
    inter = len(q_tokens & a_tokens)
    union = len(q_tokens | a_tokens)
    return inter / union


def topic_match_score(topic_true: str | None, topic_pred: str | None) -> float:
    """
    토픽 일치도: 완전 일치 1.0, 아니면 0.0
    topic match 모델 붙이면 수정 예정.
    """
    if topic_true is None or topic_pred is None:
        return 0.0
    return 1.0 if topic_true == topic_pred else 0.0

# 5. QA 리스트에서 feature 추출

def extract_features_from_qa_pairs(qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    qa_pairs: [{"question": str, "answer": str, "topic": Optional[str]}, ...]
    를 넣으면, 각 QA에 대해 질문/답변 퀄리티 + 유사도 feature를 계산해서 리스트로 반환.
    """
    feature_rows: List[Dict[str, Any]] = []

    for i, item in enumerate(qa_pairs):
        q = item["question"]
        a = item["answer"]
        topic = item.get("topic", None)

        print(f"[INFO] [{i+1}/{len(qa_pairs)}] Scoring question/answer...")

        # 1) 질문 퀄리티
        q_scores = score_question_quality(q, topic=topic)

        # 2) 답변 퀄리티
        a_scores = score_answer_quality(q, a, law_context="")

        # 3) QA 유사도 (고정 변수)
        sim_qa = qa_similarity(q, a)

        # 4) 토픽 일치도 
        topic_match = 1.0  # 가변 변수, 수정 필요

        # 5) 답변 종합 퀄리티 (타깃 y 후보)
        answer_quality_total = float(np.mean(list(a_scores.values())))

        row: Dict[str, Any] = {
            "question": q,
            "answer": a,
            "topic": topic,
            "qa_similarity": float(sim_qa),
            "topic_match": float(topic_match),
            "answer_quality_total": answer_quality_total,
        }

        # 질문 변수 prefix: q_
        for k, v in q_scores.items():
            row[f"q_{k}"] = float(v)

        # 답변 변수 prefix: a_
        for k, v in a_scores.items():
            row[f"a_{k}"] = float(v)

        feature_rows.append(row)

    return feature_rows

# 6. SHAP: 모델 학습 + 전역 중요도

def train_model_and_compute_shap(feature_rows: List[Dict[str, Any]]):
    """
    feature_rows에서:
    - y = answer_quality_total
    - X = ["qa_similarity", "topic_match"] + q_* 변수들
    RandomForestRegressor 학습 후 SHAP(TreeExplainer)로 전역/로컬 기여도 계산.
    """
    df = pd.DataFrame(feature_rows)

    y = df["answer_quality_total"].values

    feature_cols = ["qa_similarity", "topic_match"] + [
        c for c in df.columns if c.startswith("q_")
    ]
    X = df[feature_cols].values

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    base_value = explainer.expected_value

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    global_importance = {
        col: float(val) for col, val in zip(feature_cols, mean_abs_shap)
    }

    for col, val in sorted(global_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {col}: {val:.4f}")

    return {
        "model": model,
        "explainer": explainer,
        "feature_cols": feature_cols,
        "shap_values": shap_values,
        "base_value": float(base_value),
        "global_importance": global_importance,
    }

def get_local_question_shap(
    feature_row: Dict[str, Any],
    explainer,
    feature_cols: List[str],
) -> Dict[str, float]:
    """
    민원에 대해 SHAP 값 구해서 q_ 변수들만 dict로 반환.
    """
    x = np.array([[feature_row[col] for col in feature_cols]])
    shap_vals = explainer.shap_values(x)[0]  

    local_all = {col: float(val) for col, val in zip(feature_cols, shap_vals)}
    local_q_shap = {k: v for k, v in local_all.items() if k.startswith("q_")}
    return local_q_shap

# 7. SHAP + 점수 기반 교정 가이드 + 리라이팅

def build_guideline_with_shap(
    question: str,
    feature_row: Dict[str, Any],
    local_q_shap: Dict[str, float],
) -> str:
    """
    개별 민원에 대해:
    - feature_row: q_* 점수들
    - local_q_shap: q_*마다 SHAP 값
    샘플에서 해당 요소가 답변 퀄리티를 얼마나 올리거나 내렸는지 기준으로 LLM에게 교정 가이드를 생성하게 함.
    """
    items = []
    for var, shap_val in local_q_shap.items():
        score = float(feature_row.get(var, 0.0))
        items.append({
            "name": var,
            "score": score,
            "shap_value": shap_val,
        })

    # 영향력(|shap|) 큰 순으로 정렬
    items_sorted = sorted(items, key=lambda x: abs(x["shap_value"]), reverse=True)

    info_lines = "\n".join(
        f"{it['name']}: score={it['score']:.2f}, shap={it['shap_value']:.3f}"
        for it in items_sorted
    )

    system_prompt = """You are a writing assistant for citizen complaints.
For this specific complaint, you have per-factor SHAP values indicating
how each QUESTION quality factor affects the predicted ANSWER quality.

Interpretation:
- score: current quality of that factor (0~1).
- shap: local contribution to answer quality. 
  Negative shap = this factor is currently making the answer quality worse.
  Larger |shap| = stronger impact.

Write your advice in Korean.
Do NOT rewrite the question yet; only give bullet-point editing guidelines.

Guideline rules:
- Prioritize factors with LOW score and strongly NEGATIVE shap first.
- Also consider factors with high |shap| even if score is mid-level.
- Keep the citizen's original intent.
- Suggest specific, concrete edits (what information to add, how to clarify, etc.).
"""

    user_content = f"""
[원본 민원]
{question}

[질문 품질 점수 및 SHAP 기여도]
{info_lines}

위 정보를 바탕으로, 3~6개의 구체적인 교정 가이드를 bullet point로 작성하세요.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    out = pipe(
        messages,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
        max_new_tokens=256,
    )[0]["generated_text"]

    return out


def rewrite_question_with_guideline(question: str, guideline: str) -> str:
    """
    교정 가이드라인을 적용해 새 질문(Q_refined)을 생성.
    """
    system_prompt = """You are a rewriting assistant for citizen complaints.
Rewrite the question in Korean:
- Keep track of the original intensions, the rewriiten verison should match it.
- Apply the given editing guidelines.
- Improve clarity, concreteness, and legal/administrative match.
Return ONLY the rewritten question, no explanation."""

    user_content = f"""
[원본 민원]
{question}

[교정 가이드]
{guideline}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    out = pipe(
        messages,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
        max_new_tokens=256,
    )[0]["generated_text"]

    return out

# main 실행 코드
# # 8. 예시 실행 (메인)

# if __name__ == "__main__":
#     qa_pairs = [
        
#     ]

#     # 1) QA → feature 추출
#     features = extract_features_from_qa_pairs(qa_pairs)

#     # 2) SHAP 기반 모델 학습
#     shap_bundle = train_model_and_compute_shap(features)
#     explainer = shap_bundle["explainer"]
#     feature_cols = shap_bundle["feature_cols"]

#     # 3) 첫 민원에 대해 로컬 SHAP + 교정 가이드 + 리라이팅
#     first = features[0]
#     local_q_shap = get_local_question_shap(first, explainer, feature_cols)

#     guideline = build_guideline_with_shap(
#         question=first["question"],
#         feature_row=first,
#         local_q_shap=local_q_shap,
#     )
#     new_question = rewrite_question_with_guideline(first["question"], guideline)

#     print("\n=== [원본 민원] ===")
#     print(first["question"])
#     print("\n=== [교정 가이드] ===")
#     print(guideline)
#     print("\n=== [교정된 민원] ===")
#     print(new_question)
