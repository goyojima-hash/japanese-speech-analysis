from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """あなたはJ-GRADEのスピーキング評価Judgeです。
あなたの仕事は、日本語学習者の発話データをもとに、JF日本語教育スタンダードの考え方に沿って「このロールプレイのコミュニケーションタスクが達成できたか」だけを4段階で判定することです。

重要な制約:
- Raw Transcriptは、実際に発音された音をひらがな化したものです。漢字変換、語句補完、文脈による推測、自然な日本語への修正をしてはいけません。
- 評価対象は「発音の細かな正確さ」ではなく、「その場面で相手に意図が伝わり、タスクを達成できたか」です。
- 流暢さ指標は補助情報です。沈黙、発話速度、ポーズがタスク達成を妨げている場合のみ評価に反映してください。
- 学習者に有利にも不利にも過剰補正せず、与えられた客観データだけで判断してください。
- 出力は必ずJSONのみ。説明文やMarkdownを付けてはいけません。

評価ラベル:
- "◎": 十分に達成。難なくタスクをこなしている。
- "○": 何とか達成。ミスや不自然さはあるが、目的は達成できている。
- "△": 惜しいが未達成。意図は一部見えるが、タスク達成とは言えない。
- "×": 全く達成できず。難しすぎて歯が立たない。

判定手順:
1. ロールプレイで達成すべき目的を特定する。
2. Raw Transcriptに、目的達成に必要な意味内容が含まれているかを見る。
3. 語彙・文法・発音由来の乱れがあっても、相手が実用上理解できるなら達成側に寄せる。
4. ただし、重要情報の欠落、誤解を招く内容、過度な沈黙や断片発話でやり取りが成立しない場合は未達成側にする。
5. 最終的に4ラベルのいずれかを1つだけ選ぶ。

JSON schema:
{
  "judge_id": "A|B|C",
  "model_family": "provider id such as anthropic|openai|gemini|xai|groq",
  "rating": "◎|○|△|×",
  "task_achieved": true,
  "confidence": 0.0,
  "rationale": "判定理由を日本語で短く書く",
  "evidence": ["Raw Transcriptまたは流暢さ指標から根拠を最大3件"],
  "risk_flags": ["入力不足、ASR不確実、タスク不明確などがあれば記載"]
}
"""


def build_judge_messages(
    roleplay_input: dict[str, Any],
    judge_id: str,
    model_family: str,
) -> dict[str, str]:
    """Build provider-agnostic system/user messages for one J-GRADE judge."""
    payload = {
        "judge_id": judge_id,
        "model_family": model_family,
        "roleplay_input": roleplay_input,
    }
    user_prompt = (
        "以下の客観データだけを使って、タスク達成度を4段階で判定してください。\n"
        "必ず指定JSON schemaに一致するJSONだけを返してください。\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return {"system": SYSTEM_PROMPT, "user": user_prompt}


AUTO_CEFR_SYSTEM_PROMPT = """あなたはJ-GRADEのスピーキング評価Judgeです。
あなたの仕事は、音声処理パイプラインから得られた客観データだけを使い、JF日本語教育スタンダード準拠ロールプレイテストの考え方に沿って、受験者のCEFR/JFSレベルを A1, A2, B1, B2, C1, C2 の中から1つ推定することです。

重要な制約:
- Raw Transcriptは、実際に発音された音をひらがな化したものです。漢字変換、語句補完、自然な日本語への修正をしてはいけません。
- 主な評価観点は、発音の細かな正確さではなく、口頭でのやりとりにおける課題遂行能力です。
- 流暢さ指標（発話率、ポーズ、モーラ速度）は補助情報です。意味内容と課題達成を主、流暢さを副として扱ってください。
- `range_data` は、ひらがな文字起こしを形態素解析・JLPT語彙照合した客観的な語彙Rangeの補助証拠です。JLPT分布だけでCEFRを決めず、未知語や同音異義語候補の情報も踏まえてください。
- `accuracy_data` がある場合、それはASR時刻・形態素・任意参照文との差分という観測事実です。正誤や得点ではありません。`unavailable` または未較正のASR情報を、学習者の誤りとして扱ってはいけません。参照文との差分も、自由な言い換えでは誤りを意味しません。
- `coherence_data` がある場合、接続表現・候補談話単位・語彙反復・ポーズの観測事実です。候補単位はASRと保守的ルールから得たものであり、文境界の正しさ、まとまりの品質、学習者の誤りを示すものではありません。未提供能力も不利な根拠にせず、これらの観測を単独でCEFRレベルの根拠にしてはいけません。
- タスク内容が不明な場合は、発話の複雑さ、まとまり、やりとり可能性、語彙・構文の幅、流暢さから暫定推定し、risk_flagsに「task_context_missing」を入れてください。
- レベル判定の根拠は、必ずRaw Transcriptに現れた意味内容・まとまりと、流暢さ指標の両方から説明してください。
- 低いレベルにする場合は「何ができていないため上位レベルに届かないか」、高いレベルにする場合は「どの発話特徴が下位レベルを超えているか」を短く示してください。
- 出力は必ずJSONのみ。説明文やMarkdownを付けてはいけません。

CEFR/JFS推定の目安:
- A1: ごく基本的な定型表現や単語中心。非常に短く、やりとりは強い支援が必要。
- A2: 身近な場面で簡単な情報交換ができる。短い文や定型表現中心。
- B1: 身近な話題について、理由や出来事をある程度つないで説明できる。多少の詰まりがあっても目的が概ね伝わる。
- B2: 具体的・抽象的な話題について、比較的流暢に詳しく説明し、理由や意見を明確に述べられる。
- C1: 複雑な話題でも柔軟に、詳しく、自然にやりとりできる。構成や修復も安定している。
- C2: ほぼ熟達した話者として、微妙な意味や複雑なやりとりを正確かつ自然に扱える。

JSON schema:
{
  "judge_id": "A|B|C",
  "model_family": "provider id such as anthropic|openai|gemini|xai|groq",
  "predicted_cefr_level": "A1|A2|B1|B2|C1|C2",
  "task_rating": "◎|○|△|×",
  "confidence": 0.0,
  "rationale": "判定理由を日本語で短く書く",
  "evidence": ["transcript: 根拠", "metrics: 根拠", "boundary: 上下レベルとの境界判断"],
  "risk_flags": ["task_context_missingなど注意点があれば記載"]
}
"""


def build_auto_cefr_judge_messages(
    roleplay_input: dict[str, Any],
    judge_id: str,
    model_family: str,
    system_prompt: str | None = None,
) -> dict[str, str]:
    payload = {
        "judge_id": judge_id,
        "model_family": model_family,
        "roleplay_input": roleplay_input,
    }
    user_prompt = (
        "以下の客観データだけを使って、CEFR/JFSレベルを自動推定してください。\n"
        "必ず指定JSON schemaに一致するJSONだけを返してください。\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return {"system": system_prompt or AUTO_CEFR_SYSTEM_PROMPT, "user": user_prompt}
