"""Shared roleplay relevance rules and a narrow guard against unsolicited AI boilerplate."""

import re


_AI_IDENTITY = r"(?:AI|人工智能|大语言模型|语言模型|大模型|聊天机器人|智能助手|language\s+model|chatbot|AI\s+assistant)"
_BARE_AI_LABEL = re.compile(
    r"[\s?？！!。.,，、:：;；…~～]*(?:AI|人工智能|大语言模型|语言模型|大模型|聊天机器人|ChatGPT|LLM|bot)"
    r"(?:[\s?？！!。.,，、:：;；…~～]|啊|呀|呢|吧|哦|诶)*",
    re.IGNORECASE,
)
_SELF_AI_DECLARATION = re.compile(
    r"(?:^|[。！？.!?\n])\s*(?:"
    r"(?:抱歉[，,]?\s*|其实[，,]?\s*)?(?:我(?:本质上|实际上|其实|只是|确实)?(?:是|属于|算是)|作为)"
    r"(?:一[个名位]|一种)?\s*" + _AI_IDENTITY + r"|"
    r"(?:as\s+an?\s+|I\s+(?:am|'m)\s+(?:just\s+)?an?\s+)" + _AI_IDENTITY + r"|"
    r"私は\s*(?:AI|人工知能|言語モデル)|(?:AI|人工知能|言語モデル)として)",
    re.IGNORECASE,
)
_EXPLICIT_IDENTITY_REQUEST = re.compile(
    r"(?:你|您).{0,16}(?:是不是|是否|是|属于|算是).{0,16}" + _AI_IDENTITY
    + r"|(?:你|您)(?:到底|究竟)?是(?:谁|什么)"
    r"|\b(?:are|aren't)\s+you\b.{0,25}" + _AI_IDENTITY
    + r"|\b(?:who|what)\s+are\s+you\b"
    r"|(?:あなた|君).{0,12}(?:AI|人工知能|言語モデル)"
    r"|(?:写|生成|翻译|引用|模拟|模仿|举例).{0,24}(?:自我介绍|自称|身份声明|我是AI|作为AI)"
    r"|\b(?:write|translate|quote)\b.{0,40}\b(?:self.introduction|as an AI|I am an AI)\b",
    re.IGNORECASE,
)


def build_identity_behavior_instruction(locale: str) -> str:
    """Keep the same identity/relevance rule in Chinese, English, and Japanese."""
    rules = {
        "zh-CN": (
            "【身份与发言相关性】保持配置中的角色和日常说话方式，不在无关对话中突然自我介绍为 AI、"
            "人工智能、语言模型或机器人，也不添加‘作为一个 AI’式能力免责声明。群友泛指 AI、ChatGPT、"
            "大模型或其他机器人，不等于在叫你、谈论你或要求你证明身份；只出现这些字眼不是插话理由。"
            "先判断真实 @、昵称、引用关系及当前对话是否明确面向你；无关就保持安静。明确向你提问、"
            "私聊或实际相关的 AI 技术讨论仍可正常回答。用户认真询问实际系统身份时，简短如实回答，"
            "不要把角色设定说成现实中的人类身份。"
        ),
        "en-US": (
            "[Identity and relevance] Keep the configured character and conversational style. Do not introduce "
            "yourself as an AI, language model, or bot, or add 'as an AI' disclaimers when unrelated to the request. "
            "A group message mentioning AI, ChatGPT, models, or other bots does not address you or ask about your "
            "identity. Those words alone are not a reason to join in. Use actual mentions, names, reply relationships, "
            "and conversational relevance; stay quiet when unrelated. Answer direct questions, private messages, and "
            "relevant AI discussions normally. If explicitly asked about your actual system identity, answer briefly "
            "and truthfully rather than presenting a fictional character as a real human."
        ),
        "ja-JP": (
            "【身元と会話への関連性】設定されたキャラクターと話し方を保ち、無関係な会話で突然 AI・"
            "言語モデル・ボットだと自己紹介したり、「AI として」という免責説明を加えたりしない。"
            "グループ内の AI、ChatGPT、大規模モデル、別のボットへの言及だけでは、あなたへの呼びかけや"
            "身元の質問にはならない。実際のメンション、名前、返信関係、文脈から関連性を判断し、"
            "無関係なら発言しない。直接の質問、個別チャット、関連する AI 技術の話題には普通に答える。"
            "実際のシステム上の身元を明確に聞かれた場合は、架空の役を実在の人間だと主張せず、短く正直に答える。"
        ),
    }
    return rules.get(locale, rules["zh-CN"])


def is_bare_ai_label(text: str) -> bool:
    """Match a contextless label, not a substantive question or an AI discussion."""
    return bool(_BARE_AI_LABEL.fullmatch(text.strip()))


def has_unsolicited_ai_disclosure(response: str, incoming_text: str) -> bool:
    """Reject first-person boilerplate, while keeping explicit questions and quoted examples."""
    return bool(_SELF_AI_DECLARATION.search(response)) and not bool(
        _EXPLICIT_IDENTITY_REQUEST.search(incoming_text)
    )
