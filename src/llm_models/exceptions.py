from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class InterruptedStreamOutput:
    """用于历史展示和继续推理的流式片段，不作为可执行的工具响应。"""

    reasoning: str
    content: str
    tool_calls: List[Dict[str, str]]
    model_name: str = ""

    def display_text(self) -> str:
        """保留未完成的原始文本和工具参数，不尝试修复或执行。"""

        sections = []
        if self.reasoning:
            sections.append(f"[已接收的 reasoning 片段]\n{self.reasoning}")
        if self.content:
            sections.append(f"[已接收的正文片段]\n{self.content}")
        for call in self.tool_calls:
            sections.append(f"[未执行的工具调用片段：{call['name']}]\n{call['arguments']}")
        return "\n\n".join(sections)


# 常见Error Code Mapping (以OpenAI API为例)
error_code_mapping = {
    400: "参数不正确",
    401: "API-Key错误，认证失败，请检查/config/model_list.toml中的配置是否正确",
    402: "账号余额不足",
    403: "模型拒绝访问，可能需要实名或余额不足",
    404: "Not Found",
    413: "请求体过大，请尝试压缩图片或减少输入内容",
    429: "请求过于频繁，请稍后再试",
    500: "服务器内部故障",
    503: "服务器负载过高",
}


class NetworkConnectionError(Exception):
    """连接异常，常见于网络问题或服务器不可用"""

    def __init__(self, message: str | None = None):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message or "连接异常，请检查网络连接状态或URL是否正确"


class ImageEmbeddingUnsupportedError(Exception):
    """当前 Provider 或模型没有实现图片嵌入协议。"""


class ReqAbortException(Exception):
    """请求异常退出，常见于请求被中断或取消"""

    def __init__(self, message: str | None = None, partial_output: InterruptedStreamOutput | None = None):
        super().__init__(message)
        self.message = message
        self.partial_output = partial_output

    def __str__(self):
        return self.message or "请求因未知原因异常终止"


class RespNotOkException(Exception):
    """请求响应异常，见于请求未能成功响应（非 '200 OK'）"""

    def __init__(self, status_code: int, message: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message

    def __str__(self):
        if self.status_code in error_code_mapping:
            return error_code_mapping[self.status_code]
        elif self.message:
            return self.message
        else:
            return f"未知的异常响应代码：{self.status_code}"


class ResponseContextException(Exception):
    """携带原始响应上下文的异常基类。"""

    default_message: str = "请求失败"

    def __init__(self, ext_info: Any = None, message: str | None = None):
        super().__init__(message)
        self.ext_info = ext_info
        self.message = message

    def __str__(self):
        return self.message or self.default_message


class RespParseException(ResponseContextException):
    """响应解析错误，常见于响应格式不正确或解析方法不匹配"""

    default_message = "解析响应内容时发生未知错误，请检查是否配置了正确的解析方法"


class EmptyResponseException(ResponseContextException):
    """响应内容为空"""

    default_message = "响应内容为空，这可能是一个临时性问题"


class ModelAttemptFailed(Exception):
    """当在单个模型上的所有重试都失败后，由“执行者”函数抛出，以通知“调度器”切换模型。"""

    def __init__(self, message: str, original_exception: Exception | None = None):
        super().__init__(message)
        self.message = message
        self.original_exception = original_exception

    def __str__(self):
        return self.message


class LLMTaskTimeoutError(ModelAttemptFailed):
    """任务级 hard_timeout 触发的异常。

    `_execute_request` 为整个操作设置一次期限，所有候选模型及重试共享该预算。
    到期取消当前尝试、释放模型使用名额后上抛本异常，不再切换候选模型。
    若尚未选出模型，model_name 为空，异常只标记任务。
    """

    def __init__(self, task_name: str, model_name: str, timeout_s: float):
        model_detail = f" 模型 '{model_name}'" if model_name else ""
        super().__init__(
            f"任务 '{task_name}'{model_detail} 触发总请求硬超时 {timeout_s}s",
            original_exception=None,
        )
        self.task_name = task_name
        self.model_name = model_name
        self.timeout_s = timeout_s
