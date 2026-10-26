import os
from typing import Any

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

# Bedrock 没有全球统一默认区域；us-west-2 提供较全面的模型。
_DEFAULT_REGION = "us-west-2"
_BEDROCK_CLASS = None


def _bedrock_class():
    """按需导入可选依赖 ``[bedrock]`` 提供的 langchain-aws，
    并返回标准化字符串输出的 ChatBedrockConverse 子类。
    只有使用 Bedrock 时才导入，因此其他部分不需要安装
    langchain-aws 或 boto3；首次加载后缓存结果。"""
    global _BEDROCK_CLASS
    if _BEDROCK_CLASS is not None:
        return _BEDROCK_CLASS

    try:
        from langchain_aws import ChatBedrockConverse
    except ImportError as exc:
        raise ImportError(
            "AWS Bedrock support requires the optional 'langchain-aws' dependency. "
            'Install it with: pip install "tradingagents[bedrock]"'
        ) from exc

    class NormalizedChatBedrockConverse(ChatBedrockConverse):
        """将响应内容标准化为字符串的 ChatBedrockConverse。"""

        def invoke(self, input, config=None, **kwargs):
            return normalize_content(super().invoke(input, config, **kwargs))

    _BEDROCK_CLASS = NormalizedChatBedrockConverse
    return _BEDROCK_CLASS


class BedrockClient(BaseLLMClient):
    """通过 Converse API（langchain-aws）调用 Amazon Bedrock 的客户端。

    既支持 ``AWS_BEARER_TOKEN_BEDROCK`` 提供的 Bedrock API Key
    （不需要 AWS Access Key），也支持标准 AWS 凭证链：环境变量、
    ``~/.aws/credentials`` 或 IAM Role，并可使用 ``AWS_PROFILE``。
    两种方式都需配置 ``AWS_REGION`` 或 ``AWS_DEFAULT_REGION``，
    因为 Bearer Token 不包含区域。
    模型名称为 Bedrock 模型 ID 或跨地区推理配置 ID，
    例如 ``us.anthropic.claude-opus-4-8-v1:0``。"""

    def get_llm(self) -> Any:
        """返回已配置的 ChatBedrockConverse 实例。"""
        self.warn_if_unknown_model()
        chat_cls = _bedrock_class()

        region = (
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or _DEFAULT_REGION
        )
        llm_kwargs = {"model": self.model, "region_name": region}
        # Bedrock API Key 不依赖 AWS Access Key。
        # 将它作为 api_key 传入后，langchain-aws 优先使用 Bearer 鉴权，
        # 避免已有 AWS_PROFILE 或 SigV4 凭证覆盖它（#1103）。
        bearer_token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if bearer_token:
            llm_kwargs["api_key"] = bearer_token
        for key in ("temperature", "max_tokens", "max_retries", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return chat_cls(**llm_kwargs)

    def validate_model(self) -> bool:
        """验证 Bedrock 模型名称（接受任意模型 ID）。"""
        return validate_model("bedrock", self.model)
