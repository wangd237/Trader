"""供不同数据提供商共用的 HTTP 辅助函数。"""

import requests


def get_scrubbed(url: str, *, params: dict, timeout: float, secret: str, passthrough=()):
    """调用 ``requests.get`` 并执行 ``raise_for_status``，同时避免在异常中泄露 ``secret``。

    使用查询参数鉴权的提供商会把密钥写入 URL，而 requests 的 HTTP、
    连接或超时异常可能引用完整 URL，使日志和异常栈意外包含密钥（#1324）。
    因此将 requests 异常重建为同一类型，并替换敏感密钥；
    不附带 request、response（两者可能保存原 URL）或异常链。
    这也是为什么在 except 块结束后才重新抛出异常。
    ``passthrough`` 中的状态码交给调用者自行处理。"""
    try:
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code not in passthrough:
            response.raise_for_status()
        return response
    except requests.RequestException as exc:
        error = type(exc)(str(exc).replace(secret, "***")) if secret else exc
    raise error


def vendor_reachable(url: str, timeout: float = 5.0) -> bool:
    """检测提供商是否能够响应，用于区分“确实没有数据”和“服务故障”。

    若客户端在出错时只返回空结果而不抛异常，这两类情况会难以区分。
    仅在实际结果为空时调用此检查。"""
    try:
        requests.head(url, timeout=timeout, allow_redirects=True)
        return True
    except requests.RequestException:
        return False
