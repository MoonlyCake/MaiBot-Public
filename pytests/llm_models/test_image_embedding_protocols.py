"""原生图片嵌入协议的离线回归，不导入应用或配置。"""

from pathlib import Path
from unittest import TestCase

import ast
import base64
import struct
import sys
import types


_ROOT = Path(__file__).resolve().parents[2]
_MODULE_NAME = "maibot_image_embedding_protocols_regression"
_MODULE = types.ModuleType(_MODULE_NAME)
sys.modules[_MODULE_NAME] = _MODULE
# 直接加载生产规范化函数和协议代码，隔离仅提供类型的 base_client 以及配置初始化。
_COMPAT_TREE = ast.parse((_ROOT / "src/llm_models/openai_compat.py").read_text(encoding="utf-8"))
_NORMALIZE = next(
    node for node in _COMPAT_TREE.body if isinstance(node, ast.FunctionDef) and node.name == "normalize_openai_base_url"
)
exec(compile(ast.Module(body=[_NORMALIZE], type_ignores=[]), "openai_compat.py", "exec"), _MODULE.__dict__)
_MODULE.UsageTuple = tuple[int, int, int, int, int, bool]
_PROTOCOL_PATH = _ROOT / "src/llm_models/model_client/image_embedding_protocols.py"
_PROTOCOL_TREE = ast.parse(_PROTOCOL_PATH.read_text(encoding="utf-8"))
_PROTOCOL_TREE.body = [
    node for node in _PROTOCOL_TREE.body
    if not (
        isinstance(node, ast.ImportFrom)
        and node.module in {"src.llm_models.openai_compat", "base_client"}
    )
]
exec(compile(_PROTOCOL_TREE, str(_PROTOCOL_PATH), "exec"), _MODULE.__dict__)

ARK_PROTOCOL = _MODULE.ARK_PROTOCOL
DASHSCOPE_PROTOCOL = _MODULE.DASHSCOPE_PROTOCOL
VOYAGE_PROTOCOL = _MODULE.VOYAGE_PROTOCOL
build_native_image_embedding_diagnostic_payload = _MODULE.build_native_image_embedding_diagnostic_payload
build_native_image_embedding_fingerprint = _MODULE.build_native_image_embedding_fingerprint
build_native_image_embedding_request = _MODULE.build_native_image_embedding_request
parse_native_image_embedding_response = _MODULE.parse_native_image_embedding_response
resolve_native_image_embedding_protocol = _MODULE.resolve_native_image_embedding_protocol



class TestImageEmbeddingProtocols(TestCase):
    def test_native_image_embedding_routes_only_exact_official_voyage_hosts(self) -> None:
        for hostname in ("api.voyageai.com", "ai.mongodb.com", "us.ai.mongodb.com", "eu.ai.mongodb.com"):
            self.assertEqual(resolve_native_image_embedding_protocol(f"https://{hostname}/v1/"), VOYAGE_PROTOCOL)
        for url in (
            "https://ai.mongodb.com.example.org/v1",
            "https://other.ai.mongodb.com/v1",
            "https://api.voyageai.com/v2",
            "https://ai.mongodb.com/v1/embeddings",
        ):
            self.assertIsNone(resolve_native_image_embedding_protocol(url))
        self.assertEqual(resolve_native_image_embedding_protocol("https://dashscope.aliyuncs.com/compatible-mode/v1"), DASHSCOPE_PROTOCOL)
        self.assertEqual(resolve_native_image_embedding_protocol("https://ark.cn-beijing.volces.com/api/v3"), ARK_PROTOCOL)


    def test_voyage_image_payload_endpoint_and_diagnostics(self) -> None:
        request = build_native_image_embedding_request(
            VOYAGE_PROTOCOL,
            base_url="https://us.ai.mongodb.com/v1",
            model_identifier="voyage-multimodal-3.5",
            image_bytes=b"picture",
            mime_type="image/png",
            extra_params={"input_type": "document", "headers": {"x-test": "yes"}, "query": {"trace": "1"}},
        )
        self.assertEqual(request.endpoint_url, "https://us.ai.mongodb.com/v1/multimodalembeddings")
        self.assertEqual(request.payload, {
            "model": "voyage-multimodal-3.5",
            "inputs": [{"content": [{"type": "image_base64", "image_base64": "data:image/png;base64,cGljdHVyZQ=="}]}],
            "output_encoding": "base64",
            "input_type": "document",
        })
        self.assertEqual(request.semantic_params, {"dimension": 1024, "input_type": "document"})
        self.assertEqual(request.extra_headers, {"x-test": "yes"})
        self.assertEqual(request.extra_query, {"trace": "1"})
        diagnostic = build_native_image_embedding_diagnostic_payload(request.payload)
        self.assertIn("已省略", diagnostic["inputs"][0]["content"][0]["image_base64"])
        self.assertTrue(request.payload["inputs"][0]["content"][0]["image_base64"].startswith("data:image/png;base64,"))


    def test_voyage_image_response_accepts_float_and_little_endian_base64(self) -> None:
        expected = [0.25, -0.5, 1.0]
        for embedding in (expected, base64.b64encode(struct.pack("<3f", *expected)).decode("ascii")):
            result = parse_native_image_embedding_response(
                VOYAGE_PROTOCOL,
                {"data": [{"embedding": embedding}], "usage": {"total_tokens": 203, "text_tokens": 3, "image_pixels": 1024}},
            )
            self.assertEqual(result.embedding, expected)
            self.assertEqual(result.usage, (203, 0, 203, 0, 0, False))


    def test_voyage_image_rejects_invalid_vectors_and_unsupported_dimensions(self) -> None:
        for embedding in ("not-base64!", base64.b64encode(b"abc").decode("ascii"), [float("nan")]):
            with self.assertRaises(ValueError):
                parse_native_image_embedding_response(VOYAGE_PROTOCOL, {"data": [{"embedding": embedding}]})
        for key in ("dimensions", "output_dimension", "encoding_format"):
            with self.assertRaisesRegex(ValueError, key):
                build_native_image_embedding_request(
                    VOYAGE_PROTOCOL,
                    base_url="https://ai.mongodb.com/v1",
                    model_identifier="voyage-multimodal-3.5",
                    image_bytes=b"image",
                    mime_type="image/png",
                    extra_params={key: 1024},
                )


    def test_voyage_image_fingerprint_tracks_vector_semantics_without_image_or_credentials(self) -> None:
        def fingerprint(image: bytes, input_type: str, base_url: str, api_key: str) -> str:
            request = build_native_image_embedding_request(
                VOYAGE_PROTOCOL,
                base_url=base_url,
                model_identifier="voyage-multimodal-3.5",
                image_bytes=image,
                mime_type="image/png",
                extra_params={"body": {"input_type": input_type}, "headers": {"authorization": api_key}},
            )
            return build_native_image_embedding_fingerprint(request, "voyage-multimodal-3.5")

        baseline = fingerprint(b"first", "document", "https://ai.mongodb.com/v1", "secret-one")
        self.assertEqual(fingerprint(b"second", "document", "https://api.voyageai.com/v1", "secret-two"), baseline)
        self.assertNotEqual(fingerprint(b"first", "query", "https://ai.mongodb.com/v1", "secret-one"), baseline)
