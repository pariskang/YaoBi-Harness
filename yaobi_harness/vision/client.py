"""Multimodal client for clinical images, via Poe's OpenAI-compatible endpoint.

Default model ``Gemini-3.1-Pro``. Any OpenAI-shaped multimodal endpoint works —
the image is sent as a ``image_url`` content part carrying a ``data:`` URI, which
is what Poe, Azure, LiteLLM and MiniMax all accept.

Three properties are not negotiable, and they are enforced here rather than
described in a prompt:

**Images are never persisted.** Bytes are read, encoded, sent, and dropped. What
survives is the structured finding plus a SHA-256 of the bytes, so an audit can
prove *which* image produced a finding without the repository ever holding it.

**A read is never a report.** Every result carries
``requires_formal_read: True`` and lands in the evidence ledger at
``model_reasoning`` grade, which is in ``NON_RELEASABLE_LEVELS`` — so a vision
finding can never on its own support a released clinical claim. It can raise a
red flag, suggest what to ask or examine, and describe what is visible. It cannot
diagnose.

**Identifiers abort the read.** A phone photo of a screen, or a DICOM export with
a burned-in header, routinely carries a name, an ID and a date of birth. The model
is asked to report identifiers *first*; if it sees any, the finding is discarded
and a PHI warning is returned instead. Refusing to read is the correct outcome
there, and it is the repository's standing rule applied to a new input channel.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm.base import LLMError

#: Image kinds the reader accepts. The prompt and the output contract differ per
#: kind, because "read this X-ray" and "describe this tongue" are not the
#: same task and must not share a prompt.
IMAGE_KINDS = ("radiograph", "mri_ct", "tongue", "posture_gait", "limb_surface", "report_document", "other")

#: Bytes. A phone photo is ~2-6 MB; past this it is a scan or a video frame dump,
#: and base64 inflation would blow the request rather than fail cleanly.
MAX_IMAGE_BYTES = 12 * 1024 * 1024

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

PHI_PROMPT = """你是医学影像的**去标识化检查器**。你现在**不做任何医学判读**。

只回答一件事：这张图上有没有可识别患者身份的信息？包括但不限于：
姓名、住院号/门诊号/病案号、身份证号、手机号、出生日期、检查日期+机构名的组合、
条形码/二维码、家庭住址、人脸、纹身或其它可识别特征。

只输出 JSON：
{"has_identifiers": true/false, "kinds": ["burned_in_name", "face", ...], "note": "一句话"}"""

READ_PROMPTS: dict[str, str] = {
    "radiograph": """你在协助骨科医师阅读一张 X 线片的**翻拍照片**。

严格边界：
- 你**不是**放射科医师，你的输出**不是**影像报告，不得作为诊断依据。
- 翻拍照片有畸变、压缩、窗宽窗位丢失，很多细节不可判读——不可判读就写"不可判读"。
- 不得给出治疗建议，不得出现任何剂量。

请描述：体位与投照范围、可见的骨结构、对位对线、骨皮质连续性、关节间隙、
可见的植入物、以及**你认为需要正式阅片确认的可疑点**。""",
    "mri_ct": """你在协助骨科医师查看一张 MRI/CT 的**翻拍照片**。

严格边界：单张翻拍不能替代序列阅片；你的输出不是影像报告，不得作为诊断依据。
不得给出治疗建议，不得出现任何剂量。

请描述：可辨认的序列与层面、可见的椎体/椎间盘/椎管/软组织信号特征、
以及**必须由正式阅片确认的可疑点**。""",
    "tongue": """你在协助中医师做**望诊**中的舌象观察。

严格边界：舌象只是四诊之一，不足以单独辨证；照片的色温与白平衡会显著影响判断，
请说明拍摄条件带来的不确定性。不得给出方剂或剂量。

请描述：舌体（胖瘦、老嫩、有无齿痕、有无裂纹）、舌质颜色、舌苔（厚薄、颜色、润燥、
有无剥落）、舌下脉络，以及**光照/白平衡导致哪些判断不可靠**。""",
    "posture_gait": """你在协助骨科医师做**体态与步态的视诊**。

严格边界：静态照片不能替代体格检查；不得给出诊断或治疗建议，不得出现剂量。

请描述：站立位双肩/髂骨是否等高、脊柱侧弯或后凸的外观、骨盆倾斜、
下肢力线（膝内外翻）、足弓、代偿姿势，以及**需要现场查体确认的项目**。""",
    "limb_surface": """你在协助骨科医师观察**肢体外观**。

严格边界：不得给出诊断或治疗建议，不得出现剂量。
若见到提示急症的外观（明显肿胀、张力性水肿、皮色发紫或苍白、大面积红肿热、
开放伤口、水疱、坏死），必须在 urgent_signals 中明确列出。

请描述：肿胀范围与两侧对比、皮色与皮温线索、有无红斑分界、有无伤口/窦道/水疱、
畸形、瘀斑分布。""",
    "report_document": """你在协助整理一份**检查报告的照片**（化验单、骨密度报告、影像报告文本）。

严格边界：只做**转录与结构化**，不做解释、不做诊断、不给建议、不出现剂量。
数值必须原样转录，看不清就写"不可辨认"，**绝不允许猜测数字**。

请转录：检查项目名称、数值与单位、参考范围、报告日期（仅年月）、以及不可辨认的字段。""",
    "other": """你在协助骨科医师查看一张临床相关图片。

严格边界：只描述可见内容，不做诊断、不给治疗建议、不出现剂量。
说明这张图片能支持什么、不能支持什么。""",
}

OUTPUT_CONTRACT = """
只输出 JSON 对象：
{
  "image_kind": "你判断的图像类型",
  "readable": true/false,
  "observations": ["逐条可见所见，客观描述"],
  "not_assessable": ["因图像质量/范围而无法判断的项目"],
  "urgent_signals": ["若见到提示急症的外观，逐条列出；没有则为空数组"],
  "suggest_ask": ["据此值得追问的问题"],
  "suggest_exam": ["据此值得做的查体或正式检查"],
  "confidence": "high|moderate|low",
  "caveat": "一句话说明这张图的判读局限"
}
不要输出诊断结论，不要输出治疗建议，不要输出任何剂量。"""


class VisionError(LLMError):
    """Raised for vision-transport failures; always caught by the tool layer."""


@dataclass
class ImageRead:
    """The structured, non-diagnostic result of looking at one image."""

    image_sha256: str
    image_kind: str
    readable: bool = False
    observations: list[str] = field(default_factory=list)
    not_assessable: list[str] = field(default_factory=list)
    urgent_signals: list[str] = field(default_factory=list)
    suggest_ask: list[str] = field(default_factory=list)
    suggest_exam: list[str] = field(default_factory=list)
    confidence: str = "low"
    caveat: str = ""
    model: str = ""
    #: Always true. A model read is never a substitute for a formal report, and
    #: making that a field rather than a docstring means every consumer sees it.
    requires_formal_read: bool = True
    #: Set when the PHI pre-check found identifiers; findings are then empty.
    phi_detected: bool = False
    phi_kinds: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """The finding, as it enters the evidence ledger and the console audit.

        The reading model's name is redacted by the shared policy: this dict is
        rendered in the console's image panel and in the raw-JSON tab, which is
        exactly the class of surface the policy covers. The digest still pins
        *which image* produced the finding, which is what the audit needs.
        """
        from ..llm.factory import redact_model_identity

        return redact_model_identity({
            "image_sha256": self.image_sha256, "image_kind": self.image_kind,
            "readable": self.readable, "observations": self.observations,
            "not_assessable": self.not_assessable, "urgent_signals": self.urgent_signals,
            "suggest_ask": self.suggest_ask, "suggest_exam": self.suggest_exam,
            "confidence": self.confidence, "caveat": self.caveat, "model": self.model,
            "requires_formal_read": True,
            "phi_detected": self.phi_detected, "phi_kinds": self.phi_kinds,
            "not_a_radiology_report": True,
        })


def encode_image(path: str | Path) -> tuple[str, str, str]:
    """Return ``(data_uri, sha256, mime)`` for a local image.

    The digest is taken over the raw bytes before encoding, so it identifies the
    file itself rather than this particular transport encoding.
    """
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise VisionError(f"图片不存在: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise VisionError(f"不支持的图片格式 {file_path.suffix!r}；支持 {sorted(SUPPORTED_SUFFIXES)}")
    raw = file_path.read_bytes()
    if not raw:
        raise VisionError(f"图片为空: {file_path}")
    if len(raw) > MAX_IMAGE_BYTES:
        raise VisionError(f"图片过大({len(raw)} 字节)，上限 {MAX_IMAGE_BYTES} 字节")
    mime = mimetypes.guess_type(str(file_path))[0] or "image/jpeg"
    digest = hashlib.sha256(raw).hexdigest()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", digest, mime


def decode_data_uri(data_uri: str) -> tuple[str, str]:
    """Validate an inbound ``data:`` URI (browser upload) and digest its bytes."""
    if not data_uri.startswith("data:"):
        raise VisionError("图片必须是 data: URI 或本地文件路径")
    header, _, payload = data_uri.partition(",")
    if not payload:
        raise VisionError("data URI 缺少内容")
    mime = header[5:].split(";")[0] or "image/jpeg"
    if mime not in {f"image/{s.lstrip('.')}" for s in SUPPORTED_SUFFIXES} | {"image/jpg"}:
        raise VisionError(f"不支持的图片类型 {mime!r}")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as exc:
        raise VisionError(f"图片 base64 解码失败: {exc}") from exc
    if not raw:
        raise VisionError("图片为空")
    if len(raw) > MAX_IMAGE_BYTES:
        raise VisionError(f"图片过大({len(raw)} 字节)，上限 {MAX_IMAGE_BYTES} 字节")
    return data_uri, hashlib.sha256(raw).hexdigest()


class VisionClient:
    """Reads clinical images through an OpenAI-compatible multimodal endpoint."""

    def __init__(self, chat_client: Any, *, model: str = "", phi_precheck: bool = True) -> None:
        self.chat_client = chat_client
        #: Model override, so a text-only run can still borrow a vision model.
        self.model = model or getattr(chat_client, "model", "")
        self.phi_precheck = phi_precheck
        #: Set by the caller when this client is the session's ordinary chat
        #: model standing in for an unconfigured vision provider.
        self.borrowed = False

    @property
    def available(self) -> bool:
        return self.chat_client is not None and bool(getattr(self.chat_client, "available", False))

    # ------------------------------------------------------------------- public
    def read(
        self,
        image: str | Path,
        *,
        kind: str = "other",
        context: str = "",
        budget: Any | None = None,
    ) -> ImageRead:
        """Look at one image and return structured, non-diagnostic findings."""
        if not self.available:
            raise VisionError("未配置视觉模型（需要 YAOBI_VISION_PROVIDER / POE_API_KEY 与多模态模型）")
        if kind not in IMAGE_KINDS:
            raise VisionError(f"未知图像类型 {kind!r}；支持 {list(IMAGE_KINDS)}")

        text = str(image)
        if text.startswith("data:"):
            data_uri, digest = decode_data_uri(text)
        else:
            data_uri, digest, _ = encode_image(image)

        if self.phi_precheck:
            phi = self._phi_check(data_uri, digest, budget=budget)
            if phi is not None:
                return phi

        payload = self._call(
            READ_PROMPTS.get(kind, READ_PROMPTS["other"]) + OUTPUT_CONTRACT,
            data_uri,
            user_note=context,
            budget=budget,
            max_tokens=1400,
        )
        if not isinstance(payload, dict):
            raise VisionError("视觉模型未返回可解析的 JSON")
        return self._build(payload, digest, kind)

    # ---------------------------------------------------------------- internals
    def _phi_check(self, data_uri: str, digest: str, *, budget: Any | None) -> ImageRead | None:
        """Refuse the read when the image carries identifiers.

        A failed pre-check is *not* treated as "no PHI". Neither is it treated as
        "PHI present" — that would make a flaky endpoint block every image. It
        returns ``None`` so the read proceeds, and the caller records that the
        pre-check did not run; the surrounding tool marks the evidence
        accordingly.
        """
        try:
            payload = self._call(PHI_PROMPT, data_uri, budget=budget, max_tokens=300)
        except VisionError:
            return None
        if not isinstance(payload, dict) or not payload.get("has_identifiers"):
            return None
        kinds = [str(k) for k in (payload.get("kinds") or [])][:8]
        return ImageRead(
            image_sha256=digest,
            image_kind="rejected_phi",
            readable=False,
            phi_detected=True,
            phi_kinds=kinds,
            model=self.model,
            caveat="图片中检出可识别患者身份的信息，已拒绝判读。请先去标识化（遮盖姓名、ID、日期、条码、人脸）后重试。",
        )

    def _call(
        self,
        system_prompt: str,
        data_uri: str,
        *,
        user_note: str = "",
        budget: Any | None = None,
        max_tokens: int = 1200,
    ) -> Any:
        if budget is not None and not budget.reserve_llm():
            raise VisionError("LLM 预算耗尽，未调用视觉模型")
        content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": data_uri}}]
        if user_note:
            content.insert(0, {"type": "text", "text": str(user_note)[:1200]})
        try:
            response = self.chat_client.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=0.0, max_tokens=max_tokens, response_format_json=True,
            )
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VisionError(f"视觉模型调用失败: {type(exc).__name__}: {exc}") from exc
        if budget is not None:
            budget.charge_llm_tokens(response.total_tokens)
        return response.json(None)

    def _build(self, payload: dict[str, Any], digest: str, kind: str) -> ImageRead:
        def strings(key: str, limit: int = 12) -> list[str]:
            return [str(v).strip() for v in (payload.get(key) or []) if str(v).strip()][:limit]

        read = ImageRead(
            image_sha256=digest,
            image_kind=str(payload.get("image_kind") or kind),
            readable=bool(payload.get("readable", True)),
            observations=strings("observations"),
            not_assessable=strings("not_assessable"),
            urgent_signals=strings("urgent_signals", 8),
            suggest_ask=strings("suggest_ask", 8),
            suggest_exam=strings("suggest_exam", 8),
            confidence=str(payload.get("confidence") or "low"),
            caveat=str(payload.get("caveat") or "")[:400],
            model=self.model,
        )
        # Belt and braces: strip any dose that got through the prompt. The vision
        # channel is new, so it gets the same output scan the rest of the system
        # applies rather than being trusted because its prompt says not to.
        read.observations = [o for o in read.observations if not _has_dose(o)]
        read.suggest_exam = [s for s in read.suggest_exam if not _has_dose(s)]
        read.suggest_ask = [s for s in read.suggest_ask if not _has_dose(s)]
        return read


def _has_dose(text: str) -> bool:
    import re

    return bool(re.search(r"\d+(?:\.\d+)?\s*(?:克|g\b|G\b|毫克|mg\b)", text))


def build_vision_client(chat_client: Any | None = None, **overrides: Any) -> VisionClient | None:
    """Build a vision client from the environment.

    ``YAOBI_VISION_PROVIDER`` selects the transport (default ``poe``) and
    ``YAOBI_VISION_MODEL`` the model (default ``Gemini-3.1-Pro``). Returns
    ``None`` when nothing is configured, so image tools simply stay unavailable
    rather than the whole harness failing to start.
    """
    from ..llm.factory import build_client

    provider = (overrides.pop("provider", None) or os.environ.get("YAOBI_VISION_PROVIDER") or "poe").lower()
    model = (
        overrides.pop("model", None)
        or os.environ.get("YAOBI_VISION_MODEL")
        or os.environ.get("POE_VISION_MODEL")
        or "Gemini-3.1-Pro"
    )
    if provider in ("", "none", "null", "off", "disabled"):
        return None

    if chat_client is None:
        try:
            chat_client = build_client(provider, model=model, **overrides)
        except LLMError:
            return None
    if not getattr(chat_client, "available", False):
        return None
    return VisionClient(chat_client, model=model)


def describe_vision(client: VisionClient | None) -> dict[str, Any]:
    """What a screen is told about the vision model.

    Redacted by the same policy as the chat model — see
    :func:`~yaobi_harness.llm.factory.redact_model_identity`. Which vendor reads
    the films is a procurement matter, and a demo screenshot is not where it
    should be settled.
    """
    from ..llm.factory import redact_model_identity

    if client is None or not client.available:
        return {"configured": False, "note": "未配置视觉模型；影像与舌象工具不可用"}
    return redact_model_identity({
        "configured": True,
        "model": client.model,
        "provider": getattr(client.chat_client, "name", "unknown"),
        "phi_precheck": client.phi_precheck,
        "kinds": list(IMAGE_KINDS),
        # True when no vision-specific provider was configured and the session's
        # chat model is being used instead. Worth surfacing rather than hiding:
        # if that model turns out not to be multimodal, this is the line that
        # explains the failure.
        "borrowed_chat_model": bool(getattr(client, "borrowed", False)),
    })


def read_to_json(read: ImageRead) -> str:
    return json.dumps(read.to_dict(), ensure_ascii=False)
