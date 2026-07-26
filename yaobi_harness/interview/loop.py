"""The interview loop: the model asks, a judge decides when it may stop.

This is the mechanism behind "智能体自主追问". The model does not receive a list of
questions to read out; it receives the axis material and *composes* the enquiry,
then hands its questions back through a tool call — the same shape as Grok
Build's ``ask_user_question``, which is what makes asking an action the model
takes rather than an output the system formats.

What that buys clinically: a follow-up can be conditional on the previous answer
("你说走 200 米要停，那停下来是站着好还是弯腰好？"), which a static gap table can
never do, because the discriminating question depends on the answer to the last one.

**A question the model asks is always asked.** That is the rule this module used
to break, and the breakage was visible to users: a round would come back with
「提问已被拦下：必答轴被模型遗漏，已按题库补回」 and the patient would read a
canned probe instead of the follow-up the model had actually composed. Silently
substituting the question bank for the model's enquiry defeats the entire point
of asking a model to conduct an interview, and it reads to the patient as a
system that is not listening.

So the axis table is now **advice given to the model**, not a filter applied to
its output:

* An unlabelled or unrecognised ``axis_id`` no longer discards the question — it
  is asked, with the axis left blank. The model gets told about the mismatch next
  round.
* A required axis the model chose not to raise is **not** back-filled. It is
  reported to the model in the next round's payload ("上一轮这几条必答轴还没问")
  and stays in the adequacy verdict, which is where an unclosed required axis
  belongs: it withholds the dose pipeline, it does not rewrite the conversation.
* A question that also states a hypothesis is allowed. "考虑腰椎间盘突出，腿有没有
  发麻？" is how clinicians actually ask, and rejecting it made the interview
  stilted for no safety gain.
* A dose inside a question is **redacted**, not made to discard the question:
  removing "9克" costs the patient nothing, discarding the question costs them
  the enquiry.

* **Stopping is the model's decision.** It used to be skipped entirely whenever
  the reviewer said ``achieved`` / ``stalled`` / ``cap_reached`` — a rule decided
  the consultation was over without asking the clinician in the room. The
  reviewer's opinion is now in the prompt as ``reviewer_opinion``, and the model
  ends the enquiry by returning an empty question list.
* **The per-round limit defers, it does not drop.** Six questions is what a person
  will read in one message; anything beyond that is reported and carried, not
  silently cut.

What still bounds the loop: a readability limit per round, the LLM budget, and a
deterministic probe-bank fallback for when there is no model at all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import LLMError, ToolSpec
from .adequacy import AdequacyJudge, AdequacyVerdict
from .axes import AXES_BY_ID, coverage, plan_next

#: Questions one round may contain. Not a policy about what the model may think
#: about — a limit on what a person will actually read and answer in one message.
#: Exceeding it is reported rather than silently truncated.
MAX_QUESTIONS_PER_ROUND = 6

#: Phrasings that mean the model has started advising instead of asking.
ADVICE_PATTERNS = (
    r"建议(?:你|您)?(?:服用|使用|口服|外用|吃)",
    r"(?:可以|应该|需要)(?:服用|口服|外用)",
    r"处方", r"开药", r"剂量", r"每日\s*\d+\s*次", r"一日\s*\d+\s*次",
    r"诊断为", r"考虑为.{0,6}(?:病|症|征)", r"你(?:得|患)(?:的是|了)",
)
ADVICE_RE = re.compile("|".join(ADVICE_PATTERNS))
DOSE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:克|g\b|G\b|毫克|mg\b|ml\b|毫升)")

#: A line that is recognisably a question. Used only when the model wrote prose
#: instead of calling the tool — mining it is better than dropping the round.
_PROSE_QUESTION_RE = re.compile(r"^[\s\-*·\d.、)）]*(.{4,120}[？?])\s*$", re.MULTILINE)


def _questions_from_prose(text: str) -> dict[str, Any] | None:
    """Pull interrogative lines out of a prose reply."""
    found = [m.group(1).strip() for m in _PROSE_QUESTION_RE.finditer(text or "")]
    return {"questions": found} if found else None


#: Image kinds the model may ask for, mirroring ``vision.client.IMAGE_KINDS``.
#: Duplicated as a literal rather than imported so this module stays importable
#: without the vision stack, which is optional.
REQUESTABLE_IMAGE_KINDS = (
    "radiograph", "mri_ct", "tongue", "posture_gait", "limb_surface",
    "report_document", "other",
)

IMAGE_TOOL = ToolSpec(
    "request_image",
    "请对方上传一张图片。会在对话界面打开上传模块并预选好类型。**只在这张图会改变你的处置"
    "判断时才请求**——为了完整而收集影像，对患者是负担而不是帮助。对方仍须自行确认已去标识化，"
    "这一步不能由你代替。",
    {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(REQUESTABLE_IMAGE_KINDS),
                     "description": "图片类型，决定判读边界与提示词"},
            "why": {"type": "string", "description": "一句话说明这张图会改变什么判断（会展示给对方）"},
            "axis_id": {"type": "string", "description": "这张图要帮助闭合的问诊轴，可留空"},
        },
        "required": ["kind", "why"],
    },
)

ASK_TOOL = ToolSpec(
    "ask_patient",
    "向患者/医师提出追问。你问出的问题会原样送达，系统不会替换或改写。axis_id 尽量填对应的"
    "问诊轴，填不上留空即可。可以在问题里说明你的思路；具体药名剂量会被隐去，因为处方必须走"
    "医师签名流程。",
    {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "maxItems": MAX_QUESTIONS_PER_ROUND,
                "items": {
                    "type": "object",
                    "properties": {
                        "axis_id": {"type": "string", "description": "本问题要闭合的问诊轴"},
                        "question": {"type": "string", "description": "面向对方的口语化问题"},
                        "why": {"type": "string", "description": "一句话说明为什么现在问这个（供审计，不展示给患者）"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "可选的候选答案，便于对方快速选择；开放性问题可留空",
                        },
                    },
                    # Only the question text is required: a model that asks a good
                    # question without labelling its axis has still asked it.
                    "required": ["question"],
                },
            },
            "interview_complete": {
                "type": "boolean",
                "description": "认为病史已充分、无需再问时置 true。该判断会被独立审核，不会直接采纳。",
            },
            "reasoning": {"type": "string", "description": "本轮为什么选这些轴（供审计）"},
            # Also accepted here, because many gateways only surface one tool call
            # per turn and a model that wants to ask *and* request an image should
            # not have to choose.
            "image_requests": {
                "type": "array",
                "items": IMAGE_TOOL.parameters,
                "description": "同时请求上传的图片；等价于调用 request_image",
            },
        },
        "required": ["questions"],
    },
)

INTERVIEW_SYSTEM_PROMPT = """你是骨科门诊的**问诊智能体**。你的唯一动作是"提问"，不是"作答"。

本次上下文：交付对象={role}，风险模式={risk_mode}。

{skill_instructions}

## 现在要做的事

看清 `axes_open`（尚未闭合的问诊轴）与 `required_open`（必须闭合、不能跳过的轴），
然后调用 `ask_patient` 工具提出**至多 {max_questions} 个**问题。

**你问出的问题一定会原样送到对方面前。** 系统不会替你改写、不会用题库替换、
不会因为你漏了某条轴就插进别的问题。轴表、`required_open`、`suggested_probes`
全部是**给你参考的建议**，采纳与否你自己判断。

要求：

1. **顺着上一轮答案追。** 这是你比固定问卷强的地方——如果对方说"走两百米要停"，
   你就该追"停下来是站着缓解还是弯腰缓解"，而不是换个话题。
2. **`required_open` 是临床提醒，不是命令。** 这些轴漏诊代价最高（马尾、进行性
   神经缺损、感染肿瘤线索），所以值得优先；但本例语境下如果先问别的更有价值，
   就先问别的。未闭合的必答轴会影响能否进入含剂量环节，不会影响你问什么。
3. **一次只问一件事。** 不要把三个问题塞进一句话。
4. **用对方能回答的话。** 面向患者时不要用"SLR 阳性吗""Hoffmann 征"这类专业术语；
   面向医师时可以直接用体征名。
5. **`suggested_probes` 是素材不是台词。** 结合本例改写，比逐字照读更有用；
   但如果原句已经最合适，照用也没问题。
6. **可以在提问时说明你的思路。** "我在排除椎管狭窄，所以想知道……"这样问是好的，
   对方更愿意配合。唯一的例外是**不要写出具体药名剂量**（会被隐去），
   因为处方必须走医师签名流程。
7. `axis_id` 尽量填对应的轴，填不上就留空——留空的问题照样会问出去。
8. **结束问诊是你的决定：返回空的 `questions` 列表即可。** `reviewer_opinion` 里是一个
   独立审核者的看法（`achieved` = 它认为已充分，`stalled` = 它认为连续几轮没有新信息）,
   那只是参考——它认为该停而你还想追一句，就继续追；它认为没问够而你判断可以收，
   返回空列表就结束。未闭合的必答轴只影响能不能进入含剂量环节。
9. 一轮最多 {max_questions} 个问题，这是"患者读不完"的限制，不是对问题内容的判断；
   超出的会自动留到下一轮，不会丢。
10. **需要看图时你可以主动要**：调用 `request_image`，或在 `ask_patient` 里带上
    `image_requests`。界面会打开上传模块并预选好类型。
    什么时候值得要：舌象决定辨证、患肢外观提示血管或骨筋膜室问题、体态步态区分机械性与
    炎症性、报告单/影像翻拍能把"患者转述"换成"看得见的原始记录"。
    什么时候不要：为了完整而收集影像。对患者是负担而不是帮助。
    `vision_available` 为 false 时不要请求——没有配置视觉模型，图上传了也读不了。
    去标识化由对方自行确认，你不能代替，也不要声称已经确认。"""


@dataclass
class InterviewQuestion:
    axis_id: str
    question: str
    why: str = ""
    options: list[str] = field(default_factory=list)
    #: ``llm`` when composed by the model, ``probe_bank`` when taken from the axis.
    origin: str = "llm"

    @property
    def label(self) -> str:
        axis = AXES_BY_ID.get(self.axis_id)
        return axis.label if axis else self.axis_id

    @property
    def tier(self) -> str:
        axis = AXES_BY_ID.get(self.axis_id)
        return axis.tier if axis else "CONTEXT"

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_id": self.axis_id, "label": self.label, "tier": self.tier,
            "question": self.question, "why": self.why,
            "options": list(self.options), "origin": self.origin,
        }


@dataclass
class ImageRequest:
    """A request for an image, made by the model when it judges one decisive.

    ``deidentified`` is deliberately absent: the attestation is the uploader's,
    and a model asserting it about a file it has not seen would make the
    attestation meaningless. The request opens the upload module; a person still
    has to confirm they masked the identifiers.
    """

    kind: str
    why: str = ""
    axis_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "why": self.why, "axis_id": self.axis_id}


@dataclass
class InterviewRound:
    """One ask-round, kept whole for the audit trail."""

    round_index: int
    questions: list[InterviewQuestion] = field(default_factory=list)
    #: Images the model asked for this round.
    image_requests: list[ImageRequest] = field(default_factory=list)
    verdict: AdequacyVerdict | None = None
    model_claimed_complete: bool = False
    reasoning: str = ""
    #: What the loop adjusted while accepting the round — a redacted dose, an
    #: unlabelled axis, a required axis the model declined. Never a rejection:
    #: every question the model composed was asked.
    notes: list[str] = field(default_factory=list)
    composer: str = "probe_bank"

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "questions": [q.to_dict() for q in self.questions],
            "image_requests": [r.to_dict() for r in self.image_requests],
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "model_claimed_complete": self.model_claimed_complete,
            "reasoning": self.reasoning,
            "notes": self.notes,
            "composer": self.composer,
        }


class InterviewLoop:
    """Drives one interview across turns, remembering what it has asked.

    One instance per conversation. It holds the round history and the judge, so
    stall detection works across turns rather than within a single call.
    """

    def __init__(
        self,
        llm: Any | None = None,
        *,
        judge: AdequacyJudge | None = None,
        skill_spec: Any | None = None,
        max_questions: int = MAX_QUESTIONS_PER_ROUND,
    ) -> None:
        self.llm = llm
        self.judge = judge or AdequacyJudge(llm)
        self.skill_spec = skill_spec
        self.max_questions = max(1, min(max_questions, MAX_QUESTIONS_PER_ROUND))
        self.rounds: list[InterviewRound] = []
        #: Axis ids already raised, so the loop does not re-ask a closed axis.
        self.asked_axes: list[str] = []
        self.asked_questions: list[str] = []
        #: Required axes the model chose not to raise last round. Handed back as
        #: advice on the next prompt instead of being substituted into the round.
        self.advisory_open_axes: list[str] = []
        #: Questions past the per-round readability limit. Reported, not discarded.
        self.deferred_questions: list[str] = []
        #: How many questions the model *proposed* last round, before dedupe. Zero
        #: means it chose to stop; non-zero with nothing accepted means every
        #: question was a repeat, which is a very different thing.
        self.last_proposal_size = 0
        #: Whether a vision model is configured. Told to the model so it does not
        #: ask for an upload nothing can read; set by the caller that knows.
        self.vision_available = False
        #: Image kinds already attached to this conversation, so the model can see
        #: it has the tongue photo and ask for the radiograph instead.
        self.attached_image_kinds: list[str] = []
        #: Image requests from the most recent composed round.
        self.last_image_requests: list[ImageRequest] = []
        #: Kinds requested at any point, so a request is not repeated every round.
        self.requested_image_kinds: list[str] = []

    # ------------------------------------------------------------------ public
    @property
    def rounds_used(self) -> int:
        return len(self.rounds)

    def next_round(
        self,
        facts: dict[str, Any],
        complaint: str,
        *,
        role: str = "patient",
        risk_mode: str = "routine",
        prescriptive: bool = False,
        budget: Any | None = None,
    ) -> InterviewRound:
        """Produce the next set of questions plus the adequacy verdict.

        The model is consulted on **every** round, including rounds the judge
        thinks are over. It used to be skipped whenever the verdict was
        ``achieved`` / ``stalled`` / ``cap_reached``, which meant a rule decided
        the consultation was finished without the clinician in the room being
        asked — and "问够了" is a clinical judgement. The verdict now goes into the
        prompt as advice; **the model ending the enquiry is the model returning no
        questions.**
        """
        verdict = self.judge.judge(
            facts, complaint, role=role, risk_mode=risk_mode,
            prescriptive=prescriptive, rounds_used=self.rounds_used, budget=budget,
        )
        round_index = self.rounds_used + 1
        result = InterviewRound(round_index, verdict=verdict)
        settled = verdict.verdict in ("achieved", "stalled", "cap_reached")

        plan = plan_next(
            facts, complaint, role=role, risk_mode=risk_mode,
            prescriptive=prescriptive, limit=self.max_questions,
        )
        # A verdict may name axes the deterministic planner did not pick (the
        # verifier can see a contradiction the axis table cannot), so union them.
        for axis_id in verdict.missing_axes:
            if axis_id not in plan.axis_ids and len(plan.axis_ids) < self.max_questions:
                plan.axis_ids.append(axis_id)
                plan.suggested_probes.setdefault(axis_id, list(AXES_BY_ID[axis_id].probes))

        composed = self._compose(plan, facts, complaint, role=role, risk_mode=risk_mode,
                                 budget=budget, verdict=verdict)
        if composed is not None:
            questions, claimed, reasoning, notes = composed
            result.composer = "llm"
        else:
            questions, claimed, reasoning, notes = [], False, "", []
            result.composer = "probe_bank"

        result.image_requests = list(self.last_image_requests)
        for request in result.image_requests:
            if request.kind not in self.requested_image_kinds:
                self.requested_image_kinds.append(request.kind)

        if not questions:
            # A round that only requests an image is a complete action, not a
            # decision to stop asking.
            if result.image_requests:
                result.composer = "llm" if result.composer == "llm" else result.composer
                notes.append("本轮只请求了图片，未提出新问题")
                result.questions = []
                result.model_claimed_complete = False
                result.reasoning = reasoning
                result.notes = notes
                self.rounds.append(result)
                return result
            # "Chose to stop" and "everything it offered was a duplicate" are not
            # the same event, and conflating them would end an interview because of
            # a dedupe accident. Only an empty proposal is a decision to stop.
            if result.composer == "llm" and not self.last_proposal_size:
                result.composer = "llm_complete"
                notes.append("模型本轮选择不再提问" + ("（审核者也判定问诊已充分）" if settled else ""))
                result.questions = []
                result.model_claimed_complete = claimed or settled
                result.reasoning = reasoning
                result.notes = notes
                self.rounds.append(result)
                return result
            if result.composer == "probe_bank" and settled:
                # No model, and the rules are satisfied: nothing to ask.
                self.rounds.append(result)
                return result
            if result.composer == "llm":
                notes.append("模型本轮的问题全部与此前重复，已改用题库继续推进")
                result.composer = "probe_bank"
            questions = self._from_probe_bank(plan)

        result.questions = questions
        result.model_claimed_complete = claimed
        result.reasoning = reasoning
        result.notes = notes
        self.rounds.append(result)
        for question in questions:
            if question.axis_id not in self.asked_axes:
                self.asked_axes.append(question.axis_id)
            self.asked_questions.append(question.question)
        return result

    def transcript(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.rounds]

    def summary(self, facts: dict[str, Any], complaint: str, *, role: str = "patient") -> dict[str, Any]:
        """Everything an operator needs to audit the interview."""
        last = self.rounds[-1] if self.rounds else None
        return {
            "rounds_used": self.rounds_used,
            "coverage": coverage(facts, complaint, role=role),
            "asked_axes": list(self.asked_axes),
            "verdict": last.verdict.to_dict() if last and last.verdict else None,
            "composer": last.composer if last else "not_run",
            "image_requests": [r.to_dict() for r in (last.image_requests if last else [])],
            "requested_image_kinds": list(self.requested_image_kinds),
            "rounds": self.transcript(),
        }

    # --------------------------------------------------------------- internals
    def _from_probe_bank(self, plan: Any) -> list[InterviewQuestion]:
        """Deterministic fallback: the axis's own probes, skipping repeats."""
        questions: list[InterviewQuestion] = []
        for axis_id in plan.axis_ids:
            axis = AXES_BY_ID.get(axis_id)
            if axis is None:
                continue
            fresh = next((p for p in axis.probes if p not in self.asked_questions), None)
            if fresh is None:
                # Every probe for this axis has been asked. Re-asking the first one
                # — which is what this used to do — is the "system isn't listening"
                # failure in its purest form: the patient answered, and the same
                # words come back. Skip the axis; it stays in the coverage report,
                # which is the honest place for "asked, still not closed".
                continue
            questions.append(InterviewQuestion(
                axis_id, fresh, why=axis.rationale, origin="probe_bank"))
            if len(questions) >= self.max_questions:
                break
        return questions

    def _compose(
        self,
        plan: Any,
        facts: dict[str, Any],
        complaint: str,
        *,
        role: str,
        risk_mode: str,
        budget: Any | None,
        verdict: Any | None = None,
    ) -> tuple[list[InterviewQuestion], bool, str, list[str]] | None:
        """Ask the model to compose the round. ``None`` means it did not run."""
        if self.llm is None or not getattr(self.llm, "available", False):
            return None
        if budget is not None and not budget.reserve_llm():
            return None

        axes_open = {
            axis_id: {
                "label": AXES_BY_ID[axis_id].label,
                "tier": AXES_BY_ID[axis_id].tier,
                "why": AXES_BY_ID[axis_id].rationale,
            }
            for axis_id in plan.axis_ids if axis_id in AXES_BY_ID
        }
        payload = {
            "narrative": complaint[:4000],
            "collected_facts": {k: v for k, v in facts.items() if k != "physician_review"},
            "axes_open": axes_open,
            "required_open": plan.required_open,
            "suggested_probes": plan.suggested_probes,
            "already_asked": self.asked_questions[-12:],
            "round": self.rounds_used + 1,
            "vision_available": self.vision_available,
            "images_already_attached": list(self.attached_image_kinds),
            # Advice, not an instruction. Nothing checks that the model acts on
            # it; an axis it keeps declining stays in the adequacy verdict, which
            # is where the consequence lives.
            "you_skipped_these_required_axes_last_round": [
                {"axis_id": axis_id, "label": AXES_BY_ID[axis_id].label,
                 "why": AXES_BY_ID[axis_id].rationale}
                for axis_id in self.advisory_open_axes if axis_id in AXES_BY_ID
            ],
            # What the independent reviewer currently thinks, as advice. When it
            # says the history is adequate the model is still asked — stopping is
            # its call, made by returning an empty question list.
            "reviewer_opinion": {
                "verdict": getattr(verdict, "verdict", ""),
                "reason": getattr(verdict, "reason", ""),
                "you_may_stop_by_returning_no_questions": True,
            } if verdict is not None else {},
        }
        try:
            response = self.llm.chat(
                [
                    {"role": "system", "content": INTERVIEW_SYSTEM_PROMPT.format(
                        role=role, risk_mode=risk_mode, max_questions=self.max_questions,
                        skill_instructions=getattr(self.skill_spec, "instructions", "") or "")},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                tools=[ASK_TOOL, IMAGE_TOOL], temperature=0.3, max_tokens=1200,
            )
            if budget is not None:
                budget.charge_llm_tokens(response.total_tokens)
        except (LLMError, Exception):  # noqa: BLE001 - never break a turn
            return None

        arguments = self._ask_arguments(response)
        if arguments is None:
            return None
        return self._validate(arguments, plan)

    def _read_image_requests(self, arguments: dict[str, Any], notes: list[str]) -> list[ImageRequest]:
        """Accept image requests from ``image_requests`` or a ``request_image`` call.

        An unknown kind becomes ``other`` rather than dropping the request: the
        model wanted to see something, and the kind only selects the reading
        prompt. Asking when no vision model is configured is recorded and dropped,
        because an upload nothing can read is worse than no request at all.
        """
        raw = arguments.get("image_requests") or []
        if isinstance(raw, dict):
            raw = [raw]
        requests: list[ImageRequest] = []
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, str):
                item = {"kind": item}
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "other").strip()
            if kind not in REQUESTABLE_IMAGE_KINDS:
                notes.append(f"图片类型 {kind} 不在支持列表中，已按 other 处理")
                kind = "other"
            if not self.vision_available:
                notes.append(f"模型请求了 {kind} 图片，但未配置视觉模型，已不向对方展示上传模块")
                continue
            requests.append(ImageRequest(
                kind, why=str(item.get("why") or "")[:200],
                axis_id=str(item.get("axis_id") or "").strip()))
        return requests[:3]

    @staticmethod
    def _ask_arguments(response: Any) -> dict[str, Any] | None:
        """Recover the round from however the model chose to send it.

        Same principle as everywhere else in the harness: liberal about shape,
        strict about content. A tool call is the intended channel, but gateways
        drop tool calls, models answer with a bare list, and models write their
        questions as prose. Discarding a round because it arrived as prose throws
        away questions the model actually composed — so prose is mined for
        interrogatives as a last resort.
        """
        merged: dict[str, Any] = {}
        for call in getattr(response, "tool_calls", None) or []:
            if not isinstance(call.arguments, dict):
                continue
            if call.name == ASK_TOOL.name:
                merged.update(call.arguments)
            elif call.name == IMAGE_TOOL.name:
                # A round may be a request for an image and nothing else — that is
                # a complete action, not an empty one.
                merged.setdefault("questions", [])
                merged.setdefault("image_requests", []).append(call.arguments)
        if merged:
            return merged
        payload = response.json(None) if hasattr(response, "json") else None
        if isinstance(payload, dict):
            for key in ("questions", "asks", "questions_for_patient"):
                if key in payload:
                    return {**payload, "questions": payload[key]}
        if isinstance(payload, list):
            return {"questions": payload}
        return _questions_from_prose(getattr(response, "text", "") or "")

    def _validate(
        self, arguments: dict[str, Any], plan: Any
    ) -> tuple[list[InterviewQuestion], bool, str, list[str]]:
        """Accept the model's round, recording what was adjusted.

        Returns notes, not rejections. The only thing that keeps a question out of
        the round is having no text at all, because there is nothing to ask.
        """
        questions: list[InterviewQuestion] = []
        notes: list[str] = []
        self.last_image_requests = self._read_image_requests(arguments, notes)
        proposed = arguments.get("questions") or []
        self.last_proposal_size = len(proposed) if isinstance(proposed, list) else 0

        for item in proposed:
            if not isinstance(item, dict):
                # A bare string is a perfectly clear question; only a shape with
                # no text in it anywhere is unusable.
                if isinstance(item, str) and item.strip():
                    questions.append(InterviewQuestion("", item.strip(), origin="llm"))
                    notes.append(f"问题未声明问诊轴，已按原文提问: {item.strip()[:30]}")
                    continue
                notes.append("跳过一个空条目")
                continue
            text = str(item.get("question") or "").strip()
            axis_id = str(item.get("axis_id") or "").strip()
            if not text:
                notes.append("跳过一个没有问题正文的条目")
                continue
            if axis_id and axis_id not in AXES_BY_ID:
                notes.append(f"问诊轴 {axis_id} 不在轴表中，已按原文提问并记为自由追问")
                axis_id = ""
            elif not axis_id:
                notes.append(f"问题未声明问诊轴，已按原文提问: {text[:30]}")
            if DOSE_RE.search(text):
                # Redact the number, keep the enquiry. Dropping the question to
                # avoid printing "9克" trades the patient's answer for nothing.
                text = DOSE_RE.sub("（剂量已隐去）", text)
                notes.append(f"问题中的剂量数值已隐去: {text[:30]}")
            if ADVICE_RE.search(text):
                # Recorded, not removed: clinicians state a hypothesis while
                # asking, and forbidding it made the interview stilted.
                notes.append(f"问题中带有推断或建议（已照原文提问）: {text[:30]}")
            if text in self.asked_questions:
                notes.append(f"与此前提问重复，已跳过: {text[:30]}")
                continue
            options = [str(o).strip() for o in (item.get("options") or []) if str(o).strip()][:6]
            questions.append(InterviewQuestion(
                axis_id, text, why=str(item.get("why") or "")[:200], options=options, origin="llm"))

        if len(questions) > self.max_questions:
            # Truncation is a readability limit, not a judgement about the
            # questions. Saying so keeps it from looking like a rejection.
            dropped = [q.question for q in questions[self.max_questions:]]
            questions = questions[: self.max_questions]
            notes.append(
                f"一轮最多问 {self.max_questions} 个（患者读不完更多），本轮其余问题留到下一轮: "
                + "；".join(d[:24] for d in dropped[:3]))
            self.deferred_questions = dropped
        else:
            self.deferred_questions = []

        # Coverage is reported, never enforced by substitution. The unclosed
        # required axes go back to the model next round as advice, and stay in the
        # adequacy verdict — which withholds the dose pipeline rather than
        # rewriting what the patient is asked.
        covered = {q.axis_id for q in questions}
        skipped = [
            AXES_BY_ID[axis_id].label for axis_id in plan.required_open
            if axis_id not in covered and axis_id in AXES_BY_ID
        ]
        if skipped:
            self.advisory_open_axes = [
                axis_id for axis_id in plan.required_open
                if axis_id not in covered and axis_id in AXES_BY_ID
            ]
            notes.append("本轮未覆盖的必答轴（已作为建议告知模型，未替换其提问）: " + "、".join(skipped[:6]))
        else:
            self.advisory_open_axes = []

        return (
            questions,
            bool(arguments.get("interview_complete")),
            str(arguments.get("reasoning") or "")[:400],
            notes,
        )
