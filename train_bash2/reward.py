import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict
from swift.plugin import ORM, orms
from src.llm import Agent
import random
import time
from openai import OpenAI
import json

PROMPT = """你是一名严谨的学术评审专家，需要根据给定的 checklist，对一份模型生成的回答进行逐项评分和点评。

【任务说明】
checklist 每一项包含：
- name：指标名称；
- criteria_description：该指标的详细说明，以及各分数区间描述。

请你按照 checklist 中条目的顺序，对每一个 name 依次进行评估。对每一项，请严格按照下面格式输出：
name：<与 checklist 中 name 完全一致>
评分：<得分>/<总分>
评价和改进建议：<你的评价与改进建议>

然后继续下一项，以此类推，直到所有 checklist 条目都被评估。

【评分要求】
- 评分为整数，只输出数字，不包含其他字符；
- 评价和改进建议需要同时包含该项做得好的地方和可以改进的方向。

【输出格式要求】
- 只输出上述三行一组的重复结构，不要输出任何其他说明；
- 各项之间可以用空行分隔。


【待评估的模型回答】
{response}

【checklist】
{checklist_str}
"""



class WriterRewardFunction(ORM):
    def _clip(self, x: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))


    def score_pair(self, pred_text: str, gt_text: str) -> float:
        try:
            agent = Agent("You are a helpful assistant.")
            response, success = agent.run(
                PROMPT.format(
                    response=pred_text, 
                    checklist_str=gt_text
                )
            )
            
            if success:
                try:
                    review_json = self._parse_reviewer_text(review_text=response)
                    get_score=0
                    total_score=0
                    for item in review_json:
                        try:
                            score = float(item["score"])
                            total = float(item["total"])
                        except Exception as e:
                            print(f"Error parsing score/total: {item}. Error: {e}",flush=True)
                            continue

                        # 合理性检查：total 必须 > 0，score 不能为负
                        if total <= 0 or score < 0:
                            continue
                        if score > total:
                            continue

                        get_score += score
                        total_score += total

                    # 防止除零
                    if total_score == 0:
                        overall_score = 0.4   # 或者 raise ValueError("No valid score parsed")
                    else:
                        overall_score = get_score / total_score
                    final=overall_score 
                except:
                  final=0.4
            else:
                try:
                    client = OpenAI(
                        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
                        api_key='', 
                        base_url="",
                    )
                    completion = client.chat.completions.create(
                        model="",  
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {'role': 'user', 'content': PROMPT.format(
                                response=pred_text, 
                                checklist_str=gt_text
                            )}
                        ],
                        temperature=0.7,
                        top_p=0.8,
                        presence_penalty=1.05, 
                        extra_body={"enable_thinking": True},
                    )
                    response=json.loads(completion.model_dump_json())["choices"][0]['message']['content']
                    try:
                        review_json = self._parse_reviewer_text(review_text=response)
                        get_score=0
                        total_score=0
                        for item in review_json:
                            try:
                                score = float(item["score"])
                                total = float(item["total"])
                            except Exception as e:
                                print(f"Error parsing score/total: {item}. Error: {e}",flush=True)
                                continue

                            # 合理性检查：total 必须 > 0，score 不能为负
                            if total <= 0 or score < 0:
                                continue
                            if score > total:
                                continue

                            get_score += score
                            total_score += total

                        # 防止除零
                        if total_score == 0:
                            overall_score = 0.4   # 或者 raise ValueError("No valid score parsed")
                        else:
                            overall_score = get_score / total_score
                        final=overall_score 
                    except:
                        final=0.4
                except:
                  final=0.4
              
        except Exception as e:
            err = traceback.format_exc()
            print(err)
            final=0.4


        return self._clip(final, 0, 1.0)
    
    def _parse_reviewer_text(self,review_text: str) -> Dict[str, Any]:
        """
        使用正则从纯文本中解析出：
            name / score / total / comment
        并计算平均得分（统一折算到 10 分制）。
        """
        text = review_text.strip()

        items = []

        pattern = re.compile(
            r"(?:评分|得分|分数)\s*(?:[:：=]|为|是)?\s*"
            r"(?P<score>\d+(?:\.\d+)?)\s*\D{0,30}?\s*"
            r"(?:/|／|\\)\s*\D{0,30}?\s*"
            r"(?P<total>\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )

        for m in pattern.finditer(text):
            score = m.group("score")
            total = m.group("total")

            items.append(
                {
                    "score": score,
                    "total": total
                }
            )

        return items

    
    def _unpack_and_score(self, args):
        """
        辅助方法：解包参数并执行评分。
        用于 executor.map，因为它只接受一个参数。
        """
        time.sleep(random.randint(1, 8))
        pred, gt = args
        try:
            return float(self.score_pair(pred_text=pred, gt_text=gt))
        except Exception as e:
            # 返回保底分
            print(f"Error in thread processing: {e}")
            return 0.4


    def __call__(self, completions, solution, **kwargs):
        data_pairs = list(zip(completions, solution))
        max_workers = 16
        # 只输出data_pairs的一个结果
        print("生成结果和真值结果示例：", [data_pairs[0]] if data_pairs else "No data pairs", flush=True)
        # print("生成结果和真值结果：",data_pairs,flush=True)
        rewards = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # map 会保持结果顺序与输入顺序一致
            results = executor.map(self._unpack_and_score, data_pairs)
            rewards = list(results)
        print(rewards,flush=True)
        return rewards



orms["writer_reward"] = WriterRewardFunction



if __name__ == "__main__":
    # 初始化
    reward_fn = WriterRewardFunction()
    pred_text='Chrono Mysteries: A Paradoxical Recipe for Long-Term Engagement  \nYouth Lecture Review (≈ 2 300 words)\n\n1.  Time to Begin, Then Again—The Hook That Never Looses Grip  \nYoung audiences crave immediacy, but television today rewards shows that can out-persist TikTok momentum.  Chrono Mysteries (first aired 2023, currently in Season 3) hooks with a premise that literally refuses to be summarized in a single sentence: “A scientist and her sister chase a false verdict across erased centuries to prove who’s actually rewriting reality.”  \n\nEpisode 1 opens on a 30-second cold-open that violates every storytelling rule of causality.  Dr. Ava Chronos steps into her lab, hits the red button on a time-travel prototype, and—boom—the camera rewinds three seconds before she presses it.  We see the same shot again, but now her reflection shows a faint scar that didn’t exist in the previous instant.  No dialogue, no title cards; only the visual rupture.  The effect is brutal in its simplicity: it forces viewers to confront the question we never ask ourselves in casual life—what if the past was a document we’re editing live?  \n\nThe episode’s second act doubles down by making the paradox feel personal.  When Ava’s sister Maya is arrested for “crimes against time,” the show refuses the easy moral alignment we expect from hero-victim tropes.  Instead, it gives us a 48-hour countdown that restarts itself whenever Ava tries to stop it.  Each reset drops another clue, but also erases the previous attempt.  The audience is stuck in a feedback loop identical to the one Ava experiences: trying, failing, trying again.  Cognitive psychologists call this “productive confusion,” and it’s one reason why Chrono Mysteries keeps its younger viewers glued—because the puzzle becomes our own second screen.\n\n2.  Layers Unpeeled: A Coherent Logic That Rewards Repeat Watching  \nTime-travel series often fall into the trap of “plot-hole porn,” where audiences feel intellectually violated.  Chrono Mysteries avoids this by anchoring every twist in a single, non-negotiable axiom: “Every temporal intervention must be traced backward to an intentional act.”  That rule is introduced early, when Director Orion shows Ava a hologram of a timeline that branches seven times but always converges at the same node.  The visual grammar is precise: each branch is a colored thread that thickens into red only when an observer’s intent is proven.  Viewers learn the rule so quickly that they start spotting deviations during re-watches, creating a sense of intellectual “play” akin to solving a jigsaw with mismatched pieces.\n\nThe payoff mechanism is surgical.  Instead of the usual “this happened because of that” reveal, each season ends with a “this did not happen because of that,” a correction that ripples back through previous seasons.  For example, in Season 1 the missing researcher Leo Quantum is blamed for the paradox; in Season 2 it’s revealed that Leo never existed in the first place—his role was a deliberate red herring designed to keep Ava’s investigation on the wrong track.  The revelation doesn’t contradict earlier episodes; it rewrites them, like deleting a typo from a living text.  Because the correction is always linked to human choice rather than random accident, viewers feel agency restored to the narrative.  This logical consistency breeds what psychologists term “trans-temporal empathy”—the feeling that our own choices matter within the story, even when they don’t in real life.\n\n3.  Dr. Elena Sharp: The Character Whose Anxiety Is Plot Fuel  \nWhile Maya’s arc provides the emotional spine, it’s Dr. Elena Sharp who turns ethical tension into tactile dread.  Introduced midway through Season 1 as a rival temporal analyst, Elena’s entire identity is built on a single paradox: she remembers the day her own sister disappeared from history, yet her family claims she never had a sister.  The show withholds any exposition—no voice-over, no flashback—and lets the contradiction hang like a noose around her neck.  Every time she appears, the camera lingers on her left hand: her fingers always trace invisible loops in the air, an unconscious habit of rewriting the narrative around her.\n\nBecause the paradox is symmetrical, viewers can’t pin blame on anyone else.  That cognitive dissonance escalates when we discover that Ava is actually her half-sister by temporal displacement—Elena was born in a timeline that had not yet been created by Maya’s eventual erasure.  The reveal is staged in a single breath-taking take: Ava watches a projection of Elena’s childhood, sees her mother with a birthmark that matches her own, and realizes she’s been talking to her dead self the entire time.  The emotional beat is so visceral that it bypasses typical advice columns and becomes a textbook case of how to make guilt feel like physics rather than melodrama.\n\nElena’s arc is engineered for long-term engagement because she embodies what philosophers call “the other-self paradox.”  Her presence guarantees that every victory for Ava feels qualified—like climbing a ladder that was built backward.  Youth viewers, who are already fluent in identity fluidity on platforms like TikTok, recognize this as a mirror of their own search for belonging in an impermanent digital world.  As a result, Elena is one of the few characters whose popularity has grown over the series, rather than plateauing.\n\n4.  Atmosphere as Narrative Prop: Set Design as Plotting Tool  \nChrono Mysteries is filmed on practical sets because the writers insist that “the cost of the show is the cost of the past.”  Instead of CGI warps and shimmering wormholes, the production team constructs physical spaces that are deliberately broken in time.  The most obvious is the Chronos Lab: a single room where walls are painted with archival photos of historical disasters, but every new season reveals a new disaster that hasn’t occurred yet.  Viewers who return to the lab in subsequent seasons feel like they’re exploring an archaeological dig—each new layer proving that the past can be unmade.  \n\nEven costume design participates in the narrative.  Dr. Ava always wears a white lab coat that is digitally printed with the timestamp of the current episode.  But because she can jump forward, the coat shows future dates that are yet to be written.  The garment therefore functions as a living diary that refuses to stay present, forcing viewers to question whether they, too, are wearing future regrets.\n\nSound design plays an equal role.  Every temporal jump is accompanied by a sub-bass frequency that increases in pitch until it reaches an audible rupture—the moment history is rewritten.  The frequency is precisely 432 Hz, the musical note historically linked to “perfect tuning,” which primes viewers to feel a dissonant perfection.  Over time, the brain associates that frequency with anticipation, turning the auditory signal into a conditioned reflex that heightens suspense before the visual cue even lands.\n\nColor theory is equally ruthless.  The show’s palette is locked to a single hex code (#4A6A43) representing “the color of erased memory.”  During jumps the hue shifts toward ultraviolet, but the camera lingers long enough for viewers to mentally re-adjust, creating a visceral after-image of temporal vertigo.  For young audiences who process visual information faster than adults, this chromatic trick converts boredom into nausea—exactly the emotional high a series needs to hold their attention.\n\n5.  Why the Formula Works for Teenage Minds  \nChrono Mysteries is ultimately a story about how identity fractures when the present is no longer stable.  Teen viewers already experience this in real time: social media profiles can be deleted overnight, friendships can move from viral to vanished, and personal timelines can be rewritten with a simple click.  The show externalizes this uncertainty into a hyper-real format that is both thrilling and familiar.\n\nThe repetition mechanism—episodes that restart, timelines that reset, characters that loop—resonates with adolescents who are learning to self-regulate their own cycles of social media usage.  Studies show that binge-watching creates a feedback loop where viewers unconsciously model their own behavior on the characters’ habit patterns.  By presenting a universe in which the characters can (and do) retry, fail, retry, the show offers a therapeutic narrative arc: “I tried, I failed, I tried again”—a mantra that mirrors the real-life experimentation youth often practice online.\n\nYet the show never romanticizes procrastination.  Each reset carries ethical stakes that force characters—and viewers—to confront opportunity cost.  This duality is what keeps the long-form format from feeling merely addictive.  Instead, it becomes a guided meditation on consequence: if every action can be undone, what becomes of responsibility?  The questions are existential, but the delivery is designed to sit quietly beside daily life—no need to binge until collapse; simply finish one season, then watch again, and notice how some clues you missed the first time feel brand new.\n\n6.  Practical Advice for Future Long-Form Creators  \n1.  Hook with a paradox that is visually self-evident.  Avoid exposition-heavy explanations; let the contradiction stare the audience in the face.  \n2.  Build a one-rule framework that allows every twist to be internally consistent.  Young viewers love intellectual tightness more than they admit.  \n3.  Give secondary characters emotional stakes that are symmetrically flawed.  Symmetry is easier to watch than asymmetry because it feels fair.  \n4.  Make the physical environment participate in the plot.  Practical sets that decay over time create a tactile memory that digital effects cannot replicate.  \n5.  Leave some answers to the future.  Chrono Mysteries ends each season with a question that future episodes must answer.  This engineered suspense is cheap to maintain and addictive to pursue.\n\n7.  Conclusion: A Time Traveler’s Guide to Being Stuck  \nChrono Mysteries succeeds where many other series fail by refusing to finish its journey.  The final episode of Season 3 ends on a loop that could continue indefinitely, suggesting that the conspiracy is only partially solved—like a TikTok trend that keeps going viral.  Teen viewers, whose lives are already curated by infinite scroll, recognize this as a mirror of their own temporal vertigo.  They return because the show offers a rare kind of companionship: a story that outlasts the binge, that rewards re-watching, that lets them feel like co-authors of an unfinished future.\n\nIn a culture where attention spans are measured in seconds, the show’s genius lies in manufacturing a paradox that refuses to obey the clock.  Like the characters who jump through time, viewers are invited to keep trying—just one more episode, one more reset, one more version of themselves—to prove that the narrative matters.  And in that process, they find themselves trapped in a moment that is not quite present, not quite past, but unmistakably theirs.'
    
    gt_text='[\n  {\n    "name": "开场悬念设置与时间悖论吸引力",\n    "criteria_description": "评估文章是否深入分析剧集如何利用时间悖论制造初始悬念，具体检查是否提及“Temporal Twins”设定如何引发观众好奇、是否分析开篇情节的节奏与信息释放策略，以及这些元素如何迅速建立观看黏性。",\n    "1-2": "文章未提及“Temporal Twins”设定或其对观众好奇心的激发，缺乏对开篇情节节奏与信息释放策略的分析，未能说明这些元素如何建立观看黏性。",\n    "3-4": "简要提及“Temporal Twins”设定，但仅描述基本情节，未深入分析其悬念机制；部分涉及开篇信息释放，但未分析节奏或具体吸引力策略，观看黏性的论证薄弱。",\n    "5-6": "明确分析“Temporal Twins”设定如何通过身份错位和时间能力制造初始悬念，并解释其引发观众好奇的逻辑；能识别开篇情节的信息释放节点，但未完整刻画节奏变化与悬念递进的关系，对观看黏性的分析有一定依据但不够深入。",\n    "7-8": "详细拆解“Temporal Twins”设定在悬念构建中的作用，包括能力差异、动机冲突和伦理困境如何强化观众好奇心；系统分析开篇情节的节奏与信息释放策略（如关键线索间隔、误导性提示），并能结合观众心理说明黏性建立机制。",\n    "9-10": "深度揭示“Temporal Twins”设定不仅是情节工具，更是时间悖论主题的核心体现，通过对比姐妹视角、能力限制与协作障碍等多维度悬念设计；精确还原开篇情节的叙事节奏（如每分钟信息密度、转折点布局）与信息释放策略（如伏笔回收周期、隐藏线索），并论证这些元素如何协同作用，形成多层次观看黏性。",\n    "11-12": "创新性解读“Temporal Twins”设定背后的哲学隐喻及其对观众认知框架的挑战，将其与剧集整体悬念体系关联；开创性地分析开篇情节的“动态节奏”（如根据观众反馈调整叙事速度）与“嵌套式信息释放”（如主线索下隐藏次级谜团），并量化评估其对不同观众群体的黏性影响。"\n  },\n  {\n    "name": "核心谜团逐步展开的逻辑连贯性",\n    "criteria_description": "考察文章对剧情渐进式揭示的分析深度，重点在于是否能识别出各季间谜题的嵌套关系、关键线索（如Leo Quantum失踪）的埋设与回收，以及整体叙事是否存在自洽的时间因果链条。",\n    "1-2": "未能识别出剧情中关键线索的埋设与回收，对谜题嵌套关系缺乏基本分析，时间因果链条存在明显断裂或矛盾。未提及Leo Quantum失踪等核心事件，无法建立各季间的联系。",\n    "3-4": "能识别出部分关键线索（如Leo Quantum失踪）的埋设，但对回收过程的分析浅显，仅简单罗列线索出现节点。初步尝试解释谜题嵌套关系，但逻辑链条不完整，存在时间线跳跃或因果倒置现象。",\n    "5-6": "明确分析出多季间关键线索的埋设与回收路径，能追踪Leo Quantum失踪事件在不同时期的演变及其与其他谜题的关联。对谜题嵌套结构有清晰描述，但未深入分析时间因果的自洽性，某些环节仍依赖推测而非直接证据。",\n    "7-8": "详细拆解了核心谜团在各季的展开过程，精准定位Leo Quantum失踪等线索的首次埋设、阶段性揭示及最终收束点。论证了谜题之间的递进式嵌套关系，并验证了时间因果链条在多数场景下的自洽性，仅个别细节需进一步澄清。",\n    "9-10": "系统性地还原了所有关键线索（包括Leo Quantum失踪）的完整生命周期，展示其在不同时间线中的交叉影响。深入剖析谜题嵌套机制，揭示多层悬念如何逐层解扣且相互支撑。严密验证整体叙事的时间因果闭环，无逻辑漏洞或矛盾。",\n    "11-12": "不仅全面覆盖线索埋设与回收，还挖掘出隐藏伏笔（如角色对话中的双关语、背景道具符号），并论证其对后续剧情的关键作用。提出创新解读，揭示谜题嵌套背后的深层设计原理（如镜像结构、递归模式）。时间因果链条呈现高度自洽，且能反向推演关键节点以验证一致性。",\n    "13-14": "构建出动态的谜团演进模型，量化分析各线索在不同时期的“可见度”与“影响力”，并模拟观众认知变化曲线。超越线性叙述，从非线性视角（如量子纠缠隐喻）重新诠释谜题嵌套关系。时间因果链条被转化为多维网络，每个决策点均有多重验证路径，实现叙事的拓扑自洽。"\n  },\n  {\n    "name": "Dr. Elena Sharp角色发展对张力的贡献",\n    "criteria_description": "虽然材料中主角为Ava和Maya，但若文中引入Dr. Elena Sharp作为虚构补充角色，需评估其角色弧光是否合理增强剧集的道德困境或情感冲突；若未提及则判定为偏离主题。",\n    "1-2": "文中未提及Dr. Elena Sharp，或其角色仅作为背景板出现，无具体行为、动机或发展，对道德困境或情感冲突毫无影响。",\n    "3-4": "Dr. Elena Sharp被提及但角色弧光不完整，其行为与情节关联较弱，虽涉及时间伦理议题（如实验、干预历史），但未深入探讨其选择带来的后果；与Ava/Maya的互动仅限于表面合作或信息传递，未能引发显著情感张力或道德争议。",\n    "5-6": "Dr. Elena Sharp有明确的行为动机和阶段性转变，其行动直接推动关键剧情节点（如揭露某时间异常、发起争议性实验），并引发Ava/Maya对其立场的质疑；角色设定与时间悖论主题紧密结合，通过其决策展现“科学责任 vs. 个人野心”的冲突，形成可辨识的道德困境，但情感层面刻画仍显单薄。",\n    "7-8": "Dr. Elena Sharp的角色发展具有清晰的内在逻辑与外部驱动，其从初始立场（如中立研究者）到后期转变（如主动干预时间线以“修正”历史）的过程合理且层层递进；其行为不仅制造实质性剧情转折（如触发关键时间事件），还通过其与Ava/Maya的价值观对立深化情感冲突（如信任破裂、伦理辩论），使观众在“理解其动机”与“质疑其手段”之间持续挣扎。",\n    "9-10": "Dr. Elena Sharp的角色弧光高度复杂且自洽，其发展轨迹贯穿多集并与其他角色（Ava/Maya/Director Orion）形成动态张力网络；其每个重大决策均基于前序经历并引发连锁反应（如早期隐瞒数据导致后期信任危机），同时其内心挣扎（如对时间伦理的矛盾态度）被细腻呈现；角色不仅是道德困境的载体，更是剧集核心主题的具象化——通过其对“时间可被操控”的信念及其代价，将抽象哲学命题转化为具体人物悲剧，极大增强剧集的深度与沉浸感。"\n  },\n  {\n    "name": "氛围与布景设计的叙事关联分析",\n    "criteria_description": "判断文章能否将视觉元素（如实验室、不同时代场景）与剧情发展结合，例如通过色调变化反映时间线混乱，或通过道具细节暗示隐藏线索，体现环境设计对沉浸感的提升作用。",\n    "1-2": "文章基本未提及视觉元素与剧情发展的关联，布景和氛围设计仅作为背景存在，缺乏对色调、道具等细节的分析，未能体现环境如何增强沉浸感或推动叙事。",\n    "3-4": "文章提及少量视觉元素（如实验室或时代场景），但分析较为浅显，仅简单说明场景变化对应时间线，未深入探讨色调、道具细节与情节发展的互动关系，缺乏对隐藏线索或情感氛围的具体论证。",\n    "5-6": "文章能够识别关键视觉元素（如实验室布局、不同时代装饰风格），并初步分析其与剧情的关联，例如指出某色调暗示时间混乱或某道具反映角色心理，但整体分析仍显零散，未能系统展示环境设计如何持续影响叙事节奏和观众沉浸感。",\n    "7-8": "文章清晰分析多个视觉元素（如色彩渐变、空间结构、道具符号）与剧情发展的动态关系，能结合具体例子说明环境设计如何随时间线推进而变化，并解释其对悬念构建、线索暗示的作用，展现出布景与叙事的有机关联，但个别细节分析深度略有不足。",\n    "9-10": "文章全面且系统地剖析视觉元素与叙事的深层互动，涵盖色调变化、空间转换、道具细节等多个维度，每个分析点均与剧情关键节点（如悖论揭示、角色转变）紧密结合，明确指出环境设计如何通过细节暗示隐藏线索、强化情感张力，并提升观众的沉浸体验。",\n    "11-12": "文章不仅详尽分析视觉元素与剧情的显性关联，还能挖掘其隐喻意义（如实验室设备象征时间操控、色彩对比映射伦理冲突），并通过连贯的逻辑链展示环境设计如何贯穿全剧，成为叙事驱动力之一；所有分析均有具体场景支撑，且具备创新性解读，充分体现布景与氛围在悬疑构建和主题表达中的核心作用。",\n    "13-14": "文章对环境设计的分析达到卓越水平，不仅能精准识别并拆解每一处视觉细节（如光影变化、家具年代、符号化道具）与剧情发展的即时及长期关联，还能将其整合为一套完整的“视觉叙事语法”，系统阐述其如何通过多感官刺激（如冷色调压抑感、杂乱布景暗示混乱）持续引导观众情绪、强化谜题逻辑，并深化主题表达，实现环境设计与叙事结构的完美融合。"\n  },\n  {\n    "name": "姐妹动态在跨时空调查中的表现力",\n    "criteria_description": "衡量文章是否突出Ava和Maya协作的独特模式——一个在现世分析，另一个穿梭历史——并分析这种双重视角如何制造悬念（如信息不对称）及推动主线进展。",\n    "1-2": "文章未提及Ava和Maya的协作模式，或仅简单提及她们在不同时空进行调查，没有分析双重视角如何影响剧情。缺乏对信息不对称、悬念制造及主线推动的具体讨论。",\n    "3-4": "文章提到Ava和Maya分别在现世和历史中调查，但协作模式的描述较为泛泛，缺少细节。有初步分析双重视角的影响，如悬念的产生，但未能深入探讨这种模式如何具体推动情节发展，分析较为表面。",\n    "5-6": "文章明确描述了Ava和Maya的协作模式，包括她们在各自时空中的角色分工。分析了双重视角如何通过信息不对称制造悬念，并举例说明这些信息差如何引发观众的好奇心。能指出一些具体情节展示这种协作如何推动主线进展，但部分例子不够详尽或逻辑链条不完整。",\n    "7-8": "文章详细描述了Ava和Maya的协作模式，突出她们在现世和历史中的互补角色，并通过多个具体场景展示了双重视角的动态互动。深入分析信息不对称如何层层递进地制造悬念，并系统性地分析这些悬念如何与主线的关键节点相连接。分析涵盖了协作模式对角色成长和剧情转折的影响，但某些细节可以进一步深挖。",\n    "9-10": "文章全面且细致地描绘了Ava和Maya的协作模式，不仅涵盖角色分工，还探讨了她们的沟通方式和决策过程。通过丰富的实例和严密的逻辑，剖析了双重视角如何创造复杂的信息不对称，以及这些不对称如何在不同时间线之间形成张力，推动主线层层推进。分析展示了协作模式如何成为剧情发展的核心驱动力，并揭示了其对角色动机和伦理困境的深层影响。",\n    "11-12": "文章卓越地呈现了Ava和Maya的协作模式，将其视为叙事结构的核心创新点。不仅详细分析了双重视角的信息不对称机制，还揭示了这种不对称如何在微观（单个场景）和宏观（整个系列）层面产生多层次的悬念，并与主线的每一个关键转折点精密对接。分析深入挖掘了协作模式对角色心理、道德选择及观众情感投入的动态影响，展现出极高的洞察力和原创性。"\n  },\n  {\n    "name": "伦理困境对长期观众参与的影响",\n    "criteria_description": "关注文章是否挖掘时间旅行引发的道德问题（如篡改历史后果），并论证这些议题如何通过角色选择（如Maya的抉择）激发观众持续思考，从而延长剧集生命周期。",\n    "1-2": "文章未提及时间旅行引发的道德问题，或仅简单带过角色选择，如Maya的抉择，未能挖掘这些议题对观众长期参与的影响。",\n    "3-4": "文章提及了部分时间旅行的伦理困境，如篡改历史后果，但缺乏深入分析，角色选择的道德探讨较为浅显，未能有效论证这些议题如何激发观众持续思考。",\n    "5-6": "文章能够识别并分析时间旅行中的主要道德问题，如Maya和Ava在不同情境下的抉择，探讨了这些选择对历史的潜在影响，但未能充分展开这些议题如何具体延长剧集生命周期。",\n    "7-8": "文章深入分析了时间旅行的伦理困境，并通过具体角色（如Maya的抉择）展示了这些道德问题的复杂性，论证了这些议题如何通过引发观众情感共鸣和讨论，促进长期观众参与。",\n    "9-10": "文章全面挖掘了时间旅行引发的多种道德问题，包括不同角色的抉择及其历史后果，详细阐述了这些议题如何激发观众的持续思考和讨论，明确指出其对剧集生命周期的延长作用，并通过实例（如Maya的关键选择）加以佐证。",\n    "11-12": "文章不仅详尽分析了时间旅行的伦理困境及其对观众的影响，还通过多个角色的选择和剧情发展，系统性地论证了这些道德问题如何在剧集的不同阶段引导观众持续关注，提出创新性观点，展示出卓越的深度和广度。"\n  },\n  {\n    "name": "细节精确性与伏笔系统的有效性",\n    "criteria_description": "检查文章是否强调剧中细节（如对话重复、物品出现位置）如何构成解谜网络，并分析这些微小但关键的线索为何能维持粉丝社群的讨论热度与二刷率。",\n    "1-2": "文章未识别或分析任何具体细节（如对话、物品、场景重现），未能建立细节与解谜网络之间的联系；未提及粉丝社群讨论热度或二刷率相关现象，缺乏对剧中伏笔系统的基本认知。",\n    "3-4": "提及少量剧中细节但未能系统归纳其作用，仅列举个别重复出现元素（如某句台词或道具）而未展示其在不同情节节点中的逻辑关联；对粉丝讨论或二刷行为有所提及但停留在表面描述，未解释细节如何驱动这些行为。",\n    "5-6": "能够识别多个关键细节并初步构建其与剧情发展的关系网络，例如指出某物品在不同时间线的出现位置构成线索链；能引用实际案例说明这些细节如何引发粉丝推测或促进二刷发现新信息，但分析深度有限，未全面覆盖所有重要伏笔。",\n    "7-8": "系统性地梳理剧中核心细节（对话、物品、场景等）并明确展示它们如何在多集或多季中形成闭环式解谜结构，如通过时间线对比揭示隐藏因果关系；深入分析细节设计为何激发社群持续讨论（如矛盾点、多重解读空间），并提供数据或实例支持其对二刷率的提升效果。",\n    "9-10": "完整绘制剧中细节构成的多维伏笔网络图，精确标注每项微小线索在时间线、角色决策、世界观设定中的交叉验证作用，体现“每一帧皆有用”的编剧原则；结合观众反馈（如论坛热帖、视频解析数量）、平台播放行为（如回放率、分集停留时长）等可量化证据，论证该伏笔系统不仅维持长期热度，更成为剧集核心魅力所在。",\n    "11-12": "超越单纯细节罗列，创造性地提出该剧伏笔系统具有“动态演化”特性——早期看似无关的细节随剧情推进被赋予新意义，形成递归式解谜体验；通过对比其他同类型作品，证明此类设计显著提升粉丝参与度（如UGC内容产出量、线下活动频率），并推动形成独特的社群文化（如术语体系、解码仪式），最终实现商业价值与艺术成就的双重突破。"\n  },\n  {\n    "name": "演讲体语言风格与青年受众适配度",\n    "criteria_description": "由于要求是\\"youth lecture\\"场合，需评估文章是否采用激励性、互动性语言（如反问句、第二人称），而非纯学术腔调，确保符合青少年听众的理解习惯与兴趣点。",\n    "1-2": "文章主要采用学术性或叙述性语言，缺乏激励性和互动性表达。几乎没有使用反问句或第二人称，未能激发青少年读者的兴趣和参与感。",\n    "3-4": "文章部分段落尝试使用激励性语言，如偶尔出现的反问句或第二人称代词，但整体上仍偏向于正式和客观的描述，互动性不强。青少年读者可能会觉得部分内容难以产生共鸣。",\n    "5-6": "文章在多数段落中成功运用激励性语言，如频繁使用反问句和第二人称（“你”、“你们”），增强与青少年读者的互动性。语言风格较为轻松，能够吸引青少年的注意力，但在某些技术性分析部分仍有改进空间。",\n    "7-8": "文章广泛采用激励性、互动性语言，包括大量反问句、第二人称代词和生动的比喻，使复杂的时间悖论和角色发展易于理解且引人入胜。整体语言风格高度契合青少年听众的理解习惯和兴趣点，仅有个别地方可进一步简化。",\n    "9-10": "文章全面采用激励性和互动性语言，通过丰富的反问句、第二人称、情景设定和情感共鸣，使内容极具吸引力。语言风格不仅贴合青少年的认知水平，还极大地激发了他们的好奇心和参与感，几乎在所有方面都达到了卓越水平。",\n    "11-12": "文章在激励性和互动性方面表现出色，不仅使用了大量反问句和第二人称，还巧妙设计了多个互动环节（如引导式提问、假设场景），让青少年读者仿佛置身其中。语言风格既生动又富有创意，完全符合“youth lecture”场合的要求，并在某些方面超越预期。",\n    "13-14": "文章完美融合了激励性和互动性语言，通过高度情境化的描述、多层次的互动设计（如虚拟对话、角色代入）和极其贴近青少年生活的比喻，使复杂的情节和角色关系变得极为清晰和有趣。语言风格不仅完全适配青少年受众，还创造了一种独特的演讲体验，堪称典范。"\n  }\n]'
    
    a=reward_fn.score_pair(pred_text, gt_text)
    print(a)