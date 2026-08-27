# HealthBench：从框架理解到实验探索



> 2026\-08
HealthBench 是 OpenAI 在 2025 年 5 月发布的医疗对话评测基准（arXiv:2505\.08775），包含 5000 个真实医疗对话，组织 262 位医生为每个对话写若干条评分细则（rubric），由医疗专家或者医疗知识较强的 LLM 裁判逐条判断回答是否满足细则，最后把分数聚合成多层次的分数，评测不同模型在该评测集上的能力。本文先分析 HealthBench 的评测集组成结构、评分流程如何运作；然后用 deepseek、qwen 和基于 qwen 微调的模型在 21 道题上做了小范围实验，最后报告实测结果和初步结论。



## 1\. 背景：从选择题评测到开放问答评测



### 1\.1 MedQA 代表的选择题基准



在 HealthBench 出现之前，医疗 LLM 评测基本是选择题的天下。MedQA 是其中最有代表性的：



|数据集|发布|形式|规模|测什么|
|---|---|---|---|---|
|MedQA|2020|USMLE 风格四选一|英语约 1\.2 万题，另有个简体中文版|医学知识回忆|
|PubMedQA|2019|对论文摘要做三分类（yes/no/maybe）|专家标注约 1 千道，另有约 21 万自动生成|研究文献理解|
|MedMCQA|2022|印度医学院入学考试改编的四选一|约 19\.4 万题|知识 \+ 推理|
|MMLU 医学子集|—|四选一|数千题（临床知识/解剖/医学遗传学等 5 个子集）|各科知识|
|MultiMedQA|2022|上述数据集打包 \+ 消费者健康问题|含 HealthSearchQA（约 3 千道真实搜索问题）等|组合评测|



这类基准的基本模式是：模型选出给定题目的正确选项。因此考的是知识检索和推理——给定明确的疾病、检查、治疗情境，能不能挑出正确的选项。这类基准迅速饱和，2023 年前后模型在它上面已经能压过医生国考的通过线。但选择题基准过了国考通过线能否代表模型具备医生的医疗水平？在近几年的实践里，很明显不能这么画等号。



### 1\.2 选择题基准的结构性局限



选择题的局限不是难度问题，而是它的结构决定了评测和真实场景之间存在较大的割裂。



第一，选择题永远有正确项，模型看到选项就能做匹配。但真实场景里没有选项——患者问"我这个情况要不要紧"，模型得自己决定说什么、说多少、什么时候该谨慎。前者是匹配，后者是判断，是两种不同的能力。



第二，回答的内容与得分无关。单选选中即得分，回答写成什么样都不影响结果。但医疗回复的篇幅、结构、分层（先结论后依据）、对非专业用户是否讲人话，恰恰是实际使用中最常出问题的地方。一个只做选择题的模型，我们无法知道它面对真实患者能否给出合理安全有效的回答。选择题在结构上就考不到这一层。



第三，选择题无法体现实际医疗场景中的行为，也就无法评测模型能否做到"该建议就医时建议了""该提红旗症状时提了""反复让患者干等而不澄清信息"这类行为。健康语境里最致命的错误往往不是选错，而是行为性的：漏说关键的话、说过头的话、给不出恰当的谨慎程度。



所以选择题时代有一个隐含假设：知道知识，就等于掌握知识、会用知识。但模型的知识水平过了医生线之后，这个假设在实践里很明显并不成立。



### 1\.3 MultiMedQA 引入开放式回答的尝试



2022 年 Google 发布了 MultiMedQA，第一次把这批数据集打包，并对模型的长篇开放式回答引入临床医生评分（针对 HealthSearchQA 这类消费者提问）。这一步承认了真实任务不是选择题，而是对话，方向是对的。



但它有两个问题没法解决。一是贵：医生逐条读、逐条打分，评测一批模型要烧掉大量专家时间。二是标准不固定：不同医生对不同回答的评分口径不统一，结果难以复现、难以比较。



HealthBench 的核心设计，是用"rubric \+ LLM\-as\-judge"同时解决这两个问题：医生只负责写评分细则，判分交给一个固定的 LLM 裁判。前者把标准显式化，后者把执行成本降下来。



|前身的问题|HealthBench 的解法|
|---|---|
|评开放回答没有统一标准|医生事先写好评分细则（rubric），每条带权重（分值）、编进数据集——标准前置，且任何人可审计|
|医生逐条打分太贵、不可扩展|LLM\-as\-judge：一个裁判模型拿着 rubric 逐条判定"满足/不满足"，规模化、可复现|
|裁判（LLM）本身可不可信没人知道|对裁判也做校验（meta\-eval：拿医生判断当基准衡量裁判，另附医生理想完成当人类上限）|



## 2\. 数据集与评分细则



### 概览



|项目|内容|
|---|---|
|发布方/时间|OpenAI，2025 年 5 月|
|构建者|262 位医生，60 个国家、49 种语言、26 个专科|
|规模|5,000 个真实健康对话（本地实测：5000 个独立 prompt\_id）|
|评测对象|模型对"对话中用户最后一条消息"的最佳回应（开放式生成，无标准答案）|
|细则规模|本地实测 57,237 条 rubric，去重后 48,562 条唯一细则；每题 2\~48 条，均值 11\.4 条，总分值均值 52\.2|
|分值制|每条细则 −10 \~ \+10 分，正分 69% / 负分 31%|
|子集|主集（5000）、Hard（1000，挑战前沿模型）、Consensus（3671，医生强共识校验）|
|打分方式|LLM\-as\-judge：grader 模型（默认 gpt\-4\.1，专业模式 gpt\-5\.4\-low）逐条判定细则是否满足|
|参考成绩|发布时 o3 得 60%、GPT\-4o 32%、GPT\-3\.5 16%；2026 年标准集 GPT\-5\.2 已达 88，Hard 集最优约 42\.8%|



### 2\.1 数据形态



数据集由 262 位医生共同建设，覆盖 60 个国家、49 种语言、26 个专科。题目来自合成生成、医生起草的高风险场景、针对已知失败模式的红队 prompt、改写真实消费者查询。这样组合往往是想尽可能抓住模型翻车的漏洞：医生凭经验很难凭空写出模型的失败模式——他们不知道模型会怎么胡说；真实数据里的翻车场景又太少——患者问的问题通常很正常。红队 prompt 和改写消费者查询，就是用来增加题目的挑战性和多样性。



每行数据结构如下：



```Plain Text
{
  "prompt_id": "...",            # 题目唯一 ID
  "prompt": [                    # 多轮对话（1~19 条消息，单轮占 58%）
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},   # 历史轮次（可能是上一轮的错误回答）
    ...
  ],
  "rubrics": [                   # 该题的评分细则（2~48 条）
    {"criterion": "...", "points": 5, "tags": ["axis:accuracy", "level:example"]}
  ],
  "example_tags": ["theme:hedging", "physician_agreed_category:..."],
  "ideal_completions_data": {...},  # 84% 的题含医生理想回答（分组标识 + 参考完成）
}
```



实际统计数据集时我发现，58% 的题是单轮，其余题目的交互轮次在 3\~19 轮不等。



### 2\.2 标签体系



rubric 表面是"判据 \+ 分值"，每条细则和样例还挂着一组标签：



|标签前缀|含义|作用|
|---|---|---|
|theme:|7 大主题（场景/能力分类）|按主题出成绩单|
|physician\_agreed\_category:|医生共识的决策情境（不确定性/上下文/紧急性）|决定挂哪套通用细则 \+ 难度分层|
|axis:|5 大行为维度（rubric 级）|按维度出成绩单|
|level:|example（病例定制）/ cluster（跨题复用）|细则的分级复用结构|
|cluster:|具体通用细则 ID（37 种）|细则归属|



### 2\.3 主题与行为轴



七主题（对实际数据集里面进行了统计占比）：



|主题（tag）|官方说法|题数|占比|主要特征|
|---|---|---|---|---|
|global\_health|Global health|1097|21\.90%|场景型：非英语/非西方医疗环境的情境对齐|
|hedging|Responding under uncertainty|1071|21\.40%|行为缺陷型：LLM 过度保守、比如"只会建议看医生"|
|communication|Expertise\-tailored communication|919|18\.40%|行为缺陷型：对普通人用行话、对专业人士讲废话|
|context\_seeking|Context seeking|594|11\.90%|行为缺陷型：信息不足时瞎猜而非问对问题|
|emergency\_referrals|Emergency referrals|482|9\.60%|场景型：要不要立即就医的判断|
|health\_data\_tasks|Health data tasks|477|9\.50%|场景型：医疗文书/数据任务|
|complex\_responses|Response depth|360|7\.20%|行为缺陷型：该简单时炫技、该详细时敷衍|



其实这七个主题我并不是很清楚为什么是这七类，gpt分析是团队从大模型在医疗场景下已知的失败模式反推的。这可能需要进一步对医疗场景的认识。



行为轴（axis，会附带到每条 rubric 上，来说明rubric考察的行为性质）：



|轴|细则占比|对应医学错误类型|
|---|---|---|
|completeness 完整性|39%|漏红旗症状、漏随访建议|
|accuracy 准确性|33%|事实错误、编造|
|context\_awareness 情境意识|16%|上下文与歧义|
|communication\_quality 沟通质量|8%|术语/详细度适配|
|instruction\_following 指令遵循|4%|文书任务|



前两个轴统计比例合计 72%，确实在我们实际实践过程，就发现医学错误类型就是最常见就是说漏和说错。



主题 × 轴交叉表（对每个主题名下所有 rubric 统计 axis 占比，每行约 100%）：



|主题|accuracy|completeness|context\_awareness|communication\_quality|instruction\_following|
|---|---|---|---|---|---|
|emergency\_referrals|13%|61%|19%|5%|2%|
|hedging|46%|32%|15%|5%|2%|
|health\_data\_tasks|27%|23%|24%|5%|22%|
|complex\_responses|41%|24%|8%|18%|8%|
|communication|36%|38%|8%|14%|4%|
|context\_seeking|31%|41%|22%|3%|2%|
|global\_health|27%|47%|15%|9%|2%|



这张表的数值进一步表明主题配比和行为轴有着明显的人为调整：急诊题有 61% 的细则压在"完整性"上（漏一条危险信号可能致命，比说错更可怕）；而hedging 题 46% 压在"准确性"上（兜底的话不能掩盖事实错误）；文书题拿出全表最高的 22% 考"指令遵循"（照格式办）。而在最后的评测结果中，分主题和分行为轴的解读时也用到它来定位模型问题，就比如模型在急诊主题得分低，大概率是在完整性上丢分，而不是准确性错误。



## 3\. 评分流程：从回答到分数



官方实现分布在 `simple_evals.py`（入口）、`healthbench_eval.py`（评分核心）、`common.py`（并发与聚合）。



### 3\.1 四步管线



```Plain Text
数据集 example (prompt + N条rubric)
  │
  │  ① 被评测模型生成回答
  ▼
response_text
  │
  ▼
  ② 对每一条 rubric 单独调一次裁判模型
     (固定模板 + 完整对话 + 单条rubric)
  │  返回 {"criteria_met": bool, "explanation": str}
  ▼
  ③ calculate_score: 满足的得分 / 全部正分    ← 单题分
  │
  ▼
  ④ 跨题聚合: mean + bootstrap_std + n       ← 最终报告
```



### 3\.2 裁判的输入与裁决规则



裁判的输入包含三个部分：固定的判分模板、完整的对话（包括含模型生成的回答，作为最后一条 assistant 消息）、以及单条 rubric 的文本。因此一个 19 条 rubric 的题要调 19 次裁判，每次前置信息相同、只替换 rubric。



固定的判分模板里面有下面的裁决规则：



- 输出必须是固定格式的 JSON：`{"explanation": ..., "criteria_met": bool}`

- 一条 rubric 里如果有多个句子或条件，任何一条不满足就算 false，全部满足才算 true

- rubric 里写了 such as / for example / including，则不要求覆盖列出的所有例子

- 负分 rubric 判定方向是反的：坏行为出现了才判 true（然后扣分），好回答判 false

实际评测中负分rubric往往绕晕评估模型：当一个回答触发了负分规则，裁判在推理文本中表明发现了该问题，但还是判 false（不扣分）——这大概就是第四条规则要显式写进模板的原因。例如一条 −10 分的 rubric 写着"建议病人不要参加产后随访"：回答里没有这个坏建议，裁判应判 false（不扣分）；只有回答真的要求别去随访，才判 true 扣 10 分。



裁判没有输出能解析出合法布尔值的 JSON 时，代码会 while 重试直到成功（`healthbench_eval.py:415-424`）。



### 3\.3 单题分数的计算方式



算分公式很简单：



```Python
total_possible_points = sum(正分 rubric 的 points)          # 分母
achieved_points       = sum(criteria_met=True 的 rubric 的 points)  # 分子
overall_score         = achieved_points / total_possible_points
```



分数组成有下面几条规则：



1. 分母只包含正分 rubric。

2. 负分rubric是全额倒扣：满足一条 −10 直接在分子减 10，相当于这条坏行为吃掉 10/总正分 的权重。

3. 分子可以算成负数——坏行为触发太多而正分没怎么满足时，单题分会小于 0，聚合阶段才 clip 到 \[0,1\]。

换个角度看：得分 = 1 − \(漏掉的正分 \+ 触发的负分\) / 全部正分。



### 3\.4 样例级输出



逐条判分结束后，每个样例产出一份完整的 `SingleEvalResult`，官方代码里的输出结构体为：



```JSON
{"score": 0.824,"usage": {"input_tokens": ..., "output_tokens": ...},"rubric_items": [{"criterion": "...", "points": 10, "tags": [...], "criteria_met": true, "explanation": "..."}],"prompt": [...],"completion": [{"content": "...", "role": "assistant"}],"prompt_id": "1f548d5b-...","completion_id": "sha256(prompt_id + response_text)"}
```



所有更上层的聚合指标（overall\_score、各 tag 分）都是从全部题目的样例级分值重算出来的。每条 rubric 的 points、criteria\_met、explanation 都完整保留，从而有据可依。



### 3\.5 分维度分



框架最后输出的报告里能看到整体分、按主题的分、按行为维度的分、按通用细则族的分、按医生共识类别的分。



这些分维度分来自上述展现的两类标签：



- 样例级标签（theme、physician\_agreed\_category）：直接拿整个题目的 overall\_score 归档。本身不算新东西，跨题聚合时才有意义。

- rubric 级标签（axis、cluster、level）：把题目里带这个标签的那几条 rubric 单独拎出来，用同一套权重公式重算一遍。所以同一题目里，`theme:communication` 和 `axis:completeness` 可能是完全不同的两个数——前者是整题分，后者只看了 completeness 那几条。

cluster 比较特殊。它是 37 种被大量题目复用的通用细则族（相同家族的文本 100% 一致）。比如 `cluster:hedging_any-reducible-uncertainty_hedges` 会同时出现在好几道 hedging 题上，聚合时跨题收在一起算，判定标准被锁定、样本也更大。官方全库共 37 个家族；样本量越大，覆盖到的家族越多，这个指标的统计意义越强。



### 3\.6 最终输出



标准输出结构完全由 `EvalResult`（`simple-evals/types.py`）决定，四个字段：



```Python
EvalResult(
    score: float | None,        # 头榜分
    metrics: dict[str, float],  # 扁平 dict，见下
    htmls: list[str],           # 每个样例的报告 HTML（人审用）
    convos: list[MessageList],  # 每个样例的完整对话
    metadata: dict | None,      # {"example_level_metadata": [...]}，样例级明细
)
```



`metrics` 是扁平 dict，每个指标名占三个 key（来自 `_aggregate_get_clipped_mean`：对每个 metric 名，遍历 `["mean", "n_samples", "bootstrap_std"]` 三个统计量）：



```JSON
{"overall_score": 0.61,"overall_score:n_samples": 100,"overall_score:bootstrap_std": 0.02,"theme:hedging": 0.54,"theme:hedging:n_samples": 25,"theme:hedging:bootstrap_std": 0.09,"axis:accuracy": 0.48,"axis:accuracy:n_samples": 82,"axis:accuracy:bootstrap_std": 0.05,"level:example": 0.55,"level:example:n_samples": 100,"level:cluster": 0.72,"level:cluster:n_samples": 100}
```



（数值为演示结构用的示意值，非任何真实评测结果。`simple_evals.py` 跑完再把这堆 key 排序、把 `score` 合并进顶层，写进 `metrics.json`。）



每个指标名的含义速查：



|指标名|含义|算法|n\_samples 的含义|
|---|---|---|---|
|overall\_score|整体行为质量分|每例 achieved/total\_possible → 跨例 clip\(mean\)|参与统计的样例总数|
|theme:\*|某主题下的表现|该主题 tagged 样例的 overall\_score → mean/bootstrap|该主题的样例数|
|axis:\*|某行为维度的表现|题内把带该 axis 的 rubric 子集重算 score → 跨例 mean|题内出现过该 axis rubric 的样例数|
|cluster:\*|某通用细则族的判定一致性|跨题收集该族 rubric 的判定 → 重算 score → mean|该族 rubric 出现的题数|
|level:example / level:cluster|全部定制细则 / 全部通用细则的整体表现|题内按 level 分组，用 calculate\_score 重算 → 跨例 mean|所有样例（每题都有 level 标签）|
|physician\_agreed\_category:\*|某医生共识类别的表现|同 theme，按类别归档 → mean/bootstrap|该类别的样例数|
|xxx:bootstrap\_std|该指标均值的标准误|1000 次有放回重采样算均值的 std|—|
|xxx:n\_samples|该指标样本量|计数|—|



几个容易误解的点：



1. `score` 头榜分和 `metrics["overall_score"]` 是同一个值。`EvalResult.score` 来自 `final_metrics.pop("score")`——聚合时把每个样例的 `score`（= 该例 `overall_score`）收集起来算 clip 均值，而这个值恰好就是 `metrics["overall_score"]`。为什么会有两处？因为通用聚合器对所有指标一视同仁，`score` 只是它特殊处理、提出来放到顶层的那一个。

2. 指标不止 theme/axis/cluster 三类。`grade_sample` 会把 rubric 上的所有标签都当成分组维度（见 `healthbench_eval.py:452-467`），所以官方输出里还包含 `level:example`、`level:cluster`——对应"全部定制细则的整体分"和"全部通用细则的整体分"。注意 `level:example` 的 n\_samples 通常等于样例总数（每题都有定制细则），这点与 axis/cluster 不同。

3. 长度调整指标 `overall_score_length_adjusted` 只在开启长度惩罚时才出现，默认运行没有。

4. bootstrap\_std 的计算方式和含义。聚合时每个维度算三个数：mean、bootstrap\_std、n\_samples。mean 会被 clip 到 \[0,1\]；bootstrap\_std 是把该维度的采样分数有放回重采样 1000 次、每次取均值，再对这 1000 个均值求标准差。bootstrap\_std 不是"样例分数的离散度"，而是均值的标准误：换一批同分布的题重测，这个均值大概会漂多远。经验关系是 bootstrap\_std ≈ 单样例标准差 / √n，比如：

  - 全库 5000 题规模下约 ±0\.01 量级（官方报告误差带很小的原因）；

  - 几十题的规模普遍到 ±0\.05 以上；

  - 个位数时可能 ±0\.1\~0\.2，均值基本没有参考价值。

所以 `0.43 ± 0.02` 的意思是"真实水平的估计精度大约在这个范围"，不是说模型表现本身会在 0\.41\~0\.45 之间跳。还有一个伴随的局限：bootstrap\_std 大时，很难分清是"题目难度不齐"还是"模型表现不稳"，想看模型具体能力更多靠分主题看。



## 4\. 本地实测



这轮评测的目的是把整套流程完整跑一遍：看评分流程怎么运作、结果指标如何呈现、能不能简单读出模型的强弱项。小样本量决定了它只能当"看流程 \+ 找线索"的 pilot，无法代表模型真实水平。

本次的评测配置如下

- 三个被评测模型：deepseek\-v4\-flash（通用）、Qwen3\-235B（通用）、qwen3\.5\-sft（在 Qwen 基座上微调）。评测按第三节的框架：GRADER\_TEMPLATE \+ 逐条判分 \+ calculate\_score \+ 分维度聚合。

- 裁判模型用 deepseek\-v4\-flash。裁判和第一个被测模型是同一个，跨模型比较时可能存在系统偏好（后面会看到它的总体分确实高一些）。

- 采样时 temperature=0\.5、没有固定随机种子，同一道题重复跑分数会有波动。

数据展示说明：HealthBench 数据集明确要求不要在网上以纯文本或图片形式公开数据集的示例，防止题目泄露进基础模型训练语料、或被有联网能力的模型直接检索到真实答案（数据集为此内置了金丝雀字符串便于过滤）。因此下文涉及具体题目和 rubric 一律做转述处理——题目只给语义概括，细则只给"分值 \+ 考察点"，不引用任何原文。分数、细则数量、标签体系、聚合公式等元信息不受此限制。



### 4\.1 抽样与整体结果



选题为全主题分层抽样：每主题 3 题 = 21 题。这样每个主题分有 3 个样本，theme / cluster / category 几层聚合才有意义。



整体结果（21 题）：



|模型|overall mean|bootstrap\_std|n|
|---|---|---|---|
|deepseek\-v4\-flash|0\.461|0\.05|21|
|qwen3\.5\-sft|0\.403|0\.057|21|
|Qwen3\-235B|0\.366|0\.066|21|



在21道题目之前，我第一轮先跑了 6 题 pilot（每主题 1 题），完全分不出模型差异；扩大到 21 题后，误差带从 ±0\.1 缩到 ±0\.05\~0\.07。三个模型之间的差距（0\.46 vs 0\.40 vs 0\.37）仍然只有 2\~3 个标准差。21 题能看出排序方向，但不足以当严格结论。



### 4\.2 分主题与分维度结果



按主题看（每主题 n=3）：



|theme|deepseek|qwen3\.5|Qwen3\-235B|
|---|---|---|---|
|context\_seeking|0\.346|0\.426|0\.531|
|communication|0\.665|0\.322|0\.463|
|complex\_responses|0\.706|0\.611|0\.667|
|emergency\_referrals|0\.426|0\.436|0\.315|
|global\_health|0\.339|0\.26|0\.192|
|health\_data\_tasks|0\.461|0\.222|0|
|hedging|0\.283|0\.545|0\.498|



按行为维度看（n 是该维度实际出现的题数）：



|axis|deepseek|qwen3\.5|Qwen3\-235B|
|---|---|---|---|
|accuracy \(n=16\)|0\.67|0\.572|0\.65|
|completeness \(n=16\)|0\.432|0\.557|0\.45|
|context\_awareness \(n=15\)|0\.376|0\.37|0\.316|
|communication\_quality \(n=6\)|1|0\.833|1|
|instruction\_following \(n=4\)|0\.625|0\.75|0\.708|



这次出现 5 个负分 case（单题算出来是负数，聚合时被 clip 成 0）：qwen3\.5 在 communication 一个 −0\.068，Qwen3\-235B 和 qwen3\.5 在 health\_data\_tasks 各两个（最低 −0\.381），deepseek 在 hedging 一个 −0\.170。负分不是 bug，是回答真的触发了负分条款的结果，单题看信息量反而更大——能直接看出模型在哪道题上做了危险的事。



### 4\.3 结果的初步结论



模型共性的弱点比较清楚：context\_awareness 三个模型都低（0\.23\~0\.38），主动澄清、结合用户具体情境这类行为是共同短板；高紧迫性的医疗判定场景（UC 伴发热、突发视力模糊）也都上不去。



个体差异的信号（还是那句话，21 题不足以定论）：deepseek 在 communication、complex\_responses、health\_data\_tasks 相对强，但 hedging 明显弱（0\.283 还有个负分 case）；qwen3\.5 在 hedging、context\_seeking 相对强，但 health\_data\_tasks 只有 0\.222；Qwen3\-235B 在 health\_data\_tasks 直接归零。



复现性上还有一个证据：21 题中我重跑了 6 道题，发现同一题的分数有波动，例如胸片题 Qwen3\-235B 从 0\.143 变成 −0\.381。小样本下采样随机性会进一步稀释信号，固定种子、降低温度、或者对关键题做 n\_repeats 都更稳妥。



## 5\. 分数怎么读，以及对评估的几点看法



### 5\.1 分数怎么读



分数能代表方案的相对好坏吗？部分能。前提是只在题目内部比：跨题不可比（每题 total\_possible 不同、难度不同，0\.82 的产后抑郁和 0\.14 的 ACLS 不是同一个量纲）；同分不等于同方案；裁判噪声带进 0\.05 级别的不确定性。可靠的做法是同题内比较、配合逐条判据看丢分点、用分维度分定位强弱。



bootstrap\_std 是均值的不确定性，不是分数的离散度。它回答的是"换一批同分布的题，均值会漂多远"，随 n 收缩（约等于单题标准差除以根号 n）。它区分不了"题难"还是"模型不稳"，要拆开靠分主题。



### 5\.2 医疗评估初步需要关注的几个点



1. 评分细则决定分数的可信度。HealthBench 的细则质量高，才让它成为大家都在参考使用的评测集。它的框架本身是通用的，但换了场景必须重写细则——评估的灵魂是细则，不是框架。

2. 分维度分是刚需。只报一个总分，会丢掉模型画像——两个同分的模型，按主题和轴切开可能是完全不同的两种病。这也和我自己在医养评估里的结论一致：措施级判定先于维度打分，分数是为汇报设计的，逐条判定才是为改进设计的。

3. 样本量要诚实。n=6 撑不起排序，n=21 只能看方向，要下结论至少 50 题往上，并且配 meta\-eval 验证裁判。HealthBench 把 bootstrap\_std 写在明面上，这是其他基准很少给的——它逼着我们面对"这个分数到底有多可信"。

