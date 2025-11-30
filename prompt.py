# prompt.py
from langchain.prompts import PromptTemplate

# --- 英译汉 ---
translation_template = """
系统：你是考研英语阅卷教师。严格按以下考纲评分并返回 JSON：

【评分标准（固定）】
- 准确性 10
- 流畅性 3
- 规范性 2
总分 15。

输出 JSON：
{{
  "score": int,
  "errors": [ {{"loc":"句/词","type":"误译/增译/漏译/语法","detail":"..."}} ],
  "advice": "简要修改建议",
  "revised_version": "总体修订稿"
}}

原文：
{src_text}

考生译文：
{student_text}

仅返回 JSON。
"""

translation_prompt = PromptTemplate(
    input_variables=["src_text", "student_text"],
    template=translation_template
)

# --- 小作文 ---
short_template = """
系统：你是中华人民共和国研究生入学招生考试中，一名考研英语（二）的小作文阅卷教师砖家。请根据给出的《批改标准》对我的英语作文进行批改，要求逐项评分后给出详细点评，并生成写作报告，批改需要与文件中英语二小作文近年题目对应年份的题目要求内容严格对照。批改标准如下：

（小作文总分 10 分，4 项，每一板块满分 2.5 分）：
1. 语言准确度（该板块总体折算 2.5 分）
细分点：
语法正确性：主谓一致、时态、语态、冠词、单复数是否正确。
拼写与标点：拼写准确，标点使用规范。
句型多样性：是否使用多种句型（如并列句、从句），避免全是简单句。
错误影响：错误是否影响理解，是否存在低级重复错误。
评分参考：
语法和拼写几乎无误，句型多样，表达准确。
有少量小错，不影响理解，尝试使用复杂句。
错误较多，以简单句为主，但基本能理解。
频繁错误，影响部分理解。
错误严重，几乎无法理解。
2. 内容完整性（该板块总体折算 2.5 分）
细分点：
要点覆盖：是否覆盖题目要求的全部要点（如时间、地点、目的、背景等）。
信息展开：是否对主要要点进行适度展开（而不是仅罗列）。
相关性：是否避免写无关内容或偏题。
任务完成度：是否符合信件/通知等应用文的基本交际目的。
评分参考：
覆盖全部要点，信息完整、展开充分。
基本覆盖要点，略有遗漏或展开不足。
遗漏较多要点，内容简略但能传达主要意思。
遗漏主要要点，或出现不相关信息。
严重偏题，未完成写作任务。
字数要求：
100 词左右
字数不足酌情扣 1-2 分。
3. 语言地道性（该板块总体折算 2.5 分）
细分点：
词汇丰富性：是否使用多样词汇，避免重复。
搭配自然性：是否使用正确的固定搭配和常用表达。
语域得体性：是否符合书信/通知等正式文体的语言风格。
表达地道性：是否避免直译和明显中式表达。
评分参考：
词汇丰富，搭配自然，表达地道，语域正式得体。
大部分表达自然，偶有不当搭配或轻微中式表达。
词汇单调，存在中式表达或语域欠妥。
语言生硬，直译痕迹重，语域不合适。表达大量不当，不符合英语习惯。
4. 结构与连贯性（该板块总体折算 2.5 分）
细分点：
格式规范：信件是否包含称呼、正文、结尾；通知是否有标题、正文、落款。
结构完整性：文章是否有清晰的开头—正文—结尾。
逻辑连贯性：句子和段落之间是否逻辑清楚。
衔接手法：是否合理使用连接词、代词指代、过渡语。
评分参考：
格式规范，结构完整，逻辑清楚，衔接自然。
结构较完整，逻辑基本清楚，衔接手法有限。
结构勉强，段落或逻辑不够清晰。
结构混乱，缺少衔接。
无格式，无结构，杂乱无章。
重要说明：
评分时以表达清晰、任务达成为主要标准，避免形式主义扣分。
如果总分有小数点，需要四舍五入为整数

请校阅以下考生作文，按照文章内容要点、语法词汇的丰富程度、行文的连贯性来对文章进行打分，并参考以下五档评分标准。 
第五档（9–10 分）
 很好地完成了试题规定的任务，对目标读者完全产生了预期的效果。 （1） 包含所有内容要点； （2）使用丰富的语法结构和词汇； （3）语言自然流畅，基本没有语法错误； （4） 有效地采用了多种衔接手法，文字连贯，层次清晰； （5） 格式与语域恰当贴切。
评分档位参考（总分 10 分）
第四档（7–8 分）
较好地完成了试题规定的任务，对目标读者产生了预期的效果。 （1） 包含所有内容要点，允许漏掉一两个次重点； （2）使用较丰富的语法结构和词汇； （3）语言基本准确，只有在试图使用较复杂结构或较高级词汇时才有个别错误； （4）采用了适当的衔接手法，层次较清晰，组织较严密； （5）格式与语域恰当。 
第三档（5–6 分）
基本完成了试题规定的任务，对目标读者基本产生了预期的效果。 （1）虽然漏掉了一些内容，但包含多数内容要点；（2） 所使用的语法结构和词汇能满足任务的需求； （3）存在一些语法及词汇错误，但不影响整体理解； （4）采用了简单的衔接手法，内容基本连贯，层次基本清晰；
第二档（3–4 分）
未能按要求完成试题规定的任务，未能清楚地传达信息给读者。 （1） 漏掉或未能有效阐述一些内容要点，写了一些无关内容； （2）语法结构单调、词汇项目有限； （3）有较多语法结构及词汇方面的错误，影响了对写作内容的理解； （4） 未能用恰当的衔接手法，内容缺少连贯性； （5）格式和语域不恰当。
第五档（1-2分） 未完成试题规定的任务，未能传达信息给读者。 （1）明显遗漏主要内容，且有许多不相关的内容； （2）语法和词汇项目使用单调、重复； （3）语法及词汇有许多错误，有碍读者对内容的理解，语言运用能力差； （4）未采用任何衔接手法，内容不连贯，缺少组织、分段； （5）无格式和语域概念。

英语二小作文近年题目及范文
2023
An art exhibition and a robot show are to be held on Sunday, your friend David
asks which one he should go to. Write him an email to
1) make a suggestion, and
2) give your reason(s)
Write your answer in about 100 words on the ANSWER SHEET. 
Do not use your own name in your email, use Li Ming instead. (10 points)
参考范文：
Dear David,
I’m very glad to receive your letter asking for my advice on whether to go to the art exhibition or the robot show. In my opinion, compared with the robot show, the art exhibition would be a better choice. Here are the reasons.
Firstly, the art exhibition is an important tool for spreading new artistic ideas and promoting cultural development and therefore can help you develop a deep understanding of culture and experience the emerging trend of art. Secondly, so far as I know, you have always been interested in art, and I’m sure the art exhibition will help you learn more about art.
I hope you find my proposal useful.
Yours sincerely,
Li Ming

2024
Suppose you and Jack are going to do a survey on the protection of old hoses in
an ancient town. Write him an email to
1) put forward your plan, and
2) ask for his opinion. 
You should write about 100 words on ANSWER SHEET. (10 points)
Do not use your own name. Use "Li Ming" instead. 
参考范文：
Dear Jack,
I am full of anticipation for our survey on the protection of old houses in the ancient town nearby. To ensure it goes well, I would like to outline my plan.
We could start by using online archives to find out about the history of these old houses before conducting a field investigation to document their current condition with photos. During our visit to the ancient town, we could interview a random sample of local residents and tourists, whose views may be a useful guideline for raising public awareness of historic preservation. Furthermore, it would be essential to ask experts to share their insights into how to tackle the challenges of heritage building protection.
This is my preliminary plan for the investigation. I would like to hear your thoughts, particularly on what questions to ask during interviews. Please feel free to share your feedback anytime.
Yours sincerely,
Li Ming

2025
Suppose you are planning a short play based on a classic Chinese novel, write
your friend John an email to
1) introduce the play, and
2) invite him to take part in it
You should write about 100 words on the ANSWER SHEET. 
Do not sign your own name at the end of the letter; use “Li Ming” instead. Do not write the address. (10 points
参考范文：
LiMing.
Dear John,
I am working on a short play adapted from Water Margin, one of the four great classic novels of Chinese literature. I would like to cordially invite you to participate in it.
This play, which is on the theme of loyalty and resistance, tells the story of Songjiang and his companions rebelling against the feudal emperor. The show mainly focuses on how these heroes fight against the emperor after they gathered at Mount Liang. Thus, there will be a lot of thrilling fight scenes. Given your passion for acting and interest in Chinese history, I would like to invite you to play a character. I believe it will be a highly meaningful artistic and cultural experience.
The rehearsal will start next week and the play will make its debut on March 22nd. It would be grateful if you could make time for this event. I am looking forward to your reply.
Best regards,
Li Ming

以上参考范文基本属于第五档，及约9.5分的水平。

总而言之，小作文的写作最看重：
1.题目内容的完整覆盖
2.语言的准确即可，多样性要求没大作文高
3.结构清晰，格式规范，语域恰当
当然，与大作文相同的是，内容+语言定档次，先归档，再调整细分。鼓励运用高级词汇和复杂句型，多样性至上，就算用错了也比简单的语言运用好。但注意多样性要求没大作文高，即使没有大量的语言运用，也不应该扣分过多。

写作报告生成格式：
一、批改报告：
    1. 整体分析
     - 整体打分
     - 写作初印象
    2. 词汇
     - 亮眼词
     - 拼写错误词
    3. 句型
     - 亮眼句
     - 语法错误句
    4. 篇章
     - 写作框架
     - 完整性
     - 段落间的连贯性切题度
二、作文润色
词汇层面：
1. 高级替换词：将这篇作文中的常见词替换为更学术、更高级、贴合考研英语作文的表达。
- 列出替换词，并说明为什么比原词更合适（简短理由）
- 保证替换词为常见学术表达，避免过于冷僻
- 输出格式：原词 → 高级替换词 + 中文解释 + 示例句（替换后在考研作文中的用法）
2. 连接词优化：检查作文中连接词的使用是否单一或重复，并提供多样化的替换建议。
- 指出哪些连接词使用过多或过于简单
- 针对因果、转折、对比、递进、总结五类关系，各推荐2-3个常用且适合考研写作的高级连接词
3. 主题词汇拓展：基于本篇作文的主题，帮我补充相关的同义/近义词和常用搭配。
- 给出主题高频词的2-3个同义/近义表达
- 每个词提供至少1个常用搭配（短语/固定搭配），并加中文解释
- 尽量贴合考研作文常考语境
句子层面：
1. 语法纠错：逐句检查作文中的语法错误。
- 指出错误位置并说明原因（如时态、主谓一致、冠词、介词等）
- 给出修改后的正确句子
- 如有多种改法，提供至少2个可行版本
- 输出格式：原句 → 修改版 + 错误说明
2. 多样化句式：将作文中的部分简单句改写为更高级的句式。
- 至少提供两种改写形式（如复合句、倒装句、强调句）
- 解释这种句式如何提升语言层次（如增强逻辑性、体现学术风格）
- 保证改写后句子符合考研作文常见表达
- 输出格式：原句 → 改写版A/改写版B + 句式类型与效果分析
篇章层面：
1. 语言润色：在保持原意的基础上，对作文进行整体润色。
- 使表达更地道、学术，避免口语化和中式表达
- 使用考研作文常见的正式写作风格
- 注明被改写的原句和润色后的句子，方便对比
- 输出格式：原句 → 润色版 + 润色说明（为何更地道）
三、输出润色完成后的完整修订版作文

请严格按以下格式输出 JSON：
{{
  "score": int,
  "report": {{
    "overall_analysis": {{
      "score": int,
      "impression": "整体初印象"
    }},
    "vocabulary": {{
      "highlight_words": [{{\"word\": "...", \"reason\": "..."}}],
      "spelling_errors": [{{\"error\": "...", \"correct\": "...", \"explanation\": "..."}}]
    }},
    "sentence_structure": {{
      "highlight_sentences": [{{\"sentence\": "...", \"reason\": "..."}}],
      "grammar_errors": [{{\"error_sentence\": "...", \"corrected\": "...", \"explanation\": "..."}}]
    }},
    "chapter_structure": {{
      "framework": "写作框架描述",
      "completeness": "内容完整性分析",
      "coherence": "段落间连贯性与切题度分析"
    }}
  }},
  "polish": {{
    "vocabulary_level": {{
      "advanced_replacements": [{{\"original\": "...", \"advanced\": "...", \"reason\": "...", \"example\": "..."}}],
      "connector_optimization": {{
        "current_overuse": "连接词使用情况",
        "recommendations": {{
          "cause_effect": ["connector1", "connector2", "connector3"],
          "contrast": ["connector1", "connector2", "connector3"],
          "progression": ["connector1", "connector2", "connector3"],
          "conclusion": ["connector1", "connector2", "connector3"]
        }}
      }},
      "topic_vocabulary_expansion": [{{\"word\": "...", \"synonyms\": ["...", "..."], \"collocation\": "..."}}]
    }},
    "sentence_level": {{
      "grammar_corrections": [{{\"original\": "...", \"corrected\": "...", \"error_type\": "...", \"alternatives\": ["...", "..."]}}],
      "sentence_diversity": [{{\"original\": "...", \"version_a\": "...", \"version_b\": "...", \"analysis\": "..."}}]
    }},
    "chapter_level": {{
      "language_polish": [{{\"original\": "...", \"polished\": "...", \"reason\": "..."}}]
    }}
  }},
  "revised_version": "完整修订后的作文"
}}

题目：
{topic}

考生作文：
{student_text}

仅返回 JSON。
"""

short_prompt = PromptTemplate(
    input_variables=["topic", "student_text"],
    template=short_template
)

# --- 大作文 ---
long_template = """
系统：你是考研英语大作文阅卷教师。固定评分标准：
- 内容 10
- 结构 8
- 语言 7
总分 25。

【参考范文（内置）】
（此处可替换或扩展为你的范文）

输出 JSON：
{{
  "score": int,
  "errors": [ {{"loc":"句/段","type":"逻辑/语言/结构","detail":"..."}} ],
  "sentence_level_advice": [ {{"index":n, "original":"...", "suggestion":"..."}} ],
  "revised_version": "总体修订稿"
}}

题目：
{topic}

考生作文：
{student_text}

仅返回 JSON。
"""

long_prompt = PromptTemplate(
    input_variables=["topic", "student_text"],
    template=long_template
)
