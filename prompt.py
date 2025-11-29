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
系统：你是考研英语小作文阅卷教师。固定评分标准如下：
- 内容完整：8
- 连贯组织：6
- 语言表达：6
总分 20。

输出 JSON：
{{
  "score": int,
  "errors": [ {{"loc":"句/段","type":"内容/语言/结构","detail":"..."}} ],
  "advice": "修改建议",
  "revised_version": "总体修订稿"
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
