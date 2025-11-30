# app.py
import os, json, logging
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# 导入提示词
from prompt import translation_prompt, short_prompt, long_prompt

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 抑制第三方库的 DEBUG 日志
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# --- 格式化显示函数 ---
def format_report_html(result):
    """将批改结果转换为 HTML 格式用于展示"""
    if isinstance(result, dict) and "error" in result:
        return f'<div style="color: red; padding: 20px;"><h3>❌ 错误</h3><p>{result["error"]}</p></div>'
    
    # 支持两种输出格式：
    # 1) 嵌套的 {"report": {...}}（详细结构）
    # 2) 扁平结构 {"score":..., "errors": [...], "advice":..., "revised_version":...}
    if not isinstance(result, dict):
        return f'<pre>{json.dumps(result, ensure_ascii=False, indent=2)}</pre>'

    # 如果是扁平 schema（常见于简单翻译批改）
    if "errors" in result and "report" not in result:
        html = '<div style="font-family: Arial, sans-serif; line-height: 1.8;">'
        score = result.get('score', 'N/A')
        html += f'<div style="background: #2196F3; padding: 15px; border-radius: 5px; margin-bottom: 20px;"><h2>📊 总分: <span style="color: #F1F6F3; font-size: 1.5em;">{score}</span></h2></div>'

        # 错误列表
        errors = result.get('errors', [])
        html += '<h3 style="border-bottom: 2px solid #2196F3; padding-bottom: 10px;">❗ 发现的问题</h3>'
        if errors:
            html += '<ul>'
            for it in errors:
                loc = it.get('loc', '')
                etype = it.get('type', '')
                detail = it.get('detail', '')
                html += f'<li><strong>{loc}</strong> — <em>{etype}</em><br/><small>{detail}</small></li>'
            html += '</ul>'
        else:
            html += '<p>未发现明显错误。</p>'

        # 建议
        advice = result.get('advice', '')
        if advice:
            html += '<h3 style="border-bottom: 2px solid #2196F3; padding-bottom: 10px; margin-top: 20px;">💡 建议</h3>'
            html += f'<p>{advice}</p>'

        # 修订版本
        revised = result.get('revised_version', '')
        if revised:
            html += '<h3 style="border-bottom: 2px solid #2196F3; padding-bottom: 10px; margin-top: 20px;">📝 修订版本</h3>'
            html += f'<div style="background: #010110; padding: 15px; border-left: 4px solid #ffc107; border-radius: 3px;"><p>{revised}</p></div>'

        html += '</div>'
        return html
    
    html = '<div style="font-family: Arial, sans-serif; line-height: 1.8;">'
    
    # 总分
    score = result.get('score', 'N/A')
    html += f'<div style="background: #2196F3; padding: 15px; border-radius: 5px; margin-bottom: 20px;"><h2>📊 总分: <span style="color: #F1F6F3; font-size: 1.5em;">{score}</span></h2></div>'
    
    report = result.get('report', {})
    
    # 整体分析
    overall = report.get('overall_analysis', {})
    html += '<h3 style="color: #F1F1FF; border-bottom: 2px solid #2196F3; padding-bottom: 10px;">📋 整体分析</h3>'
    html += f'<p><strong>初印象：</strong>{overall.get("impression", "N/A")}</p>'
    
    # 词汇分析
    vocab = report.get('vocabulary', {})
    html += '<h3 style="color: #F1F1FF; border-bottom: 2px solid #2196F3; padding-bottom: 10px; margin-top: 20px;">📚 词汇分析</h3>'
    
    highlight_words = vocab.get('highlight_words', [])
    if highlight_words:
        html += '<h4>✨ 亮眼词汇</h4><ul>'
        for item in highlight_words:
            html += f'<li><strong>{item.get("word", "")}</strong>：{item.get("reason", "")}</li>'
        html += '</ul>'
    
    spelling_errors = vocab.get('spelling_errors', [])
    if spelling_errors:
        html += '<h4>✏️ 拼写错误</h4><ul>'
        for item in spelling_errors:
            html += f'<li><strong>{item.get("error", "")}</strong> → <span style="color: green;"><strong>{item.get("correct", "")}</strong></span><br/><small>{item.get("explanation", "")}</small></li>'
        html += '</ul>'
    
    # 句型分析
    sentence = report.get('sentence_structure', {})
    html += '<h3 style="color: #F1F1FF; border-bottom: 2px solid #2196F3; padding-bottom: 10px; margin-top: 20px;">🔤 句型分析</h3>'
    
    highlight_sentences = sentence.get('highlight_sentences', [])
    if highlight_sentences:
        html += '<h4>✨ 优秀句子</h4><ul>'
        for item in highlight_sentences:
            html += f'<li><em>"{item.get("sentence", "")}"</em><br/><small>{item.get("reason", "")}</small></li>'
        html += '</ul>'
    
    grammar_errors = sentence.get('grammar_errors', [])
    if grammar_errors:
        html += '<h4>❌ 语法错误</h4><ul>'
        for item in grammar_errors:
            html += f'<li><strong>错误：</strong> {item.get("error_sentence", "")}<br/><strong style="color: green;">修正：</strong> {item.get("corrected", "")}<br/><small>{item.get("explanation", "")}</small></li>'
        html += '</ul>'
    
    # 篇章结构
    chapter = report.get('chapter_structure', {})
    html += '<h3 style="color: #F1F1FF; border-bottom: 2px solid #2196F3; padding-bottom: 10px; margin-top: 20px;">📄 篇章结构</h3>'
    html += f'<p><strong>框架：</strong>{chapter.get("framework", "N/A")}</p>'
    html += f'<p><strong>完整性：</strong>{chapter.get("completeness", "N/A")}</p>'
    html += f'<p><strong>连贯性：</strong>{chapter.get("coherence", "N/A")}</p>'
    
    # 作文润色
    polish = result.get('polish', {})
    if polish:
        html += '<h3 style="color: #F1F1FF; border-bottom: 2px solid #2196F3; padding-bottom: 10px; margin-top: 20px;">✨ 作文润色建议</h3>'
        
        vocab_polish = polish.get('vocabulary_level', {})
        if vocab_polish.get('advanced_replacements'):
            html += '<h4>🔄 高级词汇替换</h4><ul>'
            for item in vocab_polish['advanced_replacements']:
                html += f'<li><strong>{item.get("original", "")}</strong> → <span style="color: green;"><strong>{item.get("advanced", "")}</strong></span><br/><small>{item.get("reason", "")} | 例: {item.get("example", "")}</small></li>'
            html += '</ul>'
        
        connector = vocab_polish.get('connector_optimization', {})
        if connector:
            html += '<h4>🔗 连接词优化</h4>'
            html += f'<p><small>{connector.get("current_overuse", "")}</small></p>'
            recommendations = connector.get('recommendations', {})
            if recommendations:
                html += '<ul>'
                for rel_type, connectors in recommendations.items():
                    if connectors:
                        html += f'<li><strong>{rel_type}类：</strong> {", ".join(connectors)}</li>'
                html += '</ul>'
        
        # 修订版
        html += '<h3 style="color: #F1F1FF; border-bottom: 2px solid #2196F3; padding-bottom: 10px; margin-top: 20px;">📝 修订版本</h3>'
        revised = result.get('revised_version', '')
        html += f'<div style="background: #010110; padding: 15px; border-left: 4px solid #ffc107; border-radius: 3px;"><p>{revised}</p></div>'
    
    html += '</div>'
    return html

# --- 加载配置 ---
load_dotenv()
logger.info("✅ 已加载 .env 文件")

with open("config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)["llm"]
logger.info(f"✅ 配置加载完成: model={cfg['model']}, api_base={cfg['api_base']}")

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    logger.error("❌ 错误: 请在 .env 设置 DEEPSEEK_API_KEY")
    raise SystemExit("请在 .env 设置 DEEPSEEK_API_KEY")
logger.info("✅ API_KEY 已加载")

# 设置环境变量供 ChatOpenAI 使用
os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = cfg["api_base"]
logger.info("✅ 环境变量已设置")

# --- 初始化 DeepSeek LLM ---
llm_config = {
    "model": cfg["model"],
    "temperature": cfg["temperature"],
    "max_tokens": cfg["max_tokens"]
}
llm = ChatOpenAI(**llm_config)
logger.info(f"✅ LLM 初始化成功: {llm_config}")

# --- 测试 API 连接 ---
def test_api_connection():
    """测试是否能成功请求 DeepSeek API"""
    logger.info("🔍 开始测试 API 连接...")
    try:
        test_message = HumanMessage(content="你好，请回复一句中国古诗")
        response = llm.invoke([test_message])
        logger.info(f"✅ API 连接成功!")
        return {
            "status": "success",
            "message": "API 连接成功",
            "response": response.content[:100] if response.content else "无响应"
        }
    except Exception as e:
        logger.error(f"❌ API 连接失败: {str(e)}")
        return {
            "status": "error",
            "message": f"API 连接失败: {str(e)}"
        }

# --- 统一解析 ---
@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(4))
def parse_output(resp: str):
    logger.info(f"📥 收到 LLM 响应: {resp[:200]}...")
    resp = resp.strip()
    # 直接尝试 JSON
    try:
        result = json.loads(resp)
        logger.info(f"✅ 成功解析 JSON")
        return result
    except Exception as e1:
        logger.info(f"⚠️ 直接解析失败，尝试提取 JSON 片段: {str(e1)}")
        # 裁出第一个 { 和 最后一个 }
        s = resp.find("{")
        e = resp.rfind("}")
        if s >= 0 and e > s:
            try:
                extracted = resp[s:e+1]
                logger.info(f"📍 提取的 JSON 片段: {extracted[:200]}...")
                result = json.loads(extracted)
                logger.info(f"✅ 成功解析提取的 JSON 片段")
                return result
            except Exception as extract_err:
                logger.error(f"❌ 提取的 JSON 片段解析失败: {str(extract_err)}")
                logger.error(f"❌ 原始响应: {resp[:300]}")
                raise ValueError(f"无法解析 JSON：{resp[:200]}")
        logger.error(f"❌ 未找到 JSON 片段，完整响应: {resp}")
        raise ValueError(f"无法解析 JSON：{resp[:200]}")

def eval_translation(src, stu):
    logger.info("📝 [英译汉] 开始评估")
    try:
        prompt_text = translation_prompt.format(src_text=src, student_text=stu)
        response = llm.invoke([HumanMessage(content=prompt_text)])
        raw = response.content if isinstance(response.content, str) else str(response.content)
        logger.info(f"原始响应片段: {raw[:200]}")
        try:
            result = parse_output(raw)
        except Exception as pe:
            logger.error(f"❌ 解析失败: {type(pe).__name__}: {repr(pe)}")
            raise
        logger.info(f"✅ [英译汉] 评估完成, 分数: {result.get('score', 'N/A')}")
        return format_report_html(result)
    except Exception as e:
        logger.error(f"❌ [英译汉] 评估失败: {str(e)}", exc_info=True)
        error_html = f'<div style="color: red; padding: 20px; background: #ffebee; border-radius: 5px;"><h3>❌ 评估失败</h3><p>{str(e)}</p></div>'
        return error_html

def eval_short(topic, stu):
    logger.info("📝 [小作文] 开始评估")
    try:
        prompt_text = short_prompt.format(topic=topic, student_text=stu)
        response = llm.invoke([HumanMessage(content=prompt_text)])
        raw = response.content if isinstance(response.content, str) else str(response.content)
        logger.info(f"原始响应片段: {raw[:200]}")
        try:
            result = parse_output(raw)
        except Exception as pe:
            logger.error(f"❌ 解析失败: {type(pe).__name__}: {repr(pe)}")
            raise
        logger.info(f"✅ [小作文] 评估完成, 分数: {result.get('score', 'N/A')}")
        return format_report_html(result)
    except Exception as e:
        logger.error(f"❌ [小作文] 评估失败: {str(e)}", exc_info=True)
        error_html = f'<div style="color: red; padding: 20px; background: #ffebee; border-radius: 5px;"><h3>❌ 评估失败</h3><p>{str(e)}</p></div>'
        return error_html

def eval_long(topic, stu):
    logger.info("📝 [大作文] 开始评估")
    try:
        prompt_text = long_prompt.format(topic=topic, student_text=stu)
        response = llm.invoke([HumanMessage(content=prompt_text)])
        raw = response.content if isinstance(response.content, str) else str(response.content)
        logger.info(f"原始响应片段: {raw[:200]}")
        try:
            result = parse_output(raw)
        except Exception as pe:
            logger.error(f"❌ 解析失败: {type(pe).__name__}: {repr(pe)}")
            raise
        logger.info(f"✅ [大作文] 评估完成, 分数: {result.get('score', 'N/A')}")
        return format_report_html(result)
    except Exception as e:
        logger.error(f"❌ [大作文] 评估失败: {str(e)}", exc_info=True)
        error_html = f'<div style="color: red; padding: 20px; background: #ffebee; border-radius: 5px;"><h3>❌ 评估失败</h3><p>{str(e)}</p></div>'
        return error_html

# --- WebUI ---
with gr.Blocks(title="考研英语 AI 批改系统") as ui:
    gr.Markdown("# 📝 考研英语 AI 批改系统（DeepSeek + LangChain）")
    gr.Markdown("---")
    gr.HTML(r'''
<style>
:root { font-size: 20px; }
body, .gradio-container { font-size: 20px !important; }
textarea, input, .gr-textbox textarea, .gradio-textbox textarea, .gradio-input textarea, .gradio-textbox input { font-size: 18px !important; }
input, textarea { caret-color: #222; }
.gr-button, button { font-size: 18px !important; padding: 12px 18px !important; }
/* 结果区字体放大以确保 HTML 内容（含 <pre>）随全局字号变化 */
.gr-html, .gradio-html { font-size: 20px !important; }
/* 单独设置结果区各级标题的字号（仅改字号，不改颜色或其它样式） */
.gr-html h1, .gradio-html h1 { font-size: 28px !important; }
.gr-html h2, .gradio-html h2 { font-size: 26px !important; }
.gr-html h3, .gradio-html h3 { font-size: 22px !important; }
.gr-html h4, .gradio-html h4 { font-size: 20px !important; }
.gr-html h5, .gradio-html h5 { font-size: 18px !important; }
.gr-html p, .gradio-html p, .gr-html li, .gradio-html li, .gr-html pre, .gradio-html pre { font-size: 30px !important; }
.gr-html * {
    font-size: 55px !important;
}
</style>
''' )

    with gr.Tab("英译汉"):
        gr.Markdown("### 📖 输入")
        t1 = gr.Textbox(label="原文（英文）", lines=8, placeholder="请输入要翻译的英文原文")
        t2 = gr.Textbox(label="考生译文（中文）", lines=8, placeholder="请输入学生的中文翻译")
        btn = gr.Button("批改", variant="primary", size="lg")
        gr.Markdown("### 📋 批改结果")
        out = gr.HTML(label="批改报告")
        btn.click(eval_translation, [t1, t2], out)

    with gr.Tab("小作文"):
        gr.Markdown("### 📖 输入")
        s1 = gr.Textbox(label="题目", lines=4, placeholder="请输入小作文题目")
        s2 = gr.Textbox(label="考生作文", lines=12, placeholder="请输入学生的作文（约100词）")
        btn2 = gr.Button("批改", variant="primary", size="lg")
        gr.Markdown("### 📋 批改结果")
        out2 = gr.HTML(label="批改报告")
        btn2.click(eval_short, [s1, s2], out2)

    with gr.Tab("大作文"):
        gr.Markdown("### 📖 输入")
        l1 = gr.Textbox(label="题目", lines=4, placeholder="请输入大作文题目")
        l2 = gr.Textbox(label="考生作文", lines=18, placeholder="请输入学生的作文（约250词）")
        btn3 = gr.Button("批改", variant="primary", size="lg")
        gr.Markdown("### 📋 批改结果")
        out3 = gr.HTML(label="批改报告")
        btn3.click(eval_long, [l1, l2], out3)

    with gr.Tab("API测试"):
        gr.Markdown("### 🔍 API 连接测试")
        test_btn = gr.Button("测试 API 连接", variant="primary")
        test_output = gr.JSON(label="测试结果")
        test_btn.click(test_api_connection, outputs=test_output)
ui.launch()
