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
        return result
    except Exception as e:
        logger.error(f"❌ [英译汉] 评估失败: {str(e)}", exc_info=True)
        return {"error": str(e)}

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
        return result
    except Exception as e:
        logger.error(f"❌ [小作文] 评估失败: {str(e)}", exc_info=True)
        return {"error": str(e)}

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
        return result
    except Exception as e:
        logger.error(f"❌ [大作文] 评估失败: {str(e)}", exc_info=True)
        return {"error": str(e)}

# --- WebUI ---
with gr.Blocks() as ui:
    gr.Markdown("## 📝 考研英语 AI 批改系统（DeepSeek + LangChain）")

    with gr.Tab("系统检测"):
        gr.Markdown("### 🔍 API 连接测试")
        test_btn = gr.Button("测试 API 连接")
        test_output = gr.JSON(label="测试结果")
        test_btn.click(test_api_connection, outputs=test_output)

    with gr.Tab("英译汉"):
        t1 = gr.Textbox(label="原文（英文）", lines=6)
        t2 = gr.Textbox(label="考生译文（中文）", lines=6)
        btn = gr.Button("批改")
        out = gr.JSON()
        btn.click(eval_translation, [t1, t2], out)

    with gr.Tab("小作文"):
        s1 = gr.Textbox(label="题目", lines=2)
        s2 = gr.Textbox(label="考生作文", lines=10)
        btn2 = gr.Button("批改")
        out2 = gr.JSON()
        btn2.click(eval_short, [s1, s2], out2)

    with gr.Tab("大作文"):
        l1 = gr.Textbox(label="题目", lines=2)
        l2 = gr.Textbox(label="考生作文", lines=15)
        btn3 = gr.Button("批改")
        out3 = gr.JSON()
        btn3.click(eval_long, [l1, l2], out3)

ui.launch()
