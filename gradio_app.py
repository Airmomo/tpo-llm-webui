import gradio as gr
import textgrad as tg
import yaml
import logging
import requests
from typing import Optional, Dict, Any
from reward_model import TPORewardModel
from tpo_utils import run_test_time_training_tpo, run_test_time_training_bon


# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TPOWebUI')


class TPOWebUI:
    def __init__(self):
        self.llm_engine = None
        self.rm = None
        self.vllm_status = "未连接"
        # 加载配置文件
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            logger.info("配置文件加载成功")
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            raise

    def check_vllm_status(self, ip: str, port: int) -> str:
        """检查vLLM服务状态"""
        try:
            headers = {
                "Authorization": f"Bearer {self.config['vllm']['api_key']}"}
            response = requests.get(
                f"http://{ip}:{port}/v1/models", headers=headers)
            if response.status_code == 200:
                self.vllm_status = "运行中"
                return "vLLM服务正常运行"
            elif response.status_code == 401:
                self.vllm_status = "未授权"
                return "vLLM服务认证失败：API密钥无效"
            else:
                self.vllm_status = "异常"
                return f"vLLM服务异常: HTTP {response.status_code}"
        except requests.RequestException as e:
            self.vllm_status = "未连接"
            return f"无法连接到vLLM服务: {str(e)}"

    def initialize_models(self, server_model: str, reward_model: str, ip: str, port: int, max_tokens: int, api_key: str) -> str:
        """初始化模型"""
        try:
            # 更新API密钥配置
            self.config['vllm']['api_key'] = api_key

            # 检查vLLM服务状态
            status_msg = self.check_vllm_status(ip, port)
            if self.vllm_status != "运行中":
                return f"初始化失败: {status_msg}"

            # 初始化LLM引擎
            model_name = f"server-{self.config['tpo']['server_model']}"
            self.llm_engine = tg.get_engine(
                model_name,
                base_url=f"http://{self.config['tpo']['ip']}:{self.config['tpo']['port']}/v1",
                api_key="token-abc123",
                max_token=self.config['tpo']['max_tokens_all'],
            )
            logger.info(f"LLM引擎初始化成功: {server_model}")

            # 初始化奖励模型
            self.rm = TPORewardModel(reward_model)
            logger.info(f"奖励模型初始化成功: {reward_model}")

            return "模型初始化成功！"
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            return f"初始化失败: {str(e)}"

    def optimize_query(self, query: str, mode: str, max_iterations: int, sample_size: int, temperature: float) -> tuple[str, str, str]:
        """执行实时优化"""
        if not query.strip():
            return "请输入需要优化的文本！", "", ""

        if not self.llm_engine or not self.rm:
            return "请先初始化模型！", "", ""

        if self.vllm_status != "运行中":
            return "vLLM服务未连接或异常，请检查服务状态！", "", ""

        try:
            # 从配置文件获取参数
            gen_params = {
                "n": sample_size,
                "temperature": temperature,
                "top_p": 0.95,
                "max_tokens": self.config["tpo"]["max_tokens_response"]
            }

            logger.info(f"开始优化，模式：{mode}，最大迭代次数：{max_iterations}")

            # 选择优化方法
            optimize_func = run_test_time_training_bon if mode == "bon" else run_test_time_training_tpo

            # 执行优化
            result = optimize_func(
                query,
                self.llm_engine,
                self.rm,
                gen_params=gen_params,
                tpo_mode=mode,
                max_iters=max_iterations
            )

            logger.info("优化完成")

            # 解析结果
            outputs = []
            scores = []
            thoughts = []

            # 遍历所有结果
            for key, score in result.items():
                # 从键中提取信息
                _, query_text, answer_text = key.split("<SEP>")

                # 提取思考过程
                think_start = answer_text.find("<think>")
                think_end = answer_text.find("</think>")
                if think_start != -1 and think_end != -1:
                    thought = answer_text[think_start + 7:think_end].strip()
                    result_text = answer_text[think_end + 8:].strip()
                else:
                    thought = ""
                    result_text = answer_text

                thoughts.append(thought)
                outputs.append(result_text)
                scores.append(score)

            # 找到最佳结果
            if scores:
                best_idx = scores.index(max(scores))
                best_result = outputs[best_idx]
            else:
                best_result = "未能生成有效结果"

            # 格式化输出
            thoughts_text = "\n\n---\n\n".join(
                [f"推理 {i+1}:\n{t}" for i, t in enumerate(thoughts[:sample_size]) if t])
            outputs_text = "\n\n---\n\n".join(
                [f"结果 {i+1}:\n{o}" for i, o in enumerate(outputs[:sample_size])])
            scores_text = "\n\n---\n\n".join(
                [f"结果 {i+1} 评分: {s}" for i, s in enumerate(scores[:sample_size])])

            return outputs_text, scores_text, best_result

        except Exception as e:
            error_msg = f"优化过程出错：{str(e)}"
            logger.error(error_msg)
            return "", "", ""


def create_ui():
    tpo = TPOWebUI()

    with gr.Blocks(title="TPO 实时优化系统", theme=gr.themes.Soft()) as app:
        gr.Markdown("# TPO 实时优化系统")
        gr.Markdown("## 基于大语言模型的文本优化系统 —— 使输出结果更加偏向人类偏好")

        with gr.Tab("模型设置"):
            with gr.Row():
                with gr.Column():
                    server_model = gr.Textbox(
                        value=tpo.config["tpo"]["server_model"],
                        label="服务模型",
                        info="输入vLLM服务使用的模型名称"
                    )
                    reward_model = gr.Textbox(
                        value=tpo.config["tpo"]["reward_model"],
                        label="奖励模型",
                        info="输入用于评估文本质量的奖励模型路径"
                    )
                    api_key = gr.Textbox(
                        value=tpo.config["vllm"]["api_key"],
                        label="API密钥",
                        info="输入vLLM服务的API密钥",
                        type="password"
                    )
            with gr.Row():
                with gr.Column():
                    ip = gr.Textbox(
                        value=tpo.config["tpo"]["ip"],
                        label="服务器IP",
                        info="vLLM服务器的IP地址"
                    )
                    port = gr.Number(
                        value=tpo.config["tpo"]["port"],
                        label="端口号",
                        info="vLLM服务器的端口号"
                    )
                    max_tokens = gr.Number(
                        value=tpo.config["tpo"]["max_tokens_all"],
                        label="最大Token数",
                        info="模型支持的最大token数量"
                    )
            with gr.Row():
                init_btn = gr.Button("初始化模型", variant="primary")
                status_box = gr.Textbox(
                    label="系统状态",
                    value="未初始化",
                    interactive=False
                )

            init_btn.click(
                fn=tpo.initialize_models,
                inputs=[server_model, reward_model,
                        ip, port, max_tokens, api_key],
                outputs=status_box
            )

        with gr.Tab("优化设置"):
            with gr.Row():
                query = gr.Textbox(lines=5, label="输入文本")

            with gr.Row():
                mode = gr.Radio(
                    choices=["tpo", "bon", "revision"],
                    value="tpo",
                    label="优化模式"
                )
                max_iterations = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="最大迭代次数"
                )
                sample_size = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="采样数量"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="温度"
                )

            optimize_btn = gr.Button("开始优化")

            with gr.Row():
                best_result_box = gr.Textbox(
                    label="最佳结果",
                    value="",
                    lines=5,
                    interactive=False
                )

            with gr.Row():
                with gr.Column(scale=1):
                    results_box = gr.Textbox(
                        label="生成结果",
                        value="",
                        lines=10,
                        interactive=False
                    )
                with gr.Column(scale=1):
                    scores_box = gr.Textbox(
                        label="评分结果",
                        value="",
                        lines=10,
                        interactive=False
                    )

            optimize_btn.click(
                fn=tpo.optimize_query,
                inputs=[query, mode, max_iterations, sample_size, temperature],
                outputs=[results_box, scores_box, best_result_box]
            )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(share=True, server_name="0.0.0.0", server_port=7860)
