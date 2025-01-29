import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import threading
from vllm import LLM, SamplingParams
import torch
import os
import torch._dynamo
import subprocess

torch._dynamo.config.suppress_errors = True

class VLLMChatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VLLM Chatbot")
        self.root.geometry("1200x800")
        
        # 模型实例
        self.llm = None
        # 是否是 Instruct 模型
        self.is_instruct = tk.BooleanVar(value=False)
        
        # 新增模型历史记录
        self.model_history = []
        # 最大保存历史数量
        self.max_history = 5
        
        # 创建主要的框架
        self.create_frames()
        # 创建配置区域
        self.create_config_area()
        # 创建聊天区域
        self.create_chat_area()
        # 创建状态栏
        self.create_status_bar()
        
        # 加载配置
        self.load_config()

    def create_frames(self):
        # 左侧配置框架
        self.config_frame = ttk.LabelFrame(self.root, text="配置", padding="5")
        self.config_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 右侧聊天框架
        self.chat_frame = ttk.LabelFrame(self.root, text="聊天", padding="5")
        self.chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_config_area(self):
        # 模型路径配置
        ttk.Label(self.config_frame, text="模型路径:").pack(anchor=tk.W)
        self.model_path = ttk.Combobox(self.config_frame, width=40)
        self.model_path.pack(fill=tk.X, padx=5, pady=2)
        self.model_path['values'] = self.model_history
        
        # 添加浏览按钮
        browse_btn = ttk.Button(self.config_frame, text="浏览...", command=self._browse_model)
        browse_btn.pack(pady=2)
        
        # Instruct 模型选项
        self.instruct_check = ttk.Checkbutton(
            self.config_frame, 
            text="Instruct 模型",
            variable=self.is_instruct
        )
        self.instruct_check.pack(anchor=tk.W, padx=5, pady=2)
        
        # GPU数量配置
        ttk.Label(self.config_frame, text="GPU数量:").pack(anchor=tk.W)
        self.gpu_count = ttk.Spinbox(self.config_frame, from_=1, to=8, width=5)
        self.gpu_count.pack(anchor=tk.W, padx=5, pady=2)
        
        # 温度配置
        temp_frame = ttk.Frame(self.config_frame)
        temp_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(temp_frame, text="温度 (0-2):").pack(side=tk.LEFT)
        self.temp_value = tk.StringVar(value="0.7")
        self.temp_label = ttk.Label(temp_frame, textvariable=self.temp_value, width=4)
        self.temp_label.pack(side=tk.RIGHT)
        
        self.temperature = ttk.Scale(
            self.config_frame,
            from_=0,
            to=2,
            orient=tk.HORIZONTAL,
            command=lambda v: self.temp_value.set(f"{float(v):.1f}")
        )
        self.temperature.set(0.7)
        self.temperature.pack(fill=tk.X, padx=5, pady=(0, 2))
        
        # Top P配置
        ttk.Label(self.config_frame, text="Top P (0-1):").pack(anchor=tk.W)
        self.top_p = ttk.Scale(self.config_frame, from_=0, to=1, orient=tk.HORIZONTAL)
        self.top_p.set(0.95)
        self.top_p.pack(fill=tk.X, padx=5, pady=2)
        
        # 最大序列数配置
        ttk.Label(self.config_frame, text="最大序列数:").pack(anchor=tk.W)
        self.max_seqs = ttk.Entry(self.config_frame, width=10)
        self.max_seqs.insert(0, "256")
        self.max_seqs.pack(anchor=tk.W, padx=5, pady=2)
        
        # 添加最大序列长度配置
        ttk.Label(self.config_frame, text="最大序列长度:").pack(anchor=tk.W)
        self.max_seq_len = ttk.Combobox(self.config_frame, values=["8192", "16384", "32768", "65536"])
        self.max_seq_len.set("32768")
        self.max_seq_len.pack(fill=tk.X, padx=5, pady=2)
        
        # 添加最大生成Token数配置
        ttk.Label(self.config_frame, text="最大生成Token数:").pack(anchor=tk.W)
        self.max_gen_tokens = ttk.Combobox(self.config_frame, values=["1024", "2048", "4096", "8192"])
        self.max_gen_tokens.set("4096")
        self.max_gen_tokens.pack(fill=tk.X, padx=5, pady=2)
        
        # 添加显存使用率配置
        ttk.Label(self.config_frame, text="显存使用率 (0-1):").pack(anchor=tk.W)
        self.gpu_mem = ttk.Scale(self.config_frame, from_=0.5, to=1.0, orient=tk.HORIZONTAL)
        self.gpu_mem.set(0.8)
        self.gpu_mem.pack(fill=tk.X, padx=5, pady=2)
        
        # 启动/停止按钮
        self.start_button = ttk.Button(self.config_frame, text="启动模型", command=self.start_model)
        self.start_button.pack(fill=tk.X, padx=5, pady=10)
        
        # 保存配置按钮
        ttk.Button(self.config_frame, text="保存配置", command=self.save_config).pack(fill=tk.X, padx=5, pady=2)

    def create_chat_area(self):
        chat_frame = ttk.Frame(self.chat_frame)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建聊天历史区域
        self.chat_history = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, height=20)
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建按钮框架
        button_frame = ttk.Frame(chat_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        # 清除历史按钮
        clear_button = ttk.Button(button_frame, text="清除历史", command=self.clear_history)
        clear_button.pack(side=tk.RIGHT, padx=5)

        # 创建消息输入区域
        self.input_frame = ttk.Frame(chat_frame)
        self.input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.message_input = scrolledtext.ScrolledText(self.input_frame, wrap=tk.WORD, height=4)
        self.message_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.send_button = ttk.Button(self.input_frame, text="发送", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=5)
        
        # 绑定回车键发送消息
        self.message_input.bind("<Control-Return>", lambda e: self.send_message())

    def clear_history(self):
        """清除聊天历史"""
        if messagebox.askyesno("确认", "确定要清除所有聊天历史吗？"):
            self.chat_history.delete("1.0", tk.END)

    def create_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_var.set("未启动")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_config(self):
        try:
            with open('vllm_config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.model_history = config.get('model_history', [])
                if self.model_history:
                    self.model_path.set(self.model_history[0])
                self.gpu_count.delete(0, tk.END)
                self.gpu_count.insert(0, str(config.get('gpu_count', 1)))
                self.temperature.set(config.get('temperature', 0.7))
                self.top_p.set(config.get('top_p', 0.95))
                self.max_seqs.delete(0, tk.END)
                self.max_seqs.insert(0, str(config.get('max_seqs', 256)))
                self.gpu_mem.set(config.get('gpu_memory', 0.8))
                self.is_instruct.set(config.get('is_instruct', False))
                self.max_gen_tokens.set(str(config.get('max_gen_tokens', 4096)))
        except FileNotFoundError:
            pass

    def save_config(self):
        config = {
            'model_path': self.model_path.get(),
            'model_history': self.model_history,
            'gpu_count': int(self.gpu_count.get()),
            'temperature': self.temperature.get(),
            'top_p': self.top_p.get(),
            'max_seqs': int(self.max_seqs.get()),
            'gpu_memory': self.gpu_mem.get(),
            'is_instruct': self.is_instruct.get(),
            'max_gen_tokens': int(self.max_gen_tokens.get())
        }
        with open('vllm_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        messagebox.showinfo("提示", "配置已保存")

    def start_model(self):
        if self.llm is not None:
            self.llm = None
            self.start_button.config(text="启动模型")
            self.status_var.set("已停止")
            return

        def load_model():
            try:
                self.status_var.set("正在加载模型...")
                gpu_count = int(self.gpu_count.get())
                
                # 获取CUDA设备信息
                cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
                self.status_var.set(f"正在加载模型...\nCUDA可见设备: {cuda_visible_devices if cuda_visible_devices else '所有设备'}")
                
                self.llm = LLM(
                    model=self.model_path.get(),
                    trust_remote_code=True,
                    dtype="float16", # 使用float16精度，在这里可以修改为float32精度，但是需要更多显存，请量力而行。
                    gpu_memory_utilization=self.gpu_mem.get(),
                    max_model_len=int(self.max_seq_len.get()),
                    tensor_parallel_size=gpu_count,
                    enforce_eager=True
                )
                
                # 获取详细的GPU信息
                try:
                    # 获取GPU内存使用情况
                    mem_result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,memory.free', '--format=csv,noheader,nounits'], 
                        capture_output=True, text=True, check=True
                    )
                    
                    # 获取GPU计算利用率
                    util_result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=index,utilization.gpu,utilization.memory', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, check=True
                    )
                    
                    gpu_info = []
                    mem_lines = mem_result.stdout.strip().split('\n')
                    util_lines = util_result.stdout.strip().split('\n')
                    
                    for mem_line, util_line in zip(mem_lines, util_lines):
                        # 解析内存信息
                        idx, used, total, free = mem_line.split(', ')
                        # 解析利用率信息
                        _, gpu_util, mem_util = util_line.split(', ')
                        
                        info = (
                            f"GPU-{idx}:\n"
                            f"  内存: 已用 {int(used):,}MB / 总计 {int(total):,}MB (空闲: {int(free):,}MB)\n"
                            f"  利用率: GPU {gpu_util}% / 显存 {mem_util}%"
                        )
                        gpu_info.append(info)
                    
                    gpu_status = "\n".join(gpu_info)
                    tensor_parallel_info = f"\n张量并行: {gpu_count} GPUs"
                    self.status_var.set(f"模型已加载\n{gpu_status}{tensor_parallel_info}")
                    
                except Exception as e:
                    self.status_var.set(f"模型已加载（GPU信息获取失败: {str(e)}）")
                
                self.start_button.config(text="停止模型")
                
            except Exception as e:
                self.status_var.set(f"错误: {str(e)}")
                self.llm = None
                messagebox.showerror("错误", f"模型加载失败: {str(e)}")

        threading.Thread(target=load_model, daemon=True).start()

    def send_message(self):
        if self.llm is None:
            messagebox.showwarning("警告", "请先启动模型")
            return

        message = self.message_input.get("1.0", tk.END).strip()
        if not message:
            return

        self.chat_history.insert(tk.END, f"You: {message}\n\n")
        self.message_input.delete("1.0", tk.END)
        self.chat_history.see(tk.END)
        
        # 禁用发送按钮
        self.send_button.config(state='disabled')
        self.message_input.config(state='disabled')

        def generate_response():
            try:
                # 根据是否是 Instruct 模型添加提示模板
                if self.is_instruct.get():
                    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\nInstruction: {message}\n\nResponse:"
                else:
                    prompt = message

                sampling_params = SamplingParams(
                    temperature=float(self.temp_value.get()),
                    top_p=self.top_p.get(),
                    max_tokens=int(self.max_gen_tokens.get()),
                    stop=["\nHuman:", "\nAssistant:", "如果您有任何", "祝您", "希望我的回答"],
                    presence_penalty=0.2,
                    frequency_penalty=0.2
                )
                
                self.chat_history.insert(tk.END, "Assistant: ")
                
                try:
                    last_output = ""
                    is_incomplete = True
                    continuation_prompt = prompt
                    
                    while is_incomplete:
                        outputs = self.llm.generate([continuation_prompt], sampling_params)
                        current_response = ""
                        
                        # 流式输出当前部分的回答
                        for request_output in outputs:
                            output = request_output.outputs[0]
                            current_text = output.text
                            new_text = current_text[len(last_output):]
                            if new_text:
                                self.stream_token(new_text)
                                last_output = current_text
                                current_response = current_text
                        
                        # 检查回答是否完整
                        if len(current_response) < int(self.max_gen_tokens.get()) * 0.9:  # 如果生成的文本较短，说明可能已经完成
                            is_incomplete = False
                        else:
                            # 准备继续生成
                            continuation_prompt = (
                                f"{prompt}\n\n{current_response}\n\n"
                                "请继续上文未完成的内容，直接继续写，不要重复之前的内容。"
                            )
                            self.stream_token("\n[继续生成中...]\n")
                    
                    self.chat_history.insert(tk.END, "\n\n")
                    self.chat_history.see(tk.END)
                    
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        messagebox.showerror("错误", "显存不足，请减小最大序列数")
                    else:
                        raise
                
            except Exception as e:
                messagebox.showerror("错误", f"生成回复失败: {str(e)}")
            finally:
                # 重新启用发送按钮
                self.send_button.config(state='normal')
                self.message_input.config(state='normal')
                self.message_input.focus()

        threading.Thread(target=generate_response, daemon=True).start()

    def stream_token(self, token):
        """在GUI中安全地更新token"""
        def update():
            self.chat_history.insert(tk.END, token)
            self.chat_history.see(tk.END)
        self.root.after(0, update)
        self.root.update()

    def _browse_model(self):
        from tkinter import filedialog
        path = filedialog.askdirectory()
        if path:
            self.model_path.set(path)
            self._update_model_history(path)

    def _update_model_history(self, path):
        if path in self.model_history:
            self.model_history.remove(path)
        self.model_history.insert(0, path)
        if len(self.model_history) > self.max_history:
            self.model_history = self.model_history[:self.max_history]
        self.model_path['values'] = self.model_history

def main():
    root = tk.Tk()
    app = VLLMChatGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()