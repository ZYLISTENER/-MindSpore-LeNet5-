"""
LeNet5 手写数字识别可视化窗口程序
==============================
功能：
    1. 可视化窗口界面，支持图片选择（兼容所有系统）
    2. 一键识别手写数字，实时显示识别结果和置信度
    3. 显示原始图片+处理后图片双预览
    4. 完整的错误提示和状态反馈

环境依赖：
    - mindspore >= 2.0.0
    - opencv-python >= 4.5.0
    - numpy >= 1.21.0
    - pillow >= 8.0.0
    - matplotlib >= 3.3.0
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkFont
import mindspore as ms
import numpy as np
import cv2
from PIL import Image, ImageTk
import mindspore.nn as nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================== LeNet5模型定义（保持不变） ========================
class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# ======================== 模型加载函数 ========================
def load_trained_model(ckpt_path="./lenet/lenet-1_1875.ckpt"):
    network = LeNet5(num_class=10)
    try:
        param_dict = ms.load_checkpoint(ckpt_path)
        ms.load_param_into_net(network, param_dict)
        network.set_train(False)
        return network
    except Exception as e:
        raise Exception(f"模型加载失败：{str(e)}")

# ======================== 图片预处理函数 ========================
def preprocess_image_colorfixed(img_path, target_size=32):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        7, 1
    )
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_digit = np.zeros_like(thresh)
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled_digit, [max_contour], -1, 255, thickness=cv2.FILLED)
    
    white_ratio = np.sum(filled_digit == 255) / filled_digit.size
    if white_ratio < 0.05:
        filled_digit = thresh
    if white_ratio > 0.95 or white_ratio < 0.05:
        filled_digit = 255 - filled_digit
    
    img_resized = cv2.resize(filled_digit, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(np.expand_dims(img_normalized, axis=0), axis=0)
    img_tensor = ms.Tensor(img_expanded, dtype=ms.float32)
    
    return img_tensor, img_resized

# ======================== 预测函数 ========================
def predict_digit_gui(img_path, model):
    """适配GUI的预测函数，返回详细结果"""
    try:
        img_tensor, img_show = preprocess_image_colorfixed(img_path)
        output = model(img_tensor)
        pred_prob = ms.ops.Softmax(axis=1)(output)
        pred_prob_np = pred_prob.asnumpy()[0]
        pred_label = np.argmax(pred_prob_np)
        confidence = pred_prob_np[pred_label]
        
        return {
            "success": True,
            "pred_label": int(pred_label),
            "confidence": float(confidence),
            "probabilities": pred_prob_np.tolist(),
            "processed_img": img_show
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# ======================== GUI主窗口类（修复拖放问题+新增处理后图片显示） ========================
class LeNetDigitRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LeNet5 手写数字识别系统")
        self.root.geometry("1000x700")  # 稍微放大窗口，适配双图片显示
        self.root.resizable(True, True)
        
        # 设置字体
        self.font_title = tkFont.Font(family="微软雅黑", size=16, weight="bold")
        self.font_normal = tkFont.Font(family="微软雅黑", size=12)
        self.font_result = tkFont.Font(family="微软雅黑", size=20, weight="bold")
        
        # 初始化变量
        self.model = None
        self.current_img_path = None
        self.processed_img = None
        self.processed_photo = None  # 保存处理后图片的引用
        
        # 创建UI界面
        self._create_widgets()
        
        # 加载模型（后台加载）
        self._load_model_async()

    def _create_widgets(self):
        """创建所有UI组件（新增处理后图片显示区域）"""
        # 顶部标题栏
        frame_title = ttk.Frame(self.root)
        frame_title.pack(fill=tk.X, padx=20, pady=10)
        
        label_title = ttk.Label(frame_title, text="LeNet5 手写数字识别系统", font=self.font_title)
        label_title.pack()
        
        # 主要内容区
        frame_main = ttk.Frame(self.root)
        frame_main.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # 左侧图片显示区（拆分为原始图片+处理后图片）
        frame_left = ttk.Frame(frame_main, width=450)
        frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # 原始图片显示区域
        self.frame_original = ttk.LabelFrame(frame_left, text="原始图片", padding=10)
        self.frame_original.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.label_original_img = ttk.Label(self.frame_original, text="\n\n点击下方按钮选择图片\n\n支持格式：jpg/png/bmp", 
                                          font=self.font_normal)
        self.label_original_img.pack(fill=tk.BOTH, expand=True)
        
        # 处理后图片显示区域
        self.frame_processed = ttk.LabelFrame(frame_left, text="处理后图片（32×32）", padding=10)
        self.frame_processed.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.label_processed_img = ttk.Label(self.frame_processed, text="\n\n识别后显示预处理结果", 
                                            font=self.font_normal)
        self.label_processed_img.pack(fill=tk.BOTH, expand=True)
        
        # 按钮区域
        frame_buttons = ttk.Frame(frame_left)
        frame_buttons.pack(fill=tk.X, pady=10)
        
        self.btn_select = ttk.Button(frame_buttons, text="选择图片", command=self._select_image)
        self.btn_select.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.btn_recognize = ttk.Button(frame_buttons, text="开始识别", command=self._recognize_digit, state=tk.DISABLED)
        self.btn_recognize.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        
        # 右侧结果显示区
        frame_right = ttk.Frame(frame_main, width=450)
        frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # 识别结果显示
        frame_result = ttk.LabelFrame(frame_right, text="识别结果", padding=10)
        frame_result.pack(fill=tk.X, pady=10)
        
        self.label_result = ttk.Label(frame_result, text="等待识别...", font=self.font_result, foreground="gray")
        self.label_result.pack(pady=20)
        
        self.label_confidence = ttk.Label(frame_result, text="置信度：--", font=self.font_normal)
        self.label_confidence.pack(pady=5)
        
        # 概率分布图表
        frame_chart = ttk.LabelFrame(frame_right, text="类别概率分布", padding=10)
        frame_chart.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 创建matplotlib图表
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.ax.set_ylim(0, 1.1)
        self.ax.set_xticks(range(10))
        self.ax.set_xlabel("数字")
        self.ax.set_ylabel("概率")
        self.ax.set_title("请选择图片并识别")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_chart)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 状态信息栏
        frame_status = ttk.Frame(self.root)
        frame_status.pack(fill=tk.X, padx=20, pady=5)
        
        self.label_status = ttk.Label(frame_status, text="状态：初始化中...", font=self.font_normal, foreground="blue")
        self.label_status.pack(anchor=tk.W)

    def _load_model_async(self):
        """异步加载模型（避免UI卡顿）"""
        def load_model():
            try:
                # 请修改为你的模型权重路径
                self.model = load_trained_model(ckpt_path="D:\\mindspore\\checkpoints\\lenet_mnist_final_2-5_937.ckpt")
                self.label_status.config(text="状态：模型加载完成，就绪", foreground="green")
                self.btn_recognize.config(state=tk.NORMAL if self.current_img_path else tk.DISABLED)
            except Exception as e:
                self.label_status.config(text=f"状态：模型加载失败 - {str(e)}", foreground="red")
                messagebox.showerror("模型加载失败", str(e))
        
        # 使用after方法在后台加载模型
        self.root.after(100, load_model)

    def _select_image(self):
        """选择图片文件（核心导入方式）"""
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self._display_image(file_path)
            # 清空上一次的处理后图片
            self.label_processed_img.config(image="", text="\n\n识别后显示预处理结果")
            self.processed_photo = None

    def _display_image(self, file_path):
        """显示选中的原始图片"""
        try:
            # 保存当前图片路径
            self.current_img_path = file_path
            
            # 调整图片大小以适应显示区域
            img = Image.open(file_path)
            img.thumbnail((400, 250), Image.Resampling.LANCZOS)  # 调整原始图片显示尺寸
            
            # 转换为Tkinter可用格式
            photo = ImageTk.PhotoImage(img)
            
            # 更新原始图片显示
            self.label_original_img.config(image=photo, text="")
            self.label_original_img.image = photo  # 保持引用，防止被垃圾回收
            
            # 更新状态
            self.label_status.config(text=f"状态：已加载图片 - {file_path}", foreground="blue")
            
            # 启用识别按钮
            if self.model:
                self.btn_recognize.config(state=tk.NORMAL)
                
        except Exception as e:
            messagebox.showerror("显示失败", f"图片显示失败：{str(e)}")

    def _display_processed_image(self, processed_img):
        """显示预处理后的图片"""
        try:
            # 将numpy数组转换为PIL Image（32×32）
            img_pil = Image.fromarray(processed_img)
            # 放大显示（32×32太小，放大到150×150便于查看）
            img_pil = img_pil.resize((150, 150), Image.Resampling.NEAREST)
            # 转换为Tkinter可用格式
            self.processed_photo = ImageTk.PhotoImage(img_pil)
            
            # 更新处理后图片显示
            self.label_processed_img.config(image=self.processed_photo, text="")
            self.label_processed_img.image = self.processed_photo  # 保持引用
            
        except Exception as e:
            messagebox.warning("处理后图片显示失败", f"处理后图片显示失败：{str(e)}")

    def _recognize_digit(self):
        """执行数字识别（新增处理后图片显示）"""
        if not self.current_img_path or not self.model:
            messagebox.showwarning("警告", "请先选择图片并确保模型加载完成")
            return
        
        try:
            # 更新状态
            self.label_status.config(text="状态：正在识别...", foreground="orange")
            self.root.update()  # 刷新UI
            
            # 执行识别
            result = predict_digit_gui(self.current_img_path, self.model)
            
            if result["success"]:
                # 更新结果显示
                self.label_result.config(
                    text=f"识别结果：{result['pred_label']}",
                    foreground="red"
                )
                self.label_confidence.config(
                    text=f"置信度：{result['confidence']:.4f} ({result['confidence']*100:.2f}%)"
                )
                
                # 更新概率分布图
                self._update_chart(result["probabilities"], result["pred_label"])
                
                # 保存并显示处理后的图片
                self.processed_img = result["processed_img"]
                self._display_processed_image(self.processed_img)
                
                # 更新状态
                self.label_status.config(
                    text=f"状态：识别完成 - 数字{result['pred_label']} (置信度{result['confidence']:.4f})",
                    foreground="green"
                )
            else:
                raise Exception(result["error"])
                
        except Exception as e:
            self.label_status.config(text=f"状态：识别失败 - {str(e)}", foreground="red")
            messagebox.showerror("识别失败", str(e))

    def _update_chart(self, probabilities, pred_label):
        """更新概率分布图表"""
        # 清空原有图表
        self.ax.clear()
        
        # 绘制柱状图
        bars = self.ax.bar(range(10), probabilities, color='skyblue')
        
        # 高亮预测结果
        bars[pred_label].set_color('red')
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 设置图表属性
        self.ax.set_ylim(0, 1.1)
        self.ax.set_xticks(range(10))
        self.ax.set_xlabel("数字")
        self.ax.set_ylabel("概率")
        self.ax.set_title(f"识别结果：{pred_label} (置信度{probabilities[pred_label]:.4f})")
        
        # 刷新画布
        self.canvas.draw()

# ======================== 程序入口 ========================
if __name__ == "__main__":
    # 设置MindSpore上下文
    ms.set_context(
        mode=ms.GRAPH_MODE,  # 静态图模式（推理性能更优）
        device_target="CPU"  # CPU推理（无需GPU/昇腾环境）
    )
    
    # 创建并运行GUI
    root = tk.Tk()
    
    # 设置TLabelframe样式
    style = ttk.Style()
    style.configure("TLabelframe", borderwidth=2, relief="solid")
    
    app = LeNetDigitRecognitionGUI(root)
    root.mainloop()