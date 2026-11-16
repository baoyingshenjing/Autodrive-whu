from pathlib import Path
import json
h=json.loads(Path('outputs/run30/history.json').read_text()); e=json.loads(Path('outputs/run30/evaluation.json').read_text())
rows='\n'.join(f"{x['epoch']} & {x['train_mse']:.6f} & {x['val_mse']:.6f} & {x['seconds']:.1f} \\\\" for x in h)
t=r'''\documentclass[UTF8,zihao=-4,fontset=windows]{ctexart}
\usepackage[a4paper,margin=2.35cm,headheight=15pt]{geometry}
\usepackage{graphicx,booktabs,amsmath,amssymb,hyperref,fancyhdr,float,listings,xcolor,array}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=black}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{自动驾驶技术第四次实验}\fancyhead[R]{端到端自动驾驶}\fancyfoot[C]{\thepage}
\setlength{\parskip}{0.32em}\setlength{\emergencystretch}{2em}
\lstset{language=Python,basicstyle=\ttfamily\small,frame=single,backgroundcolor=\color{black!3},breaklines=true,columns=fullflexible,showstringspaces=false,keepspaces=true}
\begin{document}
\begin{titlepage}\centering\vspace*{2cm}{\zihao{1}\heiti 自动驾驶技术第四次实验报告\par}\vspace{1cm}{\zihao{3}\heiti 端到端自动驾驶：基于 NVIDIA 网络的行为克隆\par}\vfill
{\zihao{4}\renewcommand{\arraystretch}{1.8}\begin{tabular}{rl}姓名：&杜承敖\\学号：&2024302112019\\专业：&计算机科学与技术（雷军班）\\实验日期：&2026年11月9日\end{tabular}}\vfill\end{titlepage}
\setcounter{tocdepth}{1}\tableofcontents\newpage
\section{实验目的与整体流程}
本实验利用课程提供的 Udacity 一号赛道驾驶记录，训练从前向摄像头图像预测转向的卷积神经网络，理解行为克隆、数据增强、模型训练和推理服务的完整链路。实验目标包括：解释图像到转角的映射；分析模型的尺寸变化和损失函数；检查训练与推理的预处理一致性；记录真实训练及验证结果；尝试连接模拟器。
\subsection{任务边界}
输入是单帧彩色图像，输出是一个归一化转向标量。网络没有直接预测油门、制动或轨迹；推理脚本另依据当前速度与转角计算油门。训练数据中的转角来自驾驶记录，因此本质是监督学习，不是与环境交互获得奖励的强化学习。
\subsection{实验流程}
数据准备按驾驶记录切分训练和验证集；训练端读取三个相机并构造转向标签，裁剪图像、归一化并转换张量；网络输出转角，均方误差驱动反向传播；每轮在中央相机验证集计算误差并保存权重。推理端接收图像，执行同样的图像变换，输出转角，结合速度生成油门，经 SocketIO 返回。
\begin{center}驾驶记录 $\rightarrow$ 数据变换 $\rightarrow$ CNN $\rightarrow$ 转角预测 $\rightarrow$ 损失与更新\end{center}
实验实际完成 30 轮训练、验证集复算和合成遥测通信。模拟器窗口未能激活，尚无真实闭环行驶结果。下文区分已测结果、理论解释和仍需开展的测试。
\newpage
\section{端到端驾驶与行为克隆原理}
\subsection{从感知到控制的映射}
传统模块化系统通常分别估计道路、车辆与行人，再进行预测、规划和控制。端到端方法用可训练函数 $f_\theta$ 将输入直接映射为控制量：
\begin{equation}\hat{a}_t=f_\theta(I_t),\qquad (I_t,a_t)\in\mathcal D,\end{equation}
其中 $I_t$ 为第 $t$ 帧图像，$a_t$ 为记录的转向标签，$\theta$ 是网络参数。这里所谓端到端指输入到转向的可微映射；实际系统仍依赖摄像头、油门控制器、通信与仿真环境，并非所有驾驶决策都由该网络完成。
\subsection{监督学习与分布偏移}
行为克隆以驾驶记录作为示范，使预测接近示范动作。若训练图像主要来自车道中央，模型在中央附近的误差可能很小；闭环运行时预测误差会改变车辆位置，下一帧可能来自训练中少见的偏离位置。此时误差会影响后续输入，甚至逐步累积。随机抽帧验证只能比较记录帧上的输出，无法直接检验这种反馈过程。
左右相机可近似提供偏离中心的视角，并对标签加入方向修正，试图提供回到中心的学习信号。但左右摄像头的视差并不等价于真实车辆偏移后的完整状态；固定修正值也未考虑速度、曲率等因素。它是一种教学中的近似增强，而不是保证闭环恢复的充分条件。
\subsection{单帧模型的限制}
单帧模型没有显式速度、历史运动和时序状态，不易区分外观相似但运动趋势不同的场景。遮挡、光照变化或未知道路也可能使输入偏离训练分布；因此需要独立场景和闭环评估来补充离线误差。数据中原始中央相机记录的转角为零的样本占 58.56\%，整体误差可能更受常见直行帧影响。
\begin{figure}[H]\centering\includegraphics[width=.70\textwidth,height=3.2cm,keepaspectratio]{figures/steering_hist.png}\caption{全部 32,443 条记录的中央相机原始转角分布}\end{figure}
\newpage
\section{数据集与划分}
数据集中共有 32,443 条驾驶记录，每条记录含中央、左、右相机路径及转向等字段。图像共 97,329 张，CSV 中的原始路径来自录制环境，读取时保留文件名并重定位到本地 \texttt{dataset/IMG}。
\begin{table}[H]\centering\caption{数据划分与样本使用}\begin{tabular}{lrrl}\toprule 集合 & 记录数 & 图像样本数 & 使用方式\\\midrule 训练 & 25,954 & 77,862 & 中、左、右相机及随机翻转\\验证 & 6,489 & 6,489 & 中央相机，无随机增强\\\bottomrule\end{tabular}\end{table}
\subsection{固定划分}
随机种子为 42，按记录而不是按图像划分，避免同一记录的不同视角同时进入训练和验证。以下摘自实际训练脚本：
\begin{lstlisting}
rows = list(csv.reader((data / 'driving_log.csv').open(
    newline='', encoding='utf-8-sig')))
order = np.random.RandomState(42).permutation(len(rows))
n = int(0.8 * len(rows))
train = [rows[i] for i in order[:n]]
val = [rows[i] for i in order[n:]]
\end{lstlisting}
固定种子便于重新生成同一划分。验证集既用于每轮监控，也用于选择最优检查点，因此它不是独立测试集。记录随机划分无法隔离同一段视频中相邻的相似帧，验证结果可能偏乐观。更严格的实验应按完整录制片段或场景划分，并保留从未参与选模的测试集。
\begin{figure}[H]\centering\includegraphics[width=.94\textwidth,height=3.0cm,keepaspectratio]{figures/camera_views.png}\caption{同一条记录的中央、左、右相机图像示例（原始中央转角 $-0.2$）}\end{figure}
\newpage
\section{图像预处理与数据增强}
\subsection{几何裁剪和数值变换}
原始图像为 $160\times320\times3$，保留第 65 行至倒数第 25 行之前，得到 $70\times320\times3$。裁剪减少天空和车头等区域，但若重要路况位于被裁剪区域，则信息也会丢失。OpenCV 读入顺序为 BGR，像素值映射为 $x'=x/127.5-1$，再将 HWC 转换为 CHW，供卷积层使用。
\begin{lstlisting}
img = cv2.imread(str(path))
if img is None:
    raise FileNotFoundError(path)
img = img[65:-25]
img = np.ascontiguousarray(
    (img.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1))
\end{lstlisting}
\subsection{三个视角及水平翻转}
训练时每条记录构造三个样本：中央相机使用原始转向，左相机加 0.4，右相机减 0.4。图像以 50\% 概率水平翻转，同时将标签乘以 $-1$，以保持左右转向对应关系。验证集只用中央相机，不翻转。
\begin{lstlisting}
angle = float(row[3]) + (0, .4, -.4)[camera]
if self.train and random.random() < .5:
    img = cv2.flip(img, 1)
    angle = -angle
\end{lstlisting}
修正后的标签可能超出原始标签范围，网络末层也没有强制限制输出范围；因此不能假定输出必然位于 $[-1,1]$。训练侧和服务侧的颜色顺序、裁剪、尺度及通道布局必须一致，否则即使张量尺寸正确也可能出现系统性预测偏差。
\begin{figure}[H]\centering\includegraphics[width=.82\textwidth,height=3.1cm,keepaspectratio]{figures/crop.png}\caption{裁剪示例：左图黄色线条标出保留范围，右图为实际网络视野}\end{figure}
\newpage
\section{网络结构与特征尺寸}
\subsection{卷积计算}
二维卷积的输出长度由 $L_{out}=\lfloor (L_{in}+2P-K)/S\rfloor+1$ 决定，其中 $K$ 为卷积核尺寸、$S$ 为步长、$P$ 为填充。本网络各层均无填充。输入为 $3\times70\times320$，尺寸变化如下：
\begin{table}[H]\centering\caption{网络各层特征尺寸}\begin{tabular}{lccc}\toprule 层 & 卷积核 / 步长 & 通道数 & 高 $\times$ 宽\\\midrule 输入 & -- & 3 & $70\times320$\\卷积 1 & $5/2$ & 24 & $33\times158$\\卷积 2 & $5/2$ & 36 & $15\times77$\\卷积 3 & $5/2$ & 48 & $6\times37$\\卷积 4 & $3/1$ & 64 & $4\times35$\\卷积 5 & $3/1$ & 64 & $2\times33$\\展平 & -- & 4224 & --\\\bottomrule\end{tabular}\end{table}
例如第一层高度为 $\lfloor(70-5)/2\rfloor+1=33$，宽度为 $\lfloor(320-5)/2\rfloor+1=158$。连续卷积逐步扩大有效感受野并降低空间分辨率；最后展平为 $64\times2\times33=4224$ 个数，再依次进入 100、50、10、1 维全连接层。
\subsection{非线性与正则化}
多数卷积层和前两层全连接层使用 ELU。实现中第五个卷积后接 Dropout，最后两层线性层之间没有额外 ELU；应以实际代码为准。Dropout 训练时随机屏蔽部分特征，验证和推理使用 \texttt{eval()} 关闭随机屏蔽，确保输出稳定。
\begin{figure}[H]\centering\includegraphics[width=.96\textwidth,height=2.5cm,keepaspectratio]{figures/architecture.png}\caption{模型前向流程与主要特征尺寸示意}\end{figure}
\newpage
\section{网络实现与前向传播}
以下摘录来自实际使用的 \texttt{NetworkNvidia}。图像的归一化在数据加载阶段完成，不在网络内部重复执行。
\begin{lstlisting}
self.conv_layers = nn.Sequential(
    nn.Conv2d(3, 24, 5, stride=2), nn.ELU(),
    nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
    nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
    nn.Conv2d(48, 64, 3), nn.ELU(),
    nn.Conv2d(64, 64, 3), nn.Dropout(0.5))
self.linear_layers = nn.Sequential(
    nn.Linear(64 * 2 * 33, 100), nn.ELU(),
    nn.Linear(100, 50), nn.ELU(),
    nn.Linear(50, 10), nn.Linear(10, 1))
\end{lstlisting}
前向传播先计算卷积特征，再展平并计算单一转角。这里的 \texttt{view} 假定输入元素顺序已正确变为 CHW；仅对 HWC 数组强行改变形状不会完成通道重排。
\begin{lstlisting}
def forward(self, input):
    input = input.view(input.size(0), 3, 70, 320)
    output = self.conv_layers(input)
    output = output.view(output.size(0), -1)
    return self.linear_layers(output)
\end{lstlisting}
模型直接回归连续值，训练中没有分类阈值，也没有显式车道线或目标检测框。卷积核的中间特征由转角监督间接学习；不能仅凭网络结构断定某个通道一定对应车道线或车辆。
\newpage
\section{损失函数与训练策略}
\subsection{均方误差}
对 $N$ 个样本，预测值 $\hat{a}_i$ 与标签 $a_i$ 的均方误差为 $\mathrm{MSE}=N^{-1}\sum_i(\hat a_i-a_i)^2$。平方项使较大的转角误差得到更高权重；对应的 RMSE 为 $\sqrt{\mathrm{MSE}}$，与标签同量纲。MSE 数值不是错误百分比，也不直接等于行驶偏离距离。
\subsection{优化与更新}
本机使用 RTX 4060 Laptop GPU、PyTorch 2.0.1+cu118；批大小 64，Adam 初始学习率 $10^{-4}$，权重衰减 $10^{-5}$。代码中按批求和后除以批大小反向传播，并按样本数计算整轮均值，避免最后一个不满批次与其他批次等权。
\begin{lstlisting}
model.train()
for x, y in trainloader:
    x, y = x.to(device), y.to(device)
    opt.zero_grad(set_to_none=True)
    loss = lossfn(model(x), y)  # reduction='sum'
    (loss / len(x)).backward()
    opt.step()
    total += loss.item()
    count += len(x)
train_mse = total / count
\end{lstlisting}
训练依次遍历三个视角样本，样本经过随机打乱。本实现每批进行一次更新；上游原训练器按三个视角分别更新三次，因此训练过程并非原脚本的逐步复现。学习率调度节点为 30 和 50；记录的本次 1 至 30 轮训练学习率均为 $10^{-4}$。
\newpage
\section{验证、检查点与续训}
\subsection{验证模式}
验证关闭梯度计算和 Dropout，并按样本数汇总中央相机误差：
\begin{lstlisting}
model.eval()
vt, vc = 0.0, 0
with torch.no_grad():
    for x, y in valloader:
        x, y = x.to(device), y.to(device)
        vt += lossfn(model(x), y).item()
        vc += len(x)
val_mse = vt / vc
\end{lstlisting}
训练数据含三个视角和随机翻转，验证数据仅含中央视角；所以两条曲线对应不同数据分布，不能仅据两条曲线的绝对差值断定过拟合。观察验证曲线在多轮上的变化、使用未参与选模的场景测试以及闭环表现，才有助于判断泛化。
\subsection{检查点与续训}
先训练 5 轮，再加载第 5 轮检查点继续到累计 30 轮。检查点包含模型参数、Adam 状态、学习率调度状态及轮次；另保留验证误差最小的权重。
\begin{lstlisting}
ck = torch.load(resume, map_location=device)
model.load_state_dict(ck['state_dict'])
opt.load_state_dict(ck['optimizer'])
sched.load_state_dict(ck['scheduler'])
start = ck['epoch']
\end{lstlisting}
续训重新设置随机种子，未恢复中断时的 Python、NumPy、PyTorch 和数据加载器随机数状态，因此并非不中断训练的逐位等价结果。随机种子能固定数据划分，但不能单独保证所有 GPU 运算完全确定。
\newpage
\section{训练记录}
\begin{table}[H]\centering\caption{30 轮实际训练与验证结果}\small\begin{tabular}{rrrr}\toprule 轮次 & 训练 MSE & 验证 MSE & 耗时 / 秒\\\midrule
ROWS
\bottomrule\end{tabular}\end{table}
\newpage
\section{结果分析}
\begin{figure}[H]\centering\includegraphics[width=.83\textwidth]{../../outputs/run30/loss.png}\caption{实际训练和验证损失}\end{figure}
训练 MSE 从第 1 轮的 0.184793 降至第 30 轮的 0.075481；验证 MSE 从 0.106015 降至 0.077897，中间存在波动。第 30 轮是这次记录中的验证最优轮次；重新加载检查点，在相同 6,489 张验证图像上得到 MSE=EVALMSE、MAE=EVALMAE、RMSE=EVALRMSE。
MAE 是绝对误差均值，RMSE 对较大误差更敏感；二者均以归一化转向标签为单位。第 1 轮耗时明显较长，可能受到首次读取、进程启动及系统缓存影响；记录不构成单独的性能基准。损失下降表明模型在本次划分上拟合了部分映射，但不能由此推出车辆能够完成一圈。
\newpage
\section{预测分布与误差解释}
\begin{figure}[H]\centering\includegraphics[width=.68\textwidth]{../../outputs/run30/predictions.png}\caption{验证集预测值与记录值的二维频次（对数色标）}\end{figure}
横轴是记录转角，纵轴是预测转角，虚线表示两者相等。点越接近虚线，单帧误差越小；远离虚线表示该帧的预测与记录差异较大。图中使用对数色标，以同时显示高频区域和稀疏区域；颜色不代表行驶风险等级。
驾驶记录可能存在直行或小转角占比偏高的问题，此时总体 MSE 对少量大转角、恢复场景的表现不够敏感。进一步分析可按转角区间分别统计 MAE 和样本数，再抽查弯道、遮挡与偏离车道样本。当前图表和指标均来自用于选模的同一验证集，不是独立测试集。
\newpage
\section{推理服务与通信验证}
\subsection{训练和推理一致性}
模拟器提供 RGB 图像；服务端裁剪后转换为 BGR，再进行同样的归一化和通道重排。实验使用同一图像检查两条路径的数值一致，得到 $3\times70\times320$ 张量，并检查前向输出有限。该检查能发现常见预处理不一致，但不能证明模型驾驶安全。
\begin{lstlisting}
image_array = image_array[65:-25, :, :]
image_array = image_array[:, :, ::-1]  # RGB to BGR
image_array = transformations(image_array)  # HWC to CHW
image_tensor = torch.Tensor(image_array).view(1, 3, 70, 320)
steering_angle = model(image_tensor).view(-1).data.numpy()[0]
\end{lstlisting}
\subsection{油门与通信}
实际服务使用 $1.2-\hat a^2-(v/10)^2$ 计算油门，并经 SocketIO 发送转角和油门。代码虽定义 PI 控制器，但其调用被注释，故本次不能称为 PI 控制。速度来自遥测，不是网络预测量；公式输出也可能超出常见油门范围，具体行为还取决于仿真器对控制信号的处理。
使用本地合成客户端发送记录图像和速度，服务返回了转角与油门，证明该次请求的解码、模型推理与响应链路可运行。它不是模拟器生成的遥测，也不能证明模拟器版本兼容或车辆能够沿赛道行驶。
\newpage
\section{闭环测试、问题分析与总结}
\subsection{闭环测试状态}
推理服务启动后尝试运行 Udacity 模拟器，配置窗口未能被操作工具激活，因此没有取得模拟器实际连接和行驶数据。不能将合成遥测响应描述为闭环成功，也不能声称满足指导书的模拟器验收条件。
后续应先启动服务，再打开模拟器选择一号赛道和自主模式；记录连接日志、转角、油门、速度和画面，观察车辆是否持续行驶、何时偏离或卡住。多次从相同初始条件运行，并记录运行时间、完成圈数及失败位置，比单次截屏更有解释力。
\subsection{问题与改进}
第一，随机划分的相邻帧相关，且验证集参与选模；应按完整驾驶片段划分训练、验证与独立测试。第二，固定左右修正值只近似偏移恢复，可研究更有代表性的恢复数据，但须独立验证标签方向和幅值。第三，误差可能在闭环中累积，应统计偏离、碰撞和停滞，而不只看离线 MSE。第四，模型只用单帧图像，速度和历史信息可作为后续扩展输入。
\subsection{实验总结}
本次完成数据处理、网络训练、30 轮损失记录、最佳权重复算和合成通信检查。网络以图像回归转向，展示了端到端行为克隆的基本流程；实验也表明训练损失、验证误差与闭环能力是不同层面的证据。真实模拟器测试仍是当前未完成环节。
\end{document}'''
for a,b in {'ROWS':rows,'EVALMSE':f"{e['mse']:.6f}",'EVALMAE':f"{e['mae']:.6f}",'EVALRMSE':f"{e['rmse']:.6f}"}.items(): t=t.replace(a,b)
Path('output/pdf/report.tex').write_text(t,encoding='utf-8')
