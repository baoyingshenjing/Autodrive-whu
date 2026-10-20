"""Academic report layout shared with the first two course reports."""
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path


PREAMBLE = r"""\documentclass[UTF8,zihao=-4,fontset=windows]{ctexart}
\usepackage[a4paper,margin=2.25cm,headheight=15pt]{geometry}
\usepackage{graphicx,booktabs,tabularx,array,enumitem,xcolor,fancyhdr,titlesec,caption,hyperref,float,amsmath,listings,placeins,needspace}
\usepackage{colortbl}
\definecolor{pale}{HTML}{F3F3F3}
\definecolor{codebg}{HTML}{F6F6F6}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=black,citecolor=black,pdfborder={0 0 0},
  pdftitle={自动驾驶技术第三次实验报告},pdfauthor={杜承敖}}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.28em}
\setlength{\emergencystretch}{2em}
\renewcommand{\arraystretch}{1.18}
\setcounter{tocdepth}{2}
\captionsetup{font=small,labelfont={bf},skip=6pt}
\setlist[itemize]{leftmargin=2em,itemsep=0.2em,topsep=0.3em}
\lstset{language=Python,basicstyle=\ttfamily\footnotesize\color{black},frame=single,
  backgroundcolor=\color{codebg},rulecolor=\color{black},columns=fullflexible,
  breaklines=true,breakatwhitespace=true,keepspaces=true,showstringspaces=false,
  keywordstyle=\color{black},commentstyle=\color{black},stringstyle=\color{black},
  numberstyle=\color{black},xleftmargin=0.6em,xrightmargin=0.6em,
  aboveskip=0.45em,belowskip=0.35em,captionpos=b}
\renewcommand{\lstlistingname}{代码清单}
\titleformat{\section}{\Large\bfseries\heiti\color{black}}{\thesection}{0.7em}{}
\titleformat{\subsection}{\large\bfseries\heiti\color{black}}{\thesubsection}{0.7em}{}
\titlespacing*{\section}{0pt}{1.0em}{0.45em}
\titlespacing*{\subsection}{0pt}{0.7em}{0.3em}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small 自动驾驶技术第三次实验}
\fancyhead[R]{\small nuScenes mini 与 PointPillars 三维检测}
\fancyfoot[C]{\small\color{black}\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\widowpenalty=10000
\clubpenalty=10000
\begin{document}
\color{black}
\begin{titlepage}
  \thispagestyle{empty}
  \begin{center}
    \vspace*{1.8cm}
    {\zihao{1}\bfseries\heiti 自动驾驶技术第三次实验报告\par}
    \vspace{1.0cm}
    {\zihao{3}\bfseries nuScenes mini\par}
    \vspace{0.25cm}
    {\zihao{3}\bfseries LiDAR BEV 三维目标检测与 PointPillars\par}
    \vfill
    {\zihao{4}\renewcommand{\arraystretch}{1.75}
    \begin{tabular}{>{\bfseries}r l}
      姓名： & \underline{\makebox[9.2cm][c]{杜承敖}} \\
      学号： & \underline{\makebox[9.2cm][c]{2024302112019}} \\
      专业： & \underline{\makebox[9.2cm][c]{计算机科学与技术（雷军班）}} \\
      课程： & \underline{\makebox[9.2cm][c]{自动驾驶技术}} \\
    \end{tabular}}
    \vfill
    {\zihao{3}\bfseries 武汉大学\par}
    \vspace{0.45cm}
    {\zihao{4}2026 年 10 月 13 日\par}
    \vspace*{0.8cm}
  \end{center}
\end{titlepage}
\pagenumbering{arabic}
\tableofcontents
\clearpage
\phantomsection
\section*{摘要}
\addcontentsline{toc}{section}{摘要}
"""


def escape(text):
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
               '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
               '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(mapping.get(char, char) for char in str(text)).replace('/', r'/\allowbreak{}')


class ReportWriter:
    def __init__(self, root):
        self.root = Path(root)
        self.final = self.root / 'output/pdf'
        self.project = self.final / 'pointpillars-latex'
        self.tex = [PREAMBLE]
        self.md = ['# 自动驾驶技术第三次实验报告\n',
                   '杜承敖；2024302112019；计算机科学与技术（雷军班）；武汉大学；实验日期：2026 年 10 月 13 日。\n']
        self.section = 0
        self.table_count = 0
        self.figure_count = 0

    def frontmatter(self, abstract):
        self.paragraph(abstract)
        keywords = 'nuScenes mini；LiDAR；PointPillars；柱特征；三维目标检测'
        self.tex.append(r'\noindent\textbf{关键词：}' + escape(keywords) + '\n\n\\clearpage\n')
        self.md.append('关键词：'+keywords+'\n')

    def paragraph(self, text, style=None):
        body = escape(text).replace('\n', r'\newline{}')
        self.tex.append((r'{\small ' + body + '\\par}\n' if style == 'small' else body+'\n\n'))
        self.md.append(text+'\n')

    def heading(self, text):
        match = re.match(r'^(\d+)(?:\.(\d+))?\.?\s+(.+)$', text)
        if not match:
            raise ValueError('Heading must have an explicit source number: '+text)
        number, subsection, title = match.groups()
        self.tex.append(r'\clearpage' if number == '6' else r'\Needspace{8\baselineskip}')
        if subsection and int(number) != self.section:
            self.tex.append(r'\section{LiDAR BEV 结果展示}'+'\n')
            self.md.append('## '+number+' LiDAR BEV 结果展示\n')
            self.section = int(number)
        command = 'subsection' if subsection else 'section'
        self.tex.append('\\'+command+'{'+escape(title)+'}\n')
        self.md.append(('### ' if subsection else '## ')+text+'\n')
        if not subsection:
            self.section = int(number)
        if number == '4' and not subsection:
            self.tex.append(r'\subsection{训练设置与损失曲线}'+'\n')
            self.md.append('### 4.1 训练设置与损失曲线\n')

    def page(self):
        # Keep the algorithm's two subsections continuous; other major sections
        # may start on a fresh page just as in the reference reports.
        self.tex.append('\n\\FloatBarrier\n')

    def code(self, text, caption):
        self.tex.append(r'\Needspace{6\baselineskip}'+'\n'+r'\begin{lstlisting}[caption={'+escape(caption)+'}]\n'
                        +textwrap.dedent(text).strip()+'\n\\end{lstlisting}\n')
        self.md.append(caption+'\n\n```text\n'+textwrap.dedent(text).strip()+'\n```\n')

    def reference(self, label, url):
        self.tex.append(r'{\small\noindent '+escape(label)+r' \href{'+url+r'}{原文链接}\par}'+'\n')
        self.md.append(f'{label} [原文链接]({url})\n')

    def table(self, rows, widths=None):
        self.table_count += 1
        names = {'项目':'运行环境与数据设置', '比较维度':'LiDAR 点云与图像数据的比较',
                 '符号 / 设置':'柱特征的组织形式', '实际单帧前向记录':'固定验证帧 0 的中间特征尺寸',
                 '运行':'首轮训练和断点续训审计', '轮次':'各轮全部批次的平均训练损失',
                 '指标':'微调前后 NDS 与 mAP', '误差指标':'微调前后检测误差',
                 '固定验证帧':'两个展示阈值下的预测框数量'}
        caption = names[rows[0][0]]
        count = len(rows[0])
        weights = widths or [1]*count
        fractions = [value/sum(weights) for value in weights]
        # Account for inter-column padding before assigning paragraph widths.
        specs = ''.join(r'>{\raggedright\arraybackslash}p{'+f'{f:.5f}'+
                        r'\dimexpr\textwidth-'+str(2*count)+r'\tabcolsep\relax}' for f in fractions[:-1])
        specs += r'>{\raggedright\arraybackslash}X'
        self.tex.extend([r'\begin{table}[H]', r'\centering\small',r'\caption{'+escape(caption)+'}',
                         r'\label{tab:'+str(self.table_count)+'}',r'\begin{tabularx}{\textwidth}{'+specs+'}',
                         r'\toprule',r'\rowcolor{pale}'])
        self.tex.append(' & '.join(r'\textbf{'+escape(x)+'}' for x in rows[0])+r' \\')
        self.tex.append(r'\midrule')
        self.tex.extend(' & '.join('{'+escape(x)+'}' for x in row)+r' \\' for row in rows[1:])
        self.tex.extend([r'\bottomrule',r'\end{tabularx}',r'\end{table}'])
        self.md.append(f'表 {self.table_count}：{caption}\n')
        self.md.extend(['| '+' | '.join(str(x) for x in rows[0])+' |',
                        '| '+' | '.join('---' for _ in rows[0])+' |'])
        self.md.extend('| '+' | '.join(str(x) for x in row)+' |' for row in rows[1:])
        self.md.append('')

    def image(self, path, width, caption):
        self.figure_count += 1
        caption = re.sub(r'^图\s*\d+\s*[：:]\s*','',caption)
        asset = self.project/'images'/path.name
        asset.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,asset)
        relative = asset.relative_to(self.project).as_posix()
        factor = '0.90' if 'bev_' in path.name else '0.98'
        self.tex.extend([r'\begin{figure}[H]',r'\centering',
                         r'\includegraphics[width='+factor+r'\textwidth]{\detokenize{'+relative+'}}',
                         r'\caption{'+escape(caption)+'}',r'\label{fig:'+str(self.figure_count)+'}',r'\end{figure}'])
        self.md.append(f'![图 {self.figure_count}：{caption}]({Path(os.path.relpath(path,self.final)).as_posix()})\n')

    def finish(self):
        self.project.mkdir(parents=True,exist_ok=True)
        source = self.project/'main.tex'
        source.write_text('\n'.join(self.tex)+ '\n\\end{document}\n',encoding='utf-8')
        (self.final/'实验报告.md').write_text('\n'.join(self.md),encoding='utf-8')
        setup = self.root/'outputs/setup'
        command = shutil.which('xelatex') or r'D:\texlive\2026\bin\windows\xelatex.exe'
        environment = os.environ.copy()
        environment['TEXMFVAR'] = str(self.root/'tmp/texmf-var')
        for pass_index in range(1,4):
            log_file = setup/f'report_xelatex_pass{pass_index}.log'
            with log_file.open('w',encoding='utf-8') as console:
                result = subprocess.run([command,'-interaction=nonstopmode','-halt-on-error',
                                         '-file-line-error','-no-shell-escape','main.tex'],
                                        cwd=self.project,env=environment,stdout=console,stderr=subprocess.STDOUT)
            log = log_file.read_text(encoding='utf-8',errors='replace')
            if result.returncode:
                raise RuntimeError(log[-4500:])
            rerun = 'Rerun to get' in log or 'Label(s) may have changed' in log
            if pass_index>=2 and not rerun:
                break
        target = self.final/'自动驾驶技术第三次实验报告_杜承敖_2024302112019.pdf'
        shutil.copy2(self.project/'main.pdf',target)
        print('REPORT_OK',target)
