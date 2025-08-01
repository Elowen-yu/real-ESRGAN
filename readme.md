# real-ESRGAN说明

### real-ESRGAN批量图像超分辨率处理系统

这是一个基于 Real-ESRGAN 算法的批量图像超分辨率处理系统，提供 Web 界面让用户可以方便地上传、处理和下载高清化后的图片。

**注意**：1.单次上传总大小限制为500MB。2.一次只能一个用户上传和使用



#### 功能

1. 文件管理功能

- **拖拽上传**: 支持将图片文件直接拖拽到上传区域
- **批量选择**: 支持同时选择多个图片文件进行上传
- **文件预览**: 显示已选择文件的名称和大小信息
- **文件筛选**: 自动过滤非图片文件，只接受常见图片格式
- **批量操作**: 
  - 全选/取消全选功能
  - 批量删除选中文件
  - 单个文件删除
- **输入格式**
  - JPG/JPEG
  - PNG
  - BMP
  - TIFF



2. 图像处理功能

- **Real-ESRGAN 算法**: 使用 RealESRGAN_x2plus 模型进行图像超分辨率处理
- **批量处理**: 支持同时处理多个图片文件
- **智能分块**: 使用 600px 分块处理，避免内存溢出
- **会话管理**: 每个处理任务使用独立的会话ID，支持多用户并发



3. 下载功能

- **压缩打包**: 自动将处理结果打包成 ZIP 文件
- **一键下载**: 处理完成后可直接下载所有结果文件
- **文件管理**: 自动清理临时文件，避免存储空间浪费

- **输出格式**：与输入格式相同，但分辨率提升2倍



#### 使用

##### ubuntu的conda环境

**1.启动后端**

```bash
#1.进入后端目录（假设 app.py 在 aiImage 文件夹中）
cd aiImage

#2.安装必要的依赖（如果尚未安装）
pip install flask flask-cors werkzeug

#3.启动 Flask 后端服务
python app.py
```

后端应该会启动在 http://公网ip:5000

例如：http://129.204.81.116:5000



**2.启动前端**

- 2.1 确保`index.html` 中的 `API_BASE` 变量正确指向后端地址

```javascript
const API_BASE = 'http://129.204.81.116:5000';
```

- 2.2 启动前端

```bash
#1.进入后端目录（假设 index.html 在 aiImage 文件夹中）
cd aiImage

#2.使用 Python 内置 HTTP 服务器
python -m http.server 8000
```

访问 http://公网ip:8000 即可

例如：http://129.204.81.116:8000

若连接失败，尝试放行端口



**3.放行端口**

```bash
#1.打开端口：
firewall-cmd --permanent --add-port=8000/tcp
#2.重新载入，才能生效
firewall-cmd --reload
#3.查询端口是否开放
firewall-cmd --query-port=8000/tcp

#查看对应端口的协议
netstat -anp
```

再次访问并测试 http://公网ip:8000

http://129.204.81.116:8000



#### v1.0.0

- 初始版本发布
- 支持基本的文件上传和处理功能
- 实现Web界面和API接口
- 添加文件管理和批量操作功能

- 优化思路：

  - 并发执行

  - cron 定时脚本：定时删除数据



### real-ESRGAN大模型运行

#### 一.windows运行real-ESRGAN

miniconda管理python环境

1.在anaconda网页下载[miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)

2.下载好后打开Anaconda PowerShell prompt终端

- 1.清华园镜像配置

  - ```bash
    # 清华源配置
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
    conda config --set show_channel_urls yes
    
    #配置后需清理缓存并重试
    conda clean --all
    
    #查看镜像
    
    #测试清华源是否可访问
    ping mirrors.tuna.tsinghua.edu.cn
    ```

- 2/创建python3.9环境

  - ```bash
    conda create -n py39 python=3.9 -vv
    #-vv 参数显示详细日志，可定位具体失败原因。
    
    #创建完成后激活环境
    conda activate py39
    
    #检查版本
    python --version
    ```

  

- 3.在 Conda 环境中安装 PyTorch

```bash
#1.查看系统配置
nvidia-smi
#显卡型号NVIDIA GeForce RTX 3050
#驱动版本561.19
#CUDA版本：12.6（确认cuda可以下载什么版本）

#2.激活目标环境
conda activate py39

#3.安装PyTorch 我安装的是CUDA=12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

#4.验证PyTorch的GPU加速是否生效
# 进入Python交互模式
python
# 在出现的 >>> 提示符后逐行输入以下代码
>>> import torch
>>> print(f"PyTorch版本: {torch.__version__}")
>>> print(f"CUDA可用: {torch.cuda.is_available()}")
>>> print(f"GPU型号: {torch.cuda.get_device_name(0)}")
>>> print(f"CUDA版本: {torch.version.cuda}")
>>> exit()  # 退出交互模式

#PyTorch版本:2.5.1

#5.看torchvision版本
pip show torchvision
#0.20.1
```

监控GPU使用

```bash
nvidia-smi -l 1  # 实时刷新显存占用
```



- 4.更改torchvision（两个选择选其一即可）

4.1 选择1：因为torch从0.15起就弃用functional_tensor，改为functional

​	路径：C:\Users\yufeng\.conda\envs\py39\Lib\site-packages\basicsr\data\degradations.py

```python
# 先修改degradations.py文件的第八行
# from torchvision.transforms.functional_tensor import rgb_to_grayscale
from torchvision.transforms.functional import rgb_to_grayscale
```
4.2 选择2：安装正确版本的torchvision

```bash
#看torchvision版本
pip show torchvision
#0.20.1

# 先卸载现有版本
pip uninstall torchvision -y

# 安装兼容版本
pip install torchvision==0.15.2
```



- 5.使用real-ESRGAN

```bash
#1.拉取real-ESRGAN
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN

#2.安装依赖
# 安装 basicsr - https://github.com/xinntao/BasicSR
# 我们使用BasicSR来训练以及推断
pip install basicsr
# facexlib和gfpgan是用来增强人脸的
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop

#测试（先把input文件夹的video文件夹删掉）
#3.x2批量处理图片
python inference_realesrgan.py -n RealESRGAN_x2plus -i inputs
```



#### 二.Windows中venv环境运行real-ESRGAN

venv管理python环境

```bash
#1：创建虚拟环境
python -m venv realesrgan_env
realesrgan_env\Scripts\activate

#2：安装 PyTorch + CUDA 11.8 支持版本(CUDA本身向下兼容,只要本机CUDA版本不过低就不用管)
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 -f https://download.pytorch.org/whl/torch_stable.html

#3：安装 numpy 和依赖
pip install numpy==1.24.4
pip install tqdm opencv-python pillow scikit-image basicsr
#!!!要是显示numpy不可用,就检查一下前面执行结果是否有uninstall numpy1.24.4,下载2.xx版本,如果有,请uninstall非1.24.4版本的numpy,重新下载1.24.4版本的

#输入python回车后输入以下内容进行检查:
import torch
import numpy

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("Numpy version:", numpy.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0))


#检查应看到输出如下:
PyTorch version: 2.1.0+cu118
CUDA version: 11.8
Numpy version: 1.24.4
CUDA Available: True
Device Name: Tesla V100

#4：安装 Real-ESRGAN 本体
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install -r requirements.txt
python setup.py develop

#5：下载模型权重
#进入 Real-ESRGAN 目录，执行：
python scripts/download_pretrained_models.py

#6：测试是否成功运行
python inference_realesrgan.py -n RealESRGAN_x2plus -i inputs (要用四倍模型就直接把这个命令里的2改成4就行;;
#如果要开人脸增强就直接在此命令后加上--face_enhance;;
#如果要分块处理就在命令最后加--tile 600[这个数字代表你所分的小块的大小,分块大小越小,处理越慢,对cpu要求越低])
```


#### 三.Ubuntu中conda环境运行real-ESRGAN

conda管理python环境

先查询服务器是否下载了conda

- 1.清华园镜像配置

  - ```bash
    #1.移除所有现有 channels
    conda config --remove-key channels
    
    #2.清华源配置
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
    conda config --set show_channel_urls yes
    
    #3.移除 defaults 和 conda-forge（避免冲突）
    conda config --remove channels defaults
    conda config --remove channels conda-forge
    
    
    #4.配置后需清理缓存并重试
    conda clean --all
    
    #5.查看镜像
    conda config --show channels
    
    #5.1若查看镜像中不止清华源，编辑删除
    vim ~/.condarc
    
    #6.测试清华源是否可访问
    ping mirrors.tuna.tsinghua.edu.cn
    #或
    curl -I https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
    # 如果超时，可能是网络问题，尝试用官方源：
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    
    #6.1若ping未安装，安装ping工具
    apt update
    apt install -y iputils-ping
    
    #6.2清华园不行就用阿里源
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
    ```

- 2.创建python3.9环境

  - ```bash
    conda create -n py39 python=3.9 -vv
    #-vv 参数显示详细日志，可定位具体失败原因。
    
    #创建完成后激活环境
    conda activate py39
    
    #检查版本
    python --version
    ```
  
  

- 3.在 Conda 环境中安装 PyTorch

```bash
#1.查看系统配置
nvidia-smi
#驱动版本: 525.105.17（属于长期稳定分支）
#CUDA版本: 12.4（较新的版本，支持最新计算特性）
#GPU型号: Tesla V100-SXM2（NVIDIA数据中心级GPU）
#架构: Volta（2017年发布，专为AI/HPC优化）
#显存容量: 32GB（HBM2显存，带宽高）
#TDP功耗: 300W（需强散热支持）

#2.激活目标环境
conda activate py39

#3.安装PyTorch 我安装的是CUDA=12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

#4.验证PyTorch的GPU加速是否生效
#进入Python交互模式
python
#在出现的 >>> 提示符后逐行输入以下代码
>>> import torch
>>> print(f"PyTorch版本: {torch.__version__}")
>>> print(f"CUDA可用: {torch.cuda.is_available()}")
>>> print(f"GPU型号: {torch.cuda.get_device_name(0)}")
>>> print(f"CUDA版本: {torch.version.cuda}")
>>> exit()  # 退出交互模式

#PyTorch版本:2.5.1
```



因为torch从0.15起就弃用functional_tensor

- 4.安装正确版本的torchvision

```bash
#1.看torchvision版本
pip show torchvision
#0.20.1

#2.先卸载现有版本
pip uninstall torchvision -y

#3.安装兼容版本
pip install torchvision==0.15.2
```



- 5.拉取real-ESRGAN

```bash
#拉取real-ESRGAN
git clone https://github.com/xinntao/Real-ESRGAN.git
#进入目录
cd Real-ESRGAN
```


- 6.修改pip镜像源

```bash
#1.查看pip镜像源
pip config get global.index-url

#2.修改pip镜像源为清华镜像源
mkdir -p ~/.pip
echo -e "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\ntrusted-host = pypi.tuna.tsinghua.edu.cn" > ~/.pip/pip.conf
```



- 7.下载real-ESRGAN依赖

```bash
#说明：requirements.txt中要求下载basicsr,而basicsr又有很多要求的依赖要下，其中tb-nightly是临时发行的，老更新，镜像源没有同步更新，因此这个就要独立链接到官方源下载

#1.下载tb-nightly,临时指定官方源
pip install tb-nightly==2.21.0a20250725 -i https://pypi.org/simple

#2.测试tb-nightly能否在当前python环境工作
python -c "import tensorboard; print(tensorboard.__version__)"

#3.跳过依赖下载basicsr
pip install tb-nightly==2.21.0a20250725 -i https://pypi.org/simple

#4.查看basicsr需要什么依赖
pip show basicsr
#Requires: addict, future, lmdb, numpy, opencv-python, Pillow, pyyaml, requests, scikit-image, scipy, tb-nightly, torch, torchvision, tqdm, yapf

#5.手动下载所有依赖
pip install addict future lmdb numpy opencv-python Pillow pyyaml requests scikit-image scipy torch torchvision tqdm yapf

#若后续测试下载依赖失败，可以选择删除已安装的依赖
pip uninstall basicsr facexlib gfpgan opencv-python tqdm addict future lmdb scikit-image scipy -y
```



- 8.降级numpy和cv,以适应basicsr

```bash
#1.降级 NumPy 到 1.x 版本
pip install "numpy<2"

#2.降级 opencv-python 以兼容 numpy<2
pip install "opencv-python<4.12"  # 安装兼容 numpy 1.x 的 OpenCV 版本


#3.检查basicsr是否可正常导入 不可，则根据提示排除错误
python -c 'from basicsr.utils import logger; print("Dependencies check passed!")'
```

- 9.测试（先把input文件夹的video文件夹删掉）
```bash
##x2_plus推断（批量处理input）文件夹中的所有图片
python inference_realesrgan.py -n RealESRGAN_x2plus -i inputs
```



### 以下是我们踩过的坑!!

#### 一.不要尝试通过在任何云工作空间(cloudstudio)部署

- **百度云--飞浆AIStudio**:百度云自带的paddlepaddle框架与Real-ESRGAN所需的pytorch兼容度很低,非常低;其实根本就是禁用了pytorch,但是有一两个版本可以和低版本的paddle paddle兼容
  
- **腾讯云工作空间**:这个云工作空间分两种,一开始我们用轻量版的云工作空间(因为考虑到轻量版可以按阶梯计费,使用时长够久就比较便宜),但是后来发现了两个问题:第一.轻量版不提供公网IP,导致你的前后端都部署在云工作空间的话就需要用工具内网穿透两次,而且每一次内网穿透都会得到不同的公网ip,即使写了一键启动的脚本,也会导致经常报错或是无法自动将后端新的公网ip更新到前端代码中,导致前后端无法正确连接;第二.轻量级的带宽限制非常恶心,处理完的图像压缩包下载速度达到了惊人的200kb/s(以内.),意味着17张处理好的共1.5g的压缩包要下载20分钟,显然是远远不够的,所以我们放弃了轻量级的云工作空间;第二种是提供公网ip的,在图像上传与下载的速度上是好了很多,但是依旧是只有惊人的500kb/s.并且这个云工作空间无法提升带宽,因此最终我们被带宽卡死了,于是决定走回自己购买主机的方法.

并且用云工作空间的价格也并不便宜,如果要每天开启八小时,需要每天都有负责人通过命令行来开启系统;如果采用24小时开启系统的方法,每月成本会高达2500元,比佐糖成本高(2000元).

#### 二.不要尝试通过云服务器部署

市面上不论是哪家的云gpu服务器价格都非常高昂,最拉跨的(T4)基本也要9块以上每小时;我们没有做过考虑,因为这样下来一个月的成本甚至比直接用佐糖还贵

#### 三.Real-ESRGAN图像高清模型的选用
共两种模型,一个是2倍一个是4倍放大;
- **4倍模型**:4倍的模型对图像处理的效果非常令我们满意,但是由于处理太细,对gpu的要求非常高,大部分图片如果不分块处理(--tile xx[xx表示分块的大小]),都会轻松干爆显存,我们用的v100;32G在不分块的情况下32张图片只能处理两张,并且分块处理会极大程度上拉长处理时间,导致我们40张图片要处理14分钟,经过我们估计,如果采用4倍放大,应该最少需要4070ti super,32g显存才能较好完成图像的处理,整套主机搭下来成本保守估计9500以上;建议公司图像处理高清度需求或是处理速度需求非常大的时候再考虑采用;我们采用的是下面的2倍模型.
  
- **2倍模型**:2倍放大对gpu的要求非常非常低,经我们实测大概3050的显卡(我们用的是云工作空间的T4),8G显存,[也就是说成本甚至可以只要3000以内]依旧可以很好且快速的处理所有图像,处理速度也很可观,甚至将近20张只花了4分钟,佐糖的处理速度大约在300张30分钟,显然速度上是令人满意的,处理结果虽然在一些图片上显然不如4倍的模型,但是仍然是足够的,因为我们同时也必须考虑我们的客户群体的需求,外国人更喜欢自己真实的样子,4倍模型优化过于完美了,效果可能会适得其反,所以我们最终采用了2倍模型.
