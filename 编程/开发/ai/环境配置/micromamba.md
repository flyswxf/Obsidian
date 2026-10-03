`micromamba` 类似于 `conda`，并且兼容 Conda 环境。已经通过 Conda 创建的环境，也可以使用
`micromamba` 管理。

## 安装 micromamba

```powershell
Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1 -UseBasicParsing).Content)
```

安装程序会在 C 盘创建一个默认环境目录。该环境为空，通常不会占用明显空间。

## Conda 换源

Conda 的配置文件是用户目录下的 `~\.condarc`。可以直接修改该文件为清华大学镜像：

```yaml
channels:
  - defaults
  - conda-forge
show_channel_urls: true
channel_priority: flexible

default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2

custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

也可以使用命令配置：

```powershell
conda config --add channels defaults
conda config --add channels conda-forge
conda config --set show_channel_urls yes
conda config --set channel_priority flexible
conda config --add default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
conda config --set custom_channels.conda-forge https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
conda config --set custom_channels.msys2 https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

## micromamba 使用同一配置

`micromamba` 默认也会读取 `~\.condarc`。查看配置文件位置和最终配置：

```powershell
micromamba config list --sources
```

如果使用命令配置 `micromamba`，也可以执行：

```powershell
micromamba config append channels defaults
micromamba config append channels conda-forge
micromamba config set show_channel_urls true
micromamba config set channel_priority flexible
```

## 源说明

- `pkgs/main`：Anaconda 主源。
- `pkgs/r`：R 语言相关软件包。
- `conda-forge`：社区维护的第三方软件包源。
- `msys2`：Windows 编译工具链源，需要编译 C/C++ 包时使用。
- `defaults`：Conda 默认频道名称，实际地址由 `default_channels` 指向清华镜像。

安装软件包时，Conda 会按 `channels` 中的顺序搜索。一般不需要设置
`channel_priority strict`，否则可能导致依赖解析失败；出现该问题时改回：

```powershell
conda config --set channel_priority flexible
```

验证换源是否生效：

```powershell
conda config --show-sources
conda config --show channels
conda search numpy
```
