# 基本用法

## 可用脚本

`SST2` 在 `bin` 目录下提供了一组命令行脚本，方便在项目中快速部署不同的采样方案：

* `launch_ST_abinitio_seq.py`
* `launch_ST_pdb.py`
* `launch_sst2_abinitio_seq.py`
* `launch_sst2_pdb.py`
* `launch_REST2_small_molecule.py`

其余脚本仍处于实验阶段，请谨慎使用。

默认情况下，以上脚本采用以下力场组合与预处理步骤：

* 隐式溶剂模拟：Amber99sbnmr
* 显式溶剂模拟：Amber14SB + TIP3P 水模型
* 借助 `pdbfixer` 修复结构，并为肽链分配 **pH=7.0** 的质子化状态

如需使用不同的力场或 pH，可自行修改脚本参数。

## 查看脚本参数

所有脚本都支持 `--help` 选项。例如：

```bash
$ python bin/launch_ST_abinitio_seq.py --help
usage: launch_ST_abinitio_seq.py [-h] -seq SEQ -n NAME -dir OUT_DIR [-pad PAD]
                                 [-eq_time_impl EQ_TIME_IMPL] [-eq_time_expl\
...
```

## ST 折叠模拟

以下示例展示如何针对给定序列启动 ST 模拟：

```bash
python bin/launch_ST_abinitio_seq.py  -seq NLYIQWLKDGGPSSGRPPPS\
    -time 1000 -temp_time 4 -min_temp 280 -last_temp 600\
    -n TrpCage -dir tmp_TrpCage
```

上述命令会对 TrpCage 蛋白序列 `NLYIQWLKDGGPSSGRPPPS` 执行 ST 模拟，关键设置包括：

* 构建线性肽链初始构象；
* 盒子填充距离为 1.5 nm（默认值）；
* 氢质量重分配为 3.0 a.m.u.（默认值）；
* Langevin 摩擦系数 {math}`1.0 \; ps^{-1}`（默认值）；
* 隐式溶剂平衡 10 ns（默认值）；
* 显式溶剂平衡 10 ns（默认值）；
* 温度在 280–600 K 范围内呈指数分布；
* 温度切换间隔 4 ps；
* ST 日志输出间隔 2 ps（默认值）；
* 结果保存到 `tmp_TrpCage` 目录。

如果直接从现有 PDB 启动 ST 模拟，可以使用：

```bash
python bin/launch_ST_pdb.py  -pdb my_structure.pdb -time 1000\
    -temp_time 4 -min_temp 280 -last_temp 600 -n TrpCage\
    -dir tmp_TrpCage
```

此流程跳过隐式溶剂平衡步骤，直接在显式溶剂中平衡 10 ns，之后与前述流程相同。

## 小分子 REST2 模拟

若需对溶剂化小分子（例如含自定义 XML 力场的配体）执行 REST2 采样，可使用：

```bash
python bin/launch_REST2_small_molecule.py \
    -pdb ligand.pdb \
    -n ligand_rest2 \
    -o out_dir \
    --extra-ff ligand.xml \
    --time 50 \
    --dt 2.0 \
    --platform CPU
```

该命令将完成以下步骤：

- 复制输入结构（或在 `--use-fixer` 下调用 pdbfixer 进行修复）；
- 利用默认 1.5 nm 填充、0.15 M NaCl 建立矩形水盒；
- 自动检测溶质原子（非水/离子，亦可通过 `--solute-residues` 指定）；
- 构建并缩放溶质相互作用的 REST2 哈密顿量；
- 执行指定时长的 REST2 采样，输出前缀为 `ligand_rest2` 的 `.dcd`、`.csv` 与 `_rest2.csv` 文件。

如需加载配体专用 XML，请通过 `--extra-ff` 指定；若希望在采样前进行能量最小化，可启用 `--minimize`。

### 小分子力场参数准备

**重要提示**：与蛋白质不同，小分子不能依赖 `pdbfixer` 自动生成力场参数。在运行 REST2 之前，必须先为小分子生成 OpenMM 兼容的力场 XML 文件。

#### 方法 1：使用 ACPYPE（推荐）

ACPYPE 是基于 ANTECHAMBER 的自动参数化工具，可为有机小分子生成 GAFF 力场参数。

**安装**：
```bash
# 安装 AmberTools（包含 ANTECHAMBER）
conda install -c conda-forge ambertools

# 下载 ACPYPE
git clone https://github.com/alanwilter/acpype.git
cd acpype
python setup.py install
```

**使用示例**：
```bash
# 从 MOL2 文件生成参数（需包含所有氢原子）
acpype -i ligand.mol2 -c bcc -n 0

# 或从 PDB 生成（需先补全氢原子）
acpype -i ligand.pdb -c bcc -n 0

# 输出文件在 ligand.acpype/ 目录
# 关键文件：ligand_GMX.xml（OpenMM 力场参数）
```

**参数说明**：
- `-c bcc`：使用 AM1-BCC 电荷（速度快，精度适中）
- `-n 0`：净电荷为 0（根据分子实际电荷调整，如 `-n 1` 表示 +1 电荷）
- `-c gas`：使用 Gasteiger 电荷（更快但精度较低）

**转换为 OpenMM 并运行**：
```bash
# ACPYPE 已生成 OpenMM XML 格式
cp ligand.acpype/ligand_GMX.xml ligand_ff.xml

# 运行 REST2
python bin/launch_REST2_small_molecule.py \
    -pdb ligand.pdb \
    -n my_sim \
    -o output \
    --extra-ff ligand_ff.xml \
    --time 100
```

#### 方法 2：使用 OpenFF Toolkit

OpenFF 是新一代开源小分子力场，特别适合药物分子。

**安装**：
```bash
conda install -c conda-forge openff-toolkit openff-forcefields
```

**使用示例**：
```python
from openff.toolkit.topology import Molecule
from openff.interchange import Interchange
from openff.toolkit.typing.engines.smirnoff import ForceField

# 从 SDF/MOL2 读取分子
mol = Molecule.from_file('ligand.sdf')

# 或从 SMILES 生成
mol = Molecule.from_smiles('CCO')  # 乙醇
mol.generate_conformers(n_conformers=1)

# 应用 OpenFF 力场
ff = ForceField('openff-2.1.0.offxml')
topology = mol.to_topology()
interchange = Interchange.from_smirnoff(ff, topology)

# 导出 OpenMM XML（需要额外步骤，参考 OpenFF 文档）
# https://docs.openforcefield.org/
```

#### 方法 3：使用在线工具

**LigParGen**（适合 OPLS 力场）：
- 网址：http://zarbi.chem.yale.edu/ligpargen/
- 上传分子结构（SMILES、PDB、MOL2）
- 下载 OpenMM XML 格式参数

#### 常见问题

**Q: 小分子 PDB 必须包含氢原子吗？**
A: 是的！与蛋白质不同，小分子不能依赖 `pdbfixer` 自动添加氢。使用建模软件预先添加：
- **Avogadro**：免费分子编辑器，支持氢添加和几何优化
- **RDKit**：Python 库，可从 SMILES 生成 3D 结构
- **ChemDraw 3D**：商业软件，适合复杂药物分子

**Q: 如何确定小分子的净电荷？**
A:
- 使用量子化学软件（Gaussian、ORCA）计算
- 根据化学常识判断（如羧酸根 -1，铵离子 +1）
- 使用在线工具（ChemAxon、Marvin）预测 pKa

**Q: GAFF 和 OpenFF 哪个更好？**
A:
- **GAFF**：成熟、兼容性好、参数覆盖广，适合一般有机分子
- **OpenFF**：更现代、参数化更严格、持续更新，适合药物分子

**Q: 可以混合蛋白质和小分子力场吗？**
A: 可以！使用 `--ff amber14sb --extra-ff ligand.xml` 即可同时加载蛋白质和小分子参数。对于蛋白-配体复合物体系：
```bash
python bin/launch_REST2_small_molecule.py \
    -pdb complex.pdb \
    -n protein_ligand \
    -o output \
    --ff amber14sb \
    --extra-ff ligand_ff.xml \
    --solute-residues LIG \
    --time 100
```

**Q: 如何处理带电小分子？**
A: 使用 `--nonbonded-method cutoff-periodic` 避免 PME 伪影：
```bash
python bin/launch_REST2_small_molecule.py \
    -pdb charged_ligand.pdb \
    -n charged_sim \
    -o output \
    --extra-ff ligand_ff.xml \
    --nonbonded-method cutoff-periodic \
    --time 100
```

## SST2 折叠模拟

针对给定序列运行 SST2：

```bash
python bin/launch_sst2_abinitio_seq.py  -seq NLYIQWLKDGGPSSGRPPPS\
 -time 1000 -temp_time 4 -min_temp 280 -ref_temp 320 -last_temp 600\
 -n TrpCage -dir tmp_SST2_TrpCage -exclude_Pro_omega
```

流程说明：

* 构建线性肽链并在隐式溶剂中平衡 10 ns；
* 随后溶剂化并在显式溶剂中平衡 10 ns；
* 启动 1000 ns 的 SST2 生产模拟；
* 温度在 280–600 K 间呈指数分布，参考温度 320 K，温度切换间隔 4 ps；
* 结果写入 `tmp_SST2_TrpCage` 目录；
* `-exclude_Pro_omega` 用于在缩放过程中排除脯氨酸 {math}`\omega` 二面角。

若从 PDB 启动：

```bash
python bin/launch_sst2_pdb.py  -pdb my_structure.pdb -time 1000\
 -temp_time 4 -min_temp 280 -ref_temp 320 -last_temp 600 -n TrpCage\
 -dir tmp_SST2_TrpCage -exclude_Pro_omega
```

## 蛋白-配体复合物的 SST2

处理蛋白-配体体系时，可通过 `-chain` 指定溶质（配体）链：

```bash
python bin/launch_sst2_pdb.py  -pdb my_structure.pdb -time 1000\
 -temp_time 4 -min_temp 280 -ref_temp 320 -last_temp 600 -n Complex\
 -dir tmp_SST2_Complex -chain B
```

其中 `-chain B` 表示将 PDB 中链 ID 为 B 的原子作为溶质，其余原子视为溶剂。
