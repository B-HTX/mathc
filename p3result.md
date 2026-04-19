我已经按你第三问的框架，把问题1的动作指标和问题2的防守闭环整合成了一个“精简版问题3数据集”，并给出了可直接运行的 Python 仿真代码、结果表格和可视化图。这个精简版保留了第三问真正需要的核心变量：问题1里的 (A_i,S_i,q_i,e_i,T_i) 等攻击侧指标，以及问题2里的主防动作/辅助动作/兜底动作闭环；比赛按单场 5 分钟建模，不纳入故障与维修资源，这也和题面及你写的第三问框架一致。 

文件如下：

[第三问数据集 CSV](sandbox:/mnt/data/problem3_q3_core_dataset.csv)
[Python 代码](sandbox:/mnt/data/problem3_strategy_simulation.py)
[策略表（阶段×防守风格）](sandbox:/mnt/data/problem3_policy_table.csv)
[策略对比结果表](sandbox:/mnt/data/problem3_strategy_comparison.csv)
[分对手画像胜率表](sandbox:/mnt/data/problem3_profile_winrates.csv)
[代表性比赛日志](sandbox:/mnt/data/problem3_representative_match_log.csv)
[攻击-防守风格条件效能矩阵](sandbox:/mnt/data/problem3_attack_effectiveness_matrix.csv)

[图1：攻击收益-稳定性散点图](sandbox:/mnt/data/problem3_attack_scatter.png)
[图2：攻击动作对防守风格热图](sandbox:/mnt/data/problem3_style_attack_heatmap.png)
[图3：不同策略整体胜率柱状图](sandbox:/mnt/data/problem3_winrate_bar.png)
[图4：策略-对手画像胜率热图](sandbox:/mnt/data/problem3_profile_heatmap.png)
[图5：代表性比赛分差轨迹图](sandbox:/mnt/data/problem3_match_timeline.png)

这版模拟里，自适应策略的整体胜率最高，为 **88.96%**；固定高分策略为 **88.13%**；随机策略为 **80.00%**；腿法偏置策略为 **71.88%**。这里的胜率是基于我用你这套数据构造的蒙特卡洛比赛模型算出来的，属于“模型内胜率”，不是实测比赛胜率。

我顺手把第三问里最关键的策略结论也落到了表里。比如：

* 开局遇到**位移闪避型**，优先链偏向 **直拳 → 冲撞 → 组合拳**
* 开局遇到**守转反制型**，优先链偏向 **直拳 → 勾拳 → 组合拳**
* 中盘遇到**刚性格挡型**，优先链偏向 **前踢 → 冲撞 → 侧踢**

如果你要，我下一步可以直接继续帮你把这些结果整理成第三问论文正文里的“建模—求解—仿真—结论”那一节。
