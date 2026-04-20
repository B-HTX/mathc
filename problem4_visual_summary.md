# Problem 4 可视化升级摘要

## 已使用文件
- 动作输入：problem4_action_input_dataset(2).csv
- 对手画像：problem4_opponent_profile_dataset(2).csv
- 资源阈值：problem4_resource_threshold_dataset(2).csv
- 分画像结果：9466e302-c596-4f45-bb92-f166cc32d1df.csv

## 图形说明
- 所有总体胜率、分画像胜率、平均资源次数均直接来自用户上传的结果表聚合。
- 由于当前对话中未上传其余 4 张结果表，脚本用“阈值表 + 已有结果表”重构了场景资源相关可视化，用于论文展示。

## 总体结果
| 策略           |   BO3获胜率 |   平均总净胜分 |   平均人工复位次数 |   平均战术暂停次数 |   平均紧急维修次数 |   平均加时率 |   获胜率标准差 |
|:---------------|------------:|---------------:|-------------------:|-------------------:|-------------------:|-------------:|---------------:|
| 自适应资源策略 |    0.809375 |        28.638  |            1.425   |           0.178125 |           0.24375  |     0.009375 |      0.116536  |
| 保守资源策略   |    0.765625 |        26.0437 |            1.54375 |           0.078125 |           0.1125   |     0.01875  |      0.0648516 |
| 激进资源策略   |    0.696875 |        20.7213 |            1.5125  |           0.609375 |           0.565625 |     0.05     |      0.187187  |
| 无资源调度策略 |    0.734375 |        20.573  |            1.475   |           0        |           0        |     0.065625 |      0.12722   |

## 输出文件
- 00_contact_sheet.png
- 01_summary_dashboard.png
- 02_lollipop_errorbar_winrate.png
- 03_pictogram_waffle.png
- 04_heatmap_profile_matchup.png
- 05_waterfall_adaptive_gain.png
- 06_sunburst_scene_resource.png
- 07_dual_axis_efficiency.png
- 08_violin_profile_stability.png
- 09_hexbin_action_risk_score.png
- 10_polar_action_map.png
- 11_threshold_ribbon.png
- 12_stacked_bar_resource_mix.png
- 13_stacked_area_resource_share.png
- 14_polynomial_regression_tradeoff.png
- 15_layered_bar_by_profile.png
- 16_sankey_resource_pool.png
- 17_polar_curve_profile_adaptation.png
- problem4_action_input_dataset.csv
- problem4_opponent_profile_dataset.csv
- problem4_profile_results_bo3.csv
- problem4_resource_threshold_dataset.csv
- problem4_round_state_usage.csv
- problem4_scene_usage_rates.csv
- problem4_strategy_comparison_bo3.csv