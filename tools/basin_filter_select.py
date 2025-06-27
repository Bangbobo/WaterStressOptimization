# basin_filter_select.py（使用 summary_report.csv 计算目标改善率）

import pandas as pd
import os

# 读取简洁版 summary_report
summary_path = "results/PF_relaxed_max/summary_report.csv"
df = pd.read_csv(summary_path)

# 计算改善率
improve = []
for _, row in df.iterrows():
    try:
        fri_base = float(row["OrigFRI"])
        dri_base = float(row["OrigDRI"])
        fri_best = float(row["ParetoBestFRI"])
        dri_best = float(row["ParetoBestDRI"])

        fri_imp = (fri_base - fri_best) / fri_base if fri_base > 0 else 0
        dri_imp = (dri_base - dri_best) / dri_base if dri_base > 0 else 0
        total = fri_imp + dri_imp

        improve.append({
            "Basin": row["Basin"],
            "FRI_Improve": round(fri_imp, 4),
            "DRI_Improve": round(dri_imp, 4),
            "Total_Improve": round(total, 4),
            "NumPareto": row["NumPareto"]
        })
    except Exception as e:
        print(f"⚠️ Error processing {row['Basin']}: {e}")
        continue

# 排序并选前 N
df_result = pd.DataFrame(improve).sort_values("Total_Improve", ascending=False)
top_n = 10
df_result.head(top_n).to_csv("results/top_basin_by_summary_report.csv", index=False)
print("✅ 从 summary_report.csv 中筛选前 10 个目标函数改善最显著的盆地，已保存为 results/top_basin_by_summary_report.csv")
