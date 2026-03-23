import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1. Raw data (as given)
# -----------------------------
data = {
    "scale_factor": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    # plan_obj_box_col (will be converted to %)
    "plan_obj_box_col_1": [0.00019538882375928098, 0.0003907013088493846, 0.0001953506544246923,
                           0.0001953506544246923, 0.0001953506544246923, 0.0001953506544246923,
                           0.00019584802193497845],
    "plan_obj_box_col_2": [0.0025400547088706526, 0.0017581558898222308, 0.0017581558898222308,
                           0.0019535065442469234, 0.0015628052353975385, 0.0017581558898222308,
                           0.0017626321974148062],
    "plan_obj_box_col_3": [0.009769441187964049, 0.008986130103535847, 0.008790779449111155,
                           0.01093963664778277, 0.010158234030084002, 0.011134987302207463,
                           0.008029768899334117],
    # plan_boundary (will be converted to %)
    "plan_boundary_1": [0.007033997655334115, 0.006446571596014847, 0.005860519632740769,
                        0.006251220941590154, 0.006055870287165462, 0.006641922250439539,
                        0.005287896592244419],
    "plan_boundary_2": [0.028526768268855023, 0.028716546200429773, 0.028130494237155693,
                        0.029107247509279156, 0.02773979292830631, 0.028325844891580385,
                        0.023501762632197415],
    "plan_boundary_3": [0.06662758890191481, 0.06368431334244971, 0.06563781988669662,
                        0.06700527446766946, 0.06466106661457316, 0.0687634303574917,
                        0.0526831179005092],
    # l2 (keep as meters)
    "l2_1": [0.14073001438774535, 0.1405160571028371, 0.14004844205291944,
             0.1411717463533343, 0.13997492848604917, 0.14041200610094653,
             0.1384626570314747],
    "l2_2": [0.29276992533497037, 0.2901175294479828, 0.28982497465653123,
             0.2935891056388059, 0.2904003775707876, 0.29162442013803747,
             0.29268974198782055],
    "l2_3": [0.5507155500945252, 0.5432729074307647, 0.5444164966730612,
             0.5533946216869686, 0.546948404469905, 0.5484402532670687,
             0.5625959994569897],
}
df = pd.DataFrame(data)

# -----------------------------
# 2. Convert units & averages
# -----------------------------
# Convert COL and boundary columns to %
# col_cols = ["plan_obj_box_col_1", "plan_obj_box_col_2", "plan_obj_box_col_3"]
# bnd_cols = ["plan_boundary_1", "plan_boundary_2", "plan_boundary_3"]
# df[col_cols + bnd_cols] = df[col_cols + bnd_cols] * 100.0

# # Averages
# df["plan_obj_box_col_avg"] = df[col_cols].mean(axis=1)
# df["plan_boundary_avg"]    = df[bnd_cols].mean(axis=1)
# df["l2_avg"]               = df[["l2_1", "l2_2", "l2_3"]].mean(axis=1)

# # -----------------------------
# # 3. Plot helper
# # -----------------------------
# def plot_group(df, x, cols, avg_col, title, ylabel, filename):
#     plt.figure()
#     for c in cols:
#         plt.plot(df[x], df[c], marker="o", label=c)
#     plt.plot(df[x], df[avg_col], marker="o", linestyle="--", label=avg_col)
#     plt.xlabel("scale_factor")
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300)
#     plt.close()

# # -----------------------------
# # 4. Generate & save 3 figures
# # -----------------------------
# plot_group(
#     df, "scale_factor",
#     col_cols, "plan_obj_box_col_avg",
#     "Plan Obj Box Col vs. Scale Factor",
#     "Value (%)",
#     "plan_obj_box_col.png"
# )

# plot_group(
#     df, "scale_factor",
#     bnd_cols, "plan_boundary_avg",
#     "Plan Boundary vs. Scale Factor",
#     "Value (%)",
#     "plan_boundary.png"
# )

# plot_group(
#     df, "scale_factor",
#     ["l2_1", "l2_2", "l2_3"], "l2_avg",
#     "L2 Error vs. Scale Factor",
#     "Error (m)",
#     "l2_error.png"
# )

# print("Saved: plan_obj_box_col.png, plan_boundary.png, l2_error.png")


# 初始化结果列表
results = []

# 逐组处理（每个 scale_factor 对应一次）
for i in range(len(data["scale_factor"])):
    obj_col_mean = np.mean([
        data["plan_obj_box_col_1"][i],
        data["plan_obj_box_col_2"][i],
        data["plan_obj_box_col_3"][i],
    ])
    
    boundary_mean = np.mean([
        data["plan_boundary_1"][i],
        data["plan_boundary_2"][i],
        data["plan_boundary_3"][i],
    ])
    
    l2_mean = np.mean([
        data["l2_1"][i],
        data["l2_2"][i],
        data["l2_3"][i],
    ])
    
    results.append({
        "scale_factor": data["scale_factor"][i],
        "plan_obj_box_col": obj_col_mean * 100,     # 转换为百分比
        "plan_boundary": boundary_mean * 100,       # 转换为百分比
        "l2": l2_mean,                              # 保持为 meter
    })

# 打印结果
for res in results:
    print("scale_factor = {:.1f} | box_col = {:.4f}% | boundary = {:.4f}% | l2 = {:.4f}m".format(
        res["scale_factor"],
        res["plan_obj_box_col"],
        res["plan_boundary"],
        res["l2"]
    ))
