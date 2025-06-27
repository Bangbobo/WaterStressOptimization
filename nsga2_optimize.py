# ==== nsga2_batch_optimize.py ====
import os
import torch
import numpy as np
import pandas as pd
from ndmo_red import (
    load_all_basins,
    replace_pyg_flow_with_absolute,
    generate_initial_population,
    replace_flow_and_evaluate_multitask,
    fast_non_dominated_sort,
    compute_crowding_distance,
    save_pareto_solutions_with_scores,
    plot_population_and_pareto_front_with_original
)
from GNNs.models import ThreeLayerGIN

def batch_run():
    data_path = "dataset_utils/pyg_data_multi"
    raw_data_path = "rawdata/nodes_all.xlsx"
    save_dir = "results/PF_relaxed"
    os.makedirs(save_dir, exist_ok=True)

    all_basins = load_all_basins(raw_data_path)
    basin_files = [f for f in os.listdir(data_path) if f.endswith(".pt")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim, edge_feature_dim = 5, 1
    MAX_PARETO_SOLUTIONS = 50

    model_dir = "results/best_models"
    fri_model_file = sorted([f for f in os.listdir(model_dir) if "GIN_FRI" in f], key=lambda x: float(x.split("r2_")[-1].replace(".pth", "")), reverse=True)[0]
    dri_model_file = sorted([f for f in os.listdir(model_dir) if "GIN_DRI" in f], key=lambda x: float(x.split("r2_")[-1].replace(".pth", "")), reverse=True)[0]

    fri_model = ThreeLayerGIN(input_dim, 256, 2).to(device)
    fri_model.load_state_dict(torch.load(os.path.join(model_dir, fri_model_file), map_location=device))
    fri_model.eval()

    dri_model = ThreeLayerGIN(input_dim, 256, 2).to(device)
    dri_model.load_state_dict(torch.load(os.path.join(model_dir, dri_model_file), map_location=device))
    dri_model.eval()

    summary_records = []

    for file in basin_files:
        basin_name = file.replace(".pt", "")
        print(f"\n🌍 Processing basin: {basin_name}")
        try:
            if basin_name not in all_basins:
                continue

            node_flows = all_basins[basin_name]
            data = torch.load(os.path.join(data_path, file), map_location='cpu')
            data = replace_pyg_flow_with_absolute(data, node_flows)

            original_FRI, original_DRI = replace_flow_and_evaluate_multitask(data, fri_model, dri_model)
            original_F = [np.mean(original_FRI), np.mean(original_DRI)]

            population = generate_initial_population(data, steps=3, n_population=100)
            if len(population) == 0:
                raise ValueError("No feasible solutions.")

            FRI_scores, DRI_scores, solutions = [], [], []
            for solution in population:
                data_temp = replace_pyg_flow_with_absolute(data.clone(), solution)
                FRI_pred, DRI_pred = replace_flow_and_evaluate_multitask(data_temp, fri_model, dri_model)
                FRI_scores.append(np.mean(FRI_pred))
                DRI_scores.append(np.mean(DRI_pred))
                solutions.append(solution)

            objectives = np.column_stack([FRI_scores, DRI_scores])
            fronts = fast_non_dominated_sort(objectives)
            pareto_indices = fronts[0]

            crowding_distances = compute_crowding_distance(objectives[pareto_indices])
            sorted_idx = np.argsort(-crowding_distances)
            if MAX_PARETO_SOLUTIONS:
                pareto_indices = [pareto_indices[i] for i in sorted_idx[:MAX_PARETO_SOLUTIONS]]
            else:
                pareto_indices = [pareto_indices[i] for i in sorted_idx]

            pareto_F = objectives[pareto_indices]
            pareto_solutions = [solutions[i] for i in pareto_indices]

            save_pareto_solutions_with_scores(
                np.array(pareto_solutions),
                np.array(FRI_scores)[pareto_indices],
                np.array(DRI_scores)[pareto_indices],
                save_dir,
                basin_name,
                original_flow=node_flows,
                original_fri=original_F[0],
                original_dri=original_F[1]
            )
            plot_population_and_pareto_front_with_original(objectives, pareto_F, original_F, basin_name, save_dir)

            summary_records.append({
                "Basin": basin_name,
                "OrigFRI": original_F[0],
                "OrigDRI": original_F[1],
                "ParetoBestFRI": pareto_F[:, 0].min(),
                "ParetoBestDRI": pareto_F[:, 1].min(),
                "NumPareto": len(pareto_indices)
            })

        except Exception as e:
            print(f"❌ Error in {basin_name}: {e}")

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(os.path.join(save_dir, "summary_report.csv"), index=False)
    print("✅ All basins completed. Summary saved.")

if __name__ == "__main__":
    batch_run()