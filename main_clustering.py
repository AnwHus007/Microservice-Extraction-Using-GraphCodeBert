import os
import re
import random
from collections import defaultdict
from tqdm import trange

import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN

CONFIG = {
    "project_path": "./SpringBlog", # change to your project path
    "device": "cuda",

    "k_min_clusters": 2,
    "k_max_clusters": 25, 

    "eps_min": 0.5,
    "eps_max": 0.9,
    "eps_steps": 20, 

    "similarity_mode": "hybrid",
    "alpha_j": 0.125,     
    "alpha_c": 25.0,   
    "w1": 0.5,         
    "w2": 0.5,          
    
    "singleton_penalty_weight": 0.5, 

    "seed": 42,
    "save_prefix": "clustering_out_SB",
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])


def get_class_body(code, class_name):
    pattern = re.compile(r"(^|\s)class\s+" + re.escape(class_name) + r"\b[^{]*\{")
    match = pattern.search(code)
    if not match: return ""
    start_index = match.end() - 1
    balance = 0
    body_parts = []
    started = False
    for i in range(start_index, len(code)):
        char = code[i]
        if char == '{':
            balance += 1
            started = True
        elif char == '}':
            balance -= 1
        body_parts.append(char)
        if started and balance == 0:
            break
    return "".join(body_parts)

def extract_calls_java(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
    except:
        return {}
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    class_defs = re.findall(r"\bclass\s+([A-Za-z_]\w*)", code)
    calls = defaultdict(set)
    for cls in class_defs:
        body = get_class_body(code, cls)
        if not body: continue
        instantiations = re.findall(r"\bnew\s+([A-Z][A-Za-z0-9_]*)\s*\(", body)
        static_calls = re.findall(r"\b([A-Z][A-Za-z0-9_]*)\s*\.", body)
        declarations = re.findall(r"\b([A-Z][A-Za-z0-9_]*)\s+[a-z_][a-z0-9_]*", body)
        found = set(instantiations + static_calls + declarations)
        primitives = {"String", "Integer", "Boolean", "Double", "Float", "Long", "List", "Map", "Set", "Object", "System"}
        for item in found:
            if item != cls and item not in primitives:
                calls[cls].add(item)
    return calls

def extract_project_calls_java(folder):
    project_calls = defaultdict(set)
    files_processed = 0
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".java"):
                path = os.path.join(root, f)
                fcalls = extract_calls_java(path)
                for cls, s in fcalls.items():
                    project_calls[cls].update(s)
                files_processed += 1
    print(f"Parsed {files_processed} Java files.")
    return project_calls

def get_code_embedding(model, tokenizer, code, device="cpu"):
    try:
        snippet = code[:10000]
        tokens = tokenizer(snippet, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**tokens)
            vec = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return vec
    except Exception as e:
        return np.zeros(768)

def extract_embeddings_java(folder, device="cpu"):
    print("Loading GraphCodeBERT...")
    model_name = "microsoft/graphcodebert-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name).to(device)
    model.eval()
    embeddings = {}
    files_seen = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".java"):
                cls = os.path.splitext(f)[0]
                files_seen.append(cls)
                try:
                    path = os.path.join(root, f)
                    code = open(path, "r", encoding="utf-8", errors="ignore").read()
                except:
                    code = ""
                emb = get_code_embedding(model, tokenizer, code, device)
                embeddings[cls] = emb
    return files_seen, embeddings


def compute_jaccard_matrix(class_calls, classes):
    n = len(classes)
    M = np.zeros((n, n), dtype=float)
    idx_map = {name: i for i, name in enumerate(classes)}
    for a_name, calls in class_calls.items():
        if a_name not in idx_map: continue
        i = idx_map[a_name]
        valid_calls_a = {idx_map[c] for c in calls if c in idx_map}
        for j in range(i + 1, n):
            b_name = classes[j]
            calls_b = class_calls.get(b_name, set())
            valid_calls_b = {idx_map[c] for c in calls_b if c in idx_map}
            intersection = len(valid_calls_a & valid_calls_b)
            union = len(valid_calls_a | valid_calls_b)
            if union > 0:
                score = intersection / union
                M[i, j] = score
                M[j, i] = score
    return M

def compute_cosine_matrix(classes, embeddings):
    sample_vec = next(iter(embeddings.values())) if embeddings else np.zeros(768)
    vecs = []
    for c in classes:
        if c in embeddings:
            vecs.append(embeddings[c])
        else:
            vecs.append(np.zeros_like(sample_vec))
    V = np.vstack(vecs)
    norm = np.linalg.norm(V, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    Vn = V / norm
    return Vn @ Vn.T

def compose_similarity(jmat, cmat, mode, alpha_j, alpha_c, w1, w2):
    # Copy and normalize (Pre-alpha normalization)
    J = jmat.copy()
    C = cmat.copy()
    if J.max() > 1e-9: J = J / J.max()
    if C.max() > 1e-9: C = C / C.max()
    
    # Apply exponential scaling (Contrast enhancement)
    J_scaled = np.power(J, alpha_j)
    C_scaled = np.clip(C, 0, 1)
    C_scaled = np.power(C_scaled, alpha_c)
    
    # Store the scaled matrices for external analysis before mixing
    J_post_alpha = J_scaled.copy()
    C_post_alpha = C_scaled.copy()
    
    if mode == "jaccard":
        S = J_scaled
    elif mode == "cosine":
        S = C_scaled
    else:
        S = w1 * J_scaled + w2 * C_scaled
    
    # Final normalization
    if S.max() > 1e-9:
        S = S / S.max()
        
    # Return the final matrix AND the scaled components for analysis
    return S, J_post_alpha, C_post_alpha



def canonicalize_partition(labels):
    mapping = {}
    new = []
    next_id = 0
    for lbl in labels:
        if lbl not in mapping:
            mapping[lbl] = next_id
            next_id += 1
        new.append(mapping[lbl])
    return new


def compute_paper_diagnostics_and_MQ(sim, labels):

    n = sim.shape[0]
    labels = np.array(labels) 
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    interface_indices = set()
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if labels[i] != labels[j] and sim[i, j] > 1e-9:
                interface_indices.add(i)
                break 
 
    total_internal_sim = 0.0
    total_internal_pairs = 0.0
    clusters = defaultdict(list)
    for idx, lbl in enumerate(labels):
        clusters[lbl].append(idx)

    for lbl in unique_labels:
        indices = clusters[lbl]
        k = len(indices)
        if k > 1:
            sub = sim[np.ix_(indices, indices)]
            internal_sum = np.triu(sub, 1).sum()
            internal_pairs = k * (k - 1) / 2.0
            total_internal_sim += internal_sum
            total_internal_pairs += internal_pairs
    
    cohesion = total_internal_sim / total_internal_pairs if total_internal_pairs > 0 else 0.0


    total_coupling_sum = 0.0
    cluster_pair_count = 0.0
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            lbl_a = unique_labels[i]
            lbl_b = unique_labels[j]
            indices_a = clusters[lbl_a]
            indices_b = clusters[lbl_b]

            interface_a = [idx for idx in indices_a if idx in interface_indices]
            interface_b = [idx for idx in indices_b if idx in interface_indices]

            len_ia = len(interface_a)
            len_ib = len(interface_b)

            if len_ia == 0 or len_ib == 0:
                avg_coupling_ab = 0.0
            else:
                coupling_sum_ab = sim[np.ix_(interface_a, interface_b)].sum()
                num_pairs_ab = len_ia * len_ib
                avg_coupling_ab = coupling_sum_ab / num_pairs_ab
            
            total_coupling_sum += avg_coupling_ab
            cluster_pair_count += 1
    
    coupling = total_coupling_sum / cluster_pair_count if cluster_pair_count > 0 else 0.0

    # 4. Calculate Granularity
    granularity = n / num_clusters if num_clusters > 0 else float('inf')
    
    # 5. Calculate Modularity Quality (MQ)
    mq = cohesion - coupling
    
    return cohesion, coupling, granularity, mq


def compute_modularity_Q_diagnostic(sim, labels):

    n = sim.shape[0]
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    
    clusters = defaultdict(list)
    for idx, lbl in enumerate(labels):
        clusters[lbl].append(idx)
        
    m = sim.sum() / 2.0 + 1e-12
    k_vals = sim.sum(axis=1)
    Q = 0.0
    
    for lbl in unique_labels:
        indices = clusters[lbl]
        k_sub = k_vals[indices]
        if k_sub.size > 0:
            expected = np.outer(k_sub, k_sub) / (2 * m)
            sub_sim = sim[np.ix_(indices, indices)]
            Q += (sub_sim - expected).sum()
        
    Q = Q / (2 * m)
    return Q



def analyze_matrix(matrix, name):
    """Prints statistics about a similarity matrix."""
    print(f"\n--- Matrix Analysis: {name} ---")
    
    if matrix is None or matrix.size == 0:
        print("  Matrix is empty.")
        return

    n = matrix.shape[0]
    
    mat_copy = matrix.copy()
    np.fill_diagonal(mat_copy, np.nan)
    
    try:
        mat_copy[~np.isfinite(mat_copy)] = np.nan
        
        min_val = np.nanmin(mat_copy)
        max_val = np.nanmax(mat_copy)
        mean_val = np.nanmean(mat_copy)
        median_val = np.nanmedian(mat_copy)
        
        off_diagonal_elements = mat_copy[~np.isnan(mat_copy)]
        total_off_diagonal = len(off_diagonal_elements)
        
        if total_off_diagonal > 0:
            sparse_count = np.sum(off_diagonal_elements < 0.01)
            sparsity = (sparse_count / total_off_diagonal) * 100
        else:
            sparsity = 100.0 

        print(f"  Shape: {n}x{n}")
        print(f"    Min:    {min_val:.4f}")
        print(f"    Max:    {max_val:.4f}")
        print(f"    Mean:   {mean_val:.4f}")
        print(f"    Median: {median_val:.4f}")

    except Exception as e:
        print(f"  Could not compute stats (matrix might be all-NaN?): {e}")
    print("---------------------------------")


def compute_structural_metrics(sim, labels):

    n = sim.shape[0]
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    
    total_internal_sim = 0.0
    total_sim_sum = np.triu(sim, 1).sum() 

    for lbl in unique_labels:
        indices = np.where(labels == lbl)[0]
        if len(indices) > 1:
            sub = sim[np.ix_(indices, indices)]
            total_internal_sim += np.triu(sub, 1).sum()
            
    total_external_sim = total_sim_sum - total_internal_sim
    
    return total_internal_sim, total_external_sim



def calculate_penalized_mq(mq_raw, labels, penalty_weight):

    unique_labels, counts = np.unique(labels, return_counts=True)
    num_clusters = len(unique_labels)
    num_singletons = np.sum(counts == 1)
    
    if num_clusters > 0:
        # Penalty: weight * percentage of clusters that are singletons
        penalty = penalty_weight * (num_singletons / num_clusters)
    else:
        penalty = 0.0
        
    return mq_raw - penalty, penalty

def run_spectral_iteration(sim, cfg, n_classes):
    print(f"\n--- Running Spectral Clustering ---")
    best_sol = None
    best_mq_penalized = -float('inf')
    best_k = 0
    # New storage for specific metrics
    best_coh = 0.0
    best_coup = 0.0
    
    min_k = cfg["k_min_clusters"]
    max_k = min(cfg["k_max_clusters"], n_classes - 1) 
    
    print(f"  Testing n_clusters (k) from {min_k} to {max_k}...")
    iterator = trange(min_k, max_k + 1, desc="Spectral")
    
    for k in iterator:
        try:
            model = SpectralClustering(n_clusters=k, 
                                       affinity='precomputed', 
                                       random_state=cfg["seed"],
                                       assign_labels='kmeans')
            labels = model.fit_predict(sim)
            
            coh, coup, gran, mq_raw = compute_paper_diagnostics_and_MQ(sim, labels)
            mq_penalized, penalty = calculate_penalized_mq(mq_raw, labels, cfg["singleton_penalty_weight"])
            
            iterator.set_postfix(k=k, MQ_raw=f"{mq_raw:.3f}", MQ_pen=f"{mq_penalized:.3f}")
            
            if mq_penalized > best_mq_penalized:
                best_mq_penalized = mq_penalized
                best_sol = canonicalize_partition(labels)
                best_k = k
            
                best_coh = coh
                best_coup = coup
                
        except Exception as e:
            continue
            
    return best_mq_penalized, best_sol, best_k, best_coh, best_coup

def run_agglomerative_iteration(sim, cfg, n_classes):
    print(f"\n--- Running Agglomerative Clustering ---")
    best_sol = None
    best_mq_penalized = -float('inf')
    best_k = 0
    best_coh = 0.0
    best_coup = 0.0
    
    min_k = cfg["k_min_clusters"]
    max_k = min(cfg["k_max_clusters"], n_classes - 1) 
    distance_matrix = 1 - sim
    
    print(f"  Testing n_clusters (k) from {min_k} to {max_k}...")
    iterator = trange(min_k, max_k + 1, desc="Agglomerative")
    
    for k in iterator:
        try:
            model = AgglomerativeClustering(n_clusters=k, 
                                            metric='precomputed', 
                                            linkage='average')
            labels = model.fit_predict(distance_matrix)

            coh, coup, gran, mq_raw = compute_paper_diagnostics_and_MQ(sim, labels)
            mq_penalized, penalty = calculate_penalized_mq(mq_raw, labels, cfg["singleton_penalty_weight"])

            iterator.set_postfix(k=k, MQ_raw=f"{mq_raw:.3f}", MQ_pen=f"{mq_penalized:.3f}")
            
            if mq_penalized > best_mq_penalized:
                best_mq_penalized = mq_penalized
                best_sol = canonicalize_partition(labels)
                best_k = k
                best_coh = coh
                best_coup = coup
                
        except Exception as e:
            continue
            
    return best_mq_penalized, best_sol, best_k, best_coh, best_coup

def run_dbscan_iteration(sim, cfg, n_classes):
    print(f"\n--- Running DBSCAN Clustering ---")
    best_sol = None
    best_mq_penalized = -float('inf')
    best_eps = 0.0
    best_coh = 0.0
    best_coup = 0.0
    
    distance_matrix = 1 - sim
    print(f"  Testing eps from {cfg['eps_min']} to {cfg['eps_max']}...")
    iterator = trange(cfg["eps_steps"] + 1, desc="DBSCAN")

    for i in iterator:
        k_step = i / cfg["eps_steps"]
        eps_val = cfg["eps_min"] + k_step * (cfg["eps_max"] - cfg["eps_min"])
        
        try:
            model = DBSCAN(eps=eps_val, 
                           metric='precomputed', 
                           min_samples=2) 
            
            labels = model.fit_predict(distance_matrix)
            
            if len(set(labels)) < 2 or len(set(labels)) == n_classes:
                continue
                
            coh, coup, gran, mq_raw = compute_paper_diagnostics_and_MQ(sim, labels)
            mq_penalized, penalty = calculate_penalized_mq(mq_raw, labels, cfg["singleton_penalty_weight"])
            
            iterator.set_postfix(eps=f"{eps_val:.2f}", MQ_raw=f"{mq_raw:.3f}", MQ_pen=f"{mq_penalized:.3f}")
            
            if mq_penalized > best_mq_penalized:
                best_mq_penalized = mq_penalized
                best_sol = canonicalize_partition(labels)
                best_eps = eps_val
                best_coh = coh
                best_coup = coup
                
        except Exception as e:
            continue
            
    return best_mq_penalized, best_sol, best_eps, best_coh, best_coup


def print_clusters_to_console(df):

    print("\n" + "=" * 30)
    print("      Final Cluster Assignments")
    print("=" * 30)
    
    grouped = df.groupby('cluster')
    sorted_clusters = sorted(grouped.groups.keys())
    
    for cluster_id in sorted(grouped.groups.keys()):
        classes = grouped.get_group(cluster_id)['class'].sort_values().tolist()
        print(f"Service {cluster_id}: {classes}")

            
    print("\n" + "=" * 30)

def main():
    cfg = CONFIG
    if not os.path.exists(cfg["project_path"]):
        print(f"Path {cfg['project_path']} not found.")
        return

    print("1. Extracting call graph...")
    calls = extract_project_calls_java(cfg["project_path"])
    
    print("2. Extracting embeddings...")
    emb_files, embeddings = extract_embeddings_java(cfg["project_path"], cfg["device"])
    
    classes = sorted(list(set(list(calls.keys()) + emb_files)))
    n_classes = len(classes)
    if n_classes == 0:
        print("No classes found.")
        return
    print(f"Total Unique Classes: {n_classes}")
    
    print("3. Building & Analyzing Matrices...")
    
    jmat = compute_jaccard_matrix(calls, classes)
    np.fill_diagonal(jmat, 1.0) 
    cmat = compute_cosine_matrix(classes, embeddings)
    np.fill_diagonal(cmat, 1.0) 
    
    analyze_matrix(jmat, "Raw Jaccard (Structural)")
    analyze_matrix(cmat, "Raw Cosine (Semantic)")
    
    pd.DataFrame(jmat, index=classes, columns=classes).to_csv(f"{cfg['save_prefix']}_jaccard_raw.csv")
    pd.DataFrame(cmat, index=classes, columns=classes).to_csv(f"{cfg['save_prefix']}_cosine_raw.csv")
    

    sim, jmat_scaled, cmat_scaled = compose_similarity(
        jmat, cmat, cfg["similarity_mode"], 
        cfg["alpha_j"], cfg["alpha_c"], cfg["w1"], cfg["w2"]
    )

    print("\n" + "-" * 30)
    print(f"AFTER SCALING")
    print("-" * 30)

    np.fill_diagonal(jmat_scaled, 1.0)
    np.fill_diagonal(cmat_scaled, 1.0)
    

    pd.DataFrame(jmat_scaled, index=classes, columns=classes).to_csv(f"{cfg['save_prefix']}_jaccard_scaled.csv")
    pd.DataFrame(cmat_scaled, index=classes, columns=classes).to_csv(f"{cfg['save_prefix']}_cosine_scaled.csv")

    analyze_matrix(jmat_scaled, "Jaccard Scaled")
    analyze_matrix(cmat_scaled, "Cosine Scaled")

    analyze_matrix(sim, "Final Hybrid (Tuned)")
    pd.DataFrame(sim, index=classes, columns=classes).to_csv(f"{cfg['save_prefix']}_hybrid_final.csv")
    
    
    print(f"\n4. Running All Clustering Methods...")
    
    results = {}
    
    spec_mq, spec_sol, spec_k, spec_coh, spec_coup = run_spectral_iteration(sim, cfg, n_classes)
    results["Spectral"] = {
        "mq": spec_mq, "sol": spec_sol, "param": f"k={spec_k}", 
        "coh": spec_coh, "coup": spec_coup
    }

    agglo_mq, agglo_sol, agglo_k, agglo_coh, agglo_coup = run_agglomerative_iteration(sim, cfg, n_classes)
    results["Agglomerative"] = {
        "mq": agglo_mq, "sol": agglo_sol, "param": f"k={agglo_k}", 
        "coh": agglo_coh, "coup": agglo_coup
    }
    
    db_mq, db_sol, db_eps, db_coh, db_coup = run_dbscan_iteration(sim, cfg, n_classes)
    results["DBSCAN"] = {
        "mq": db_mq, "sol": db_sol, "param": f"eps={db_eps:.4f}", 
        "coh": db_coh, "coup": db_coup
    }

    
    overall_best_mq = -float('inf')
    overall_best_sol = None
    overall_best_method = "None"
    
    print("\n" + "=" * 80)
    print(f"{'Method':<15} | {'Best Pen.MQ':<12} | {'Cohesion':<10} | {'Coupling':<10} | {'Param'}")
    print("-" * 80)
    
    for method, data in results.items():
        print(f"{method:<15} | {data['mq']:<12.4f} | {data['coh']:<10.4f} | {data['coup']:<10.4f} | {data['param']}")
        
        if data['mq'] > overall_best_mq:
            overall_best_mq = data['mq']
            overall_best_sol = data['sol']
            overall_best_method = method
            
    print("=" * 80)

    if overall_best_sol is None:
        print("\nNo valid clustering solution was found by any method.")
        return

    print(f"\n--- Overall Best Solution (by {overall_best_method}) ---")
    

    best_coh, best_coup, best_gran, best_mq_raw = compute_paper_diagnostics_and_MQ(sim, overall_best_sol)

    mq_penalized, penalty_val = calculate_penalized_mq(best_mq_raw, overall_best_sol, cfg["singleton_penalty_weight"])
    

    print("-" * 30)
    print(f"Final Clustering Results (Winner: {overall_best_method})")
    
    print(f"\n  Optimization Score (Penalized MQ): {mq_penalized:.4f}")
    print(f"    (Raw MQ: {best_mq_raw:.4f})")
    print(f"    (Singleton Penalty Applied: -{penalty_val:.4f})")
    
    print(f"    Cohesion: {best_coh:.4f}")
    print(f"    Coupling: {best_coup:.4f}")
    
    print(f"    Number of Clusters: {len(set(overall_best_sol))}")
    print("-" * 30)
    
    # Save
    df = pd.DataFrame({"class": classes, "cluster": overall_best_sol})
    df.to_csv(f"{cfg['save_prefix']}_clusters.csv", index=False)
    print(f"\nSaved best clusters to {cfg['save_prefix']}_clusters.csv")

    print_clusters_to_console(df)

if __name__ == "__main__":
    main()