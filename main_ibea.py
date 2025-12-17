import os
import re
import math
import random
import copy
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel


CONFIG = {
    "project_path": "./SpringBlog", # change to your project path
    "device": "cuda",
    "similarity_mode": "hybrid",

    "alpha_j": 0.125,       
    "alpha_c": 25.0,     
    "w1": 0.5,            
    "w2": 0.5,            
    
    "pop_size": 100,
    "offspring_size": 100,
    "generations": 60,
    "k_indicator": 0.05, 
    "mutation_rate": 0.2,
    "crossover_rate": 0.8,
    "max_initial_clusters": 5, 

    "granularity_target": 0.5, 
    "seed": 42,
    "save_prefix": "ibea_log_granularity",
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
    J = jmat.copy()
    C = cmat.copy()
    if J.max() > 1e-9: J = J / J.max()
    if C.max() > 1e-9: C = C / C.max()
    
    J_scaled = np.power(J, alpha_j)
    C_scaled = np.clip(C, 0, 1)
    C_scaled = np.power(C_scaled, alpha_c)
    
    J_post_alpha = J_scaled.copy()
    C_post_alpha = C_scaled.copy()
    
    if mode == "jaccard":
        S = J_scaled
    elif mode == "cosine":
        S = C_scaled
    else:
        S = w1 * J_scaled + w2 * C_scaled
    
    if S.max() > 1e-9:
        S = S / S.max()
        
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

def mutate_partition(labels, mutation_rate=0.1):
    n = len(labels)
    labels = list(labels)
    curr_max = max(labels) if labels else 0
    for i in range(n):
        if random.random() < mutation_rate:
            if random.random() < 0.8 and curr_max > 0: # Bias towards joining
                labels[i] = random.randint(0, curr_max)
            else:
                curr_max += 1
                labels[i] = curr_max
    return canonicalize_partition(labels)

def crossover_partition(a, b):
    n = len(a)
    child = []
    for i in range(n):
        child.append(a[i] if random.random() < 0.5 else b[i])
    return canonicalize_partition(child)


def compute_metrics_with_log_granularity(sim, labels):
    n = sim.shape[0]
    labels = np.array(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
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
    
    
    if num_clusters > 0:
        raw_g = n / num_clusters

        log_g = math.log(raw_g)
        max_log = math.log(n)
        gran_norm = log_g / max_log if max_log > 0 else 0.0
    else:
        gran_norm = 0.0
        
    return cohesion, coupling, gran_norm, num_clusters

def objectives_from_partition(sim, labels):

    coh, coup, gran_norm, _ = compute_metrics_with_log_granularity(sim, labels)

    target = CONFIG["granularity_target"]
    gran_dist = abs(gran_norm - target)
    
    return (1.0 - coh, coup, gran_dist)



def eps_indicator(a_obj, b_obj):
    return max([ai - bi for ai, bi in zip(a_obj, b_obj)])

def compute_indicator_matrix(pop_objs):
    n = len(pop_objs)
    I = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                I[i, j] = eps_indicator(pop_objs[i], pop_objs[j])
    return I

def compute_fitness(I, kappa):
    n = I.shape[0]
    F = np.zeros(n)
    for i in range(n):
        col = I[:, i]
        valid = np.delete(col, i)
        F[i] = np.sum(-np.exp(-valid / kappa))
    return F

def ibea_run(sim, n_items, cfg):
    pop_size = cfg["pop_size"]
    gens = cfg["generations"]
    kappa = cfg["k_indicator"]
    
    pop = []
    for _ in range(pop_size):
        # Start with small number of clusters to force merging logic early
        labels = [random.randint(0, cfg["max_initial_clusters"]-1) for _ in range(n_items)]
        pop.append(canonicalize_partition(labels))
    pop_objs = [objectives_from_partition(sim, p) for p in pop]
    
    iterator = range(gens)
    try:
        from tqdm import tqdm
        iterator = tqdm(range(gens), desc="IBEA Generations")
    except: pass
    
    for gen in iterator:
        offspring = []
        I = compute_indicator_matrix(pop_objs)
        F = compute_fitness(I, kappa)
        
        while len(offspring) < cfg["offspring_size"]:
            p1_idx = random.randint(0, len(pop)-1)
            p2_idx = random.randint(0, len(pop)-1)
            parent1 = pop[p1_idx] if F[p1_idx] > F[p2_idx] else pop[p2_idx]
            
            p1_idx = random.randint(0, len(pop)-1)
            p2_idx = random.randint(0, len(pop)-1)
            parent2 = pop[p1_idx] if F[p1_idx] > F[p2_idx] else pop[p2_idx]
            
            if random.random() < cfg["crossover_rate"]:
                child = crossover_partition(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)
            child = mutate_partition(child, cfg["mutation_rate"])
            offspring.append(child)
            
        off_objs = [objectives_from_partition(sim, p) for p in offspring]
        combined_pop = pop + offspring
        combined_objs = pop_objs + off_objs
        
        while len(combined_pop) > pop_size:
            I_mat = compute_indicator_matrix(combined_objs)
            fit = compute_fitness(I_mat, kappa)
            worst_idx = np.argmin(fit)
            combined_pop.pop(worst_idx)
            combined_objs.pop(worst_idx)
        pop = combined_pop
        pop_objs = combined_objs
        
    return pop, pop_objs

def analyze_matrix(matrix, name):
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
        sparsity = 100.0
        if len(off_diagonal_elements) > 0:
            sparse_count = np.sum(off_diagonal_elements < 0.01)
            sparsity = (sparse_count / len(off_diagonal_elements)) * 100
        print(f"  Shape: {n}x{n}")
        print(f"    Min:    {min_val:.4f}")
        print(f"    Max:    {max_val:.4f}")
        print(f"    Mean:   {mean_val:.4f}")
        print(f"    Median: {median_val:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
    print("---------------------------------")

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
    if not classes:
        print("No classes found.")
        return
    print(f"Total Unique Classes: {len(classes)}")
    
    print("3. Building & Analyzing Matrices...")
    jmat = compute_jaccard_matrix(calls, classes)
    np.fill_diagonal(jmat, 1.0)
    cmat = compute_cosine_matrix(classes, embeddings)
    np.fill_diagonal(cmat, 1.0)
    
    analyze_matrix(jmat, "Raw Jaccard (Structural)")
    analyze_matrix(cmat, "Raw Cosine (Semantic)")
    
    sim, jmat_scaled, cmat_scaled = compose_similarity(
        jmat, cmat, cfg["similarity_mode"], 
        cfg["alpha_j"], cfg["alpha_c"], cfg["w1"], cfg["w2"]
    )
    
    print(f"\nAFTER SCALING")
    np.fill_diagonal(jmat_scaled, 1.0)
    np.fill_diagonal(cmat_scaled, 1.0)
    analyze_matrix(jmat_scaled, "Jaccard Scaled")
    analyze_matrix(cmat_scaled, "Cosine Scaled")
    analyze_matrix(sim, "Final Hybrid")
    
    pd.DataFrame(sim, index=classes, columns=classes).to_csv(f"{cfg['save_prefix']}_hybrid_final.csv")
    
    print(f"\n4. Running IBEA (Log Granularity Target 0.5)...")
    start = time.time()
    pop, objs = ibea_run(sim, len(classes), cfg)
    print(f"Done in {time.time()-start:.2f}s")
    
    
    print(f"\n5. Selecting Best Solution...")
    # We want the solution that balances all 3, but prioritized by MQ + Granularity Compliance
    best_idx = -1
    best_score = -float('inf')
    
    for i, (neg_coh, coup, dist_gran) in enumerate(objs):
        coh = 1.0 - neg_coh
        mq = coh - coup
        score = mq - dist_gran 
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    best_sol = pop[best_idx]
    best_obj = objs[best_idx]
    
    # Recalculate metrics for display
    coh, coup, gran_norm, n_clust = compute_metrics_with_log_granularity(sim, best_sol)
    
    print("-" * 30)
    print("Final Results")
    print("-" * 30)
    
    print(f"  Log-Norm Granularity: {gran_norm:.4f} (Target: {CONFIG['granularity_target']})")
    
    print(f"\n  Core Metrics:")
    print(f"    Cohesion: {coh:.4f}")
    print(f"    Coupling: {coup:.4f}")
    
    unique, counts = np.unique(best_sol, return_counts=True)
    print(f"    Number of Clusters: {n_clust}")
    print("-" * 30)
    
    # Save
    df = pd.DataFrame({"class": classes, "cluster": best_sol})
    df.to_csv(f"{cfg['save_prefix']}_clusters.csv", index=False)
    print(f"\nSaved to {cfg['save_prefix']}_clusters.csv")
    
    print_clusters_to_console(df)

def print_clusters_to_console(df):
    print("\n" + "=" * 30)
    print("      Final Cluster Assignments")
    print("=" * 30)
    grouped = df.groupby('cluster')
    for cluster_id in sorted(grouped.groups.keys()):
        classes = grouped.get_group(cluster_id)['class'].sort_values().tolist()
        print(f"Service {cluster_id}: {classes}")

    print("\n" + "=" * 30)

if __name__ == "__main__":
    main()