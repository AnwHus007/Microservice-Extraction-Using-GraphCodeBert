# Improving Microservice Extraction Using GraphCodeBERT

**Author:** Anwar Hussain K. P.  
**Institution:** National Institute of Technology Tiruchirappalli  
**Degree:** M.Tech Data Analytics

A hybrid microservice extraction framework that decomposes monolithic Java applications by integrating **structural analysis** (Call Graphs) with **deep semantic analysis** (GraphCodeBERT). This project addresses the dimensionality gap between sparse structural dependencies and dense semantic signals using a novel non-linear scaling technique.

---

## 🚀 Key Features

* **Hybrid Extraction Framework:** Integrates static code analysis with `GraphCodeBERT` embeddings to capture data-flow and deep semantic relationships often missed by standard NLP techniques like TF-IDF.
* **Novel Non-Linear Scaling:** Implements an exponential scaling technique ($\alpha_J=0.125, \alpha_C=25$) to fuse sparse structural dependencies (Mean $\approx$ 0.008) with dense semantic signals (Mean $\approx$ 0.92), effectively resolving the "scale imbalance" problem.
* **Optimization Guardrails:**
    * **Singleton Penalty:** Prevents "orphan clusters" (services with only one class) to ensure meaningful decomposition.
    * **Granularity Target:** Guides the solution toward a practical number of services (Target: 0.5).
* **Multi-Algorithm Support:** Compares Spectral Clustering, Agglomerative Clustering, DBSCAN, and a Multi-Objective Evolutionary Algorithm (IBEA).

---

## 📂 Repository Structure

* `main_clustering.py`: Implementation of standard clustering algorithms (Spectral, Agglomerative, DBSCAN) applied to the fused similarity matrices.
* `main_ibea.py`: Implementation of the Indicator-Based Evolutionary Algorithm (IBEA) for multi-objective optimization of cohesion and coupling.
* `Example Similarity Values/`: Contains pre-computed similarity matrices for test projects (e.g., Spring Blog).
* `requirements.txt`: Python dependencies required to reproduce the analysis.

---

## 🧪 Benchmark Datasets

The model was evaluated on the following open-source Java monolithic applications:

| Project Name | GitHub Repository |
| :--- | :--- |
| **JPetStore 6** | [mybatis/jpetstore-6](https://github.com/mybatis/jpetstore-6) |
| **Spring PetClinic** | [spring-projects/spring-petclinic](https://github.com/spring-projects/spring-petclinic) |
| **SpringBoot Monolith** | [mzubal/spring-boot-monolith](https://github.com/mzubal/spring-boot-monolith) |
| **SpringBlog** | [Raysmond/SpringBlog](https://github.com/Raysmond/SpringBlog) |

---

## 📊 Experimental Results

The framework was evaluated based on **Cohesion** (higher is better), **Coupling** (lower is better), and **Net Score** (Cohesion - Coupling with added penalty).

### Complete Benchmark Results

#### 1. SpringBootMonolith
| Method | Cohesion (↑) | Coupling (↓) | Net Score (Coh - Cpl) |
| :--- | :--- | :--- | :--- |
| **Spectral** | **0.3977** | 0.1125 | **0.2852** |
| Agglomerative | 0.2674 | 0.0827 | 0.1847 |
| DBSCAN | 0.3244 | 0.1062 | 0.2182 |
| **IBEA** | 0.2690 | 0.1015 | 0.1675 |

#### 2. SpringBlog
| Method | Cohesion (↑) | Coupling (↓) | Net Score (Coh - Cpl) |
| :--- | :--- | :--- | :--- |
| **Spectral** | **0.3577** | 0.0924 | **0.2653** |
| Agglomerative | 0.1251 | **0.0160** | 0.1091 |
| DBSCAN | 0.1921 | 0.0398 | 0.1523 |
| **IBEA** | 0.2061 | 0.0854 | 0.1207 |

#### 3. Spring PetClinic
| Method | Cohesion (↑) | Coupling (↓) | Net Score (Coh - Cpl) |
| :--- | :--- | :--- | :--- |
| **Spectral** | **0.4817** | 0.1353 | **0.3464** |
| Agglomerative | 0.2627 | **0.0000** | 0.2627 |
| DBSCAN | 0.2627 | **0.0000** | 0.2627 |
| **IBEA** | 0.3235 | 0.1210 | 0.2025 |

#### 4. JPetStore
| Method | Cohesion (↑) | Coupling (↓) | Net Score (Coh - Cpl) |
| :--- | :--- | :--- | :--- |
| **Spectral** | **0.5438** | 0.2785 | **0.2653** |
| Agglomerative | 0.3853 | 0.2085 | 0.1768 |
| DBSCAN | 0.3898 | 0.2156 | 0.1742 |
| **IBEA** | 0.3983 | 0.2657 | 0.1326 |

---

## 🛠️ Usage

1.  **Installation**
    ```bash
    git clone [https://github.com/AnwHus007/Microservice-Extraction-Using-GraphCodeBert.git](https://github.com/AnwHus007/Microservice-Extraction-Using-GraphCodeBert.git)
    cd Microservice-Extraction-Using-GraphCodeBert
    pip install -r requirements.txt
    ```

2.  **Running the Analysis**
    * To run standard clustering (Spectral, Agglomerative, DBSCAN):
        ```bash
        python main_clustering.py
        ```
    * To run the Evolutionary Algorithm (IBEA):
        ```bash
        python main_ibea.py
        ```

---

## 📜 References

1.  *Sellami, K., et al.* "Improving Microservices Extraction Using Evolutionary Search," Information and Software Technology, 2022.
2.  *Guo, D., et al.* "GraphCodeBERT: Pre-training Code Representations with Data Flow," ICLR 2021.
3.  *Vera-Rivera, F.H., et al.* "Defining and Measuring Microservice Granularity," PeerJ Comput. Sci., 2021.
4.  *Saidani, I., et al.* "Towards Automated Microservices Extraction Using Multi-objective Evolutionary Search," ICSOC, 2019.
