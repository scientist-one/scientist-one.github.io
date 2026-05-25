# Idea Report

## Synergistic Bipartite Alignment via Synthetic Markdown Scaffolding

**Score**: 7.75

**Abstract**:
The Google AI4Code competition challenges participants to reconstruct the original order of markdown cells within a Python notebook, given the correct order of code cells. Existing baselines, such as CodeT5-RNN (C0) and C2LLM (C1), typically approach this via pairwise cross-modal scoring (evaluating code against markdown). This paradigm is fundamentally flawed for the Kendall tau metric: independent pairwise predictions routinely produce transitive inconsistencies (e.g., predicting A > B, B > C, but C > A), severely inflating the number of sorting swaps and penalizing global ranking performance.

We propose Synergistic Bipartite Alignment, a novel two-stage architecture that reformulates the cross-modal listwise ranking task into a uni-modal assignment problem. In the first stage, we leverage an LLM to generate "synthetic markdown scaffolding" at every valid code cell boundary. These synthetically generated text blocks act as idealized natural language anchors that describe the expected transitions in the notebook's execution state. In the second stage, we project both the real shuffled markdown cells and the synthetic anchors into a shared semantic embedding space. 

By executing a global bipartite matching algorithm (such as Optimal Transport or the Hungarian algorithm) between the real and synthetic embedding sets, we mathematically guarantee a globally consistent, cycle-free assignment. This approach entirely eliminates the structural paradoxes of pairwise ranking and bounds the worst-case Kendall tau swap penalties. We expect this synergistic generate-then-match pipeline to substantially outperform both the C0 and C1 baselines by replacing unstable cross-modal heuristics with structurally guaranteed text-to-text alignment.


**Related Work**:
Current literature heavily relies on pairwise classification or generative sequence modeling for code-language alignment, which are notably vulnerable to transitive inconsistencies. For instance, Huang et al. [5] utilize dual-encoder context fusion for notebooks, but primarily focus on data-wrangling code generation rather than structural listwise ranking. Similarly, Rahman et al. [4] established a strong foundation with the CodeT5-RNN hybrid framework for code comprehension, yet their method treats notebooks as flat sequences, leading to suboptimal global structural recovery when predicting cell order.

To handle longer contexts and avoid truncation, Qin et al. [3] introduced Pooling by Multihead Attention (PMA) in the C2LLM framework. Concurrently, Lu et al. [1] proposed Ranking-Tree Loss to address listwise optimization; however, directly integrating such listwise losses into cross-modal matching networks remains challenging due to the disparate nature of code and text. 

Inspired by Trofimova et al. [19], who demonstrated a two-step synergistic framework for transformative code generation, we adapt this two-step paradigm to first generate high-level structural instructions (synthetic markdown) before conducting the ranking. We further draw upon Bhattacharya and Gupta [21], utilizing their syntax-aware entity extraction to anchor the synthetic generation. By combining these approaches, our method transforms the error-prone cross-modal ranking task into a highly robust, uni-modal assignment problem.


**Hypothesis**:
Generating synthetic markdown for each valid code cell boundary, followed by formulating the final sequence ordering as a global bipartite matching problem between these synthetic placeholders and the shuffled real markdown, strictly enforces cycle-free assignments and directly mitigates the Kendall tau penalties associated with transitive sequence inconsistencies.

**Proposed Experiments**:
- **Exp**: Using synthetic markdown scaffolding combined with global bipartite matching yields a higher Kendall tau score than direct cross-modal pairwise ranking approaches like CodeT5-RNN (C0) and C2LLM (C1).
  - Step 1: Process the AI4Code training dataset to generate synthetic markdown placeholders at each code cell boundary using a lightweight LLM.
  - Step 2: Train a uni-modal text embedding model (e.g., using contrastive loss) to map both the ground-truth markdown cells and the synthetic markdown placeholders into a shared dense vector space.
  - Step 3: During inference on the test set, compute a distance matrix between real and synthetic embeddings, apply the Hungarian matching algorithm to assign real markdowns to structural boundaries, and compute the resulting Kendall tau against the C0 and C1 baselines.
- **Exp**: The performance gains of Synergistic Bipartite Alignment are predominantly driven by the cycle-free global assignment of the bipartite matching algorithm, rather than merely the descriptive power of the synthetic text embeddings.
  - Step 1: Freeze the trained embeddings from the synthetic markdown scaffolding stage to ensure the semantic representation remains identical.
  - Step 2: Replace the global bipartite matching algorithm with a standard pairwise classification head that independently predicts the probability of a real markdown belonging to a specific synthetic boundary.
  - Step 3: Resolve the independent pairwise probabilities into a final sorted list (allowing potential transitive cycles) and measure the degradation in Kendall tau compared to the strict bipartite matching baseline.

**Risk Factors and Limitations**:
1. N-to-M Mapping Imbalance: Multiple markdown cells might logically belong to the exact same code cell boundary, violating the strict 1-to-1 assumption of basic bipartite matching. 
   Diagnostic: High frequency of adjacent, uninterrupted markdown cells in the training set orders. 
   Fallback: Relax the Hungarian assignment to an Optimal Transport (OT) formulation with learned capacity constraints for each boundary, allowing stable N-to-1 assignments.

2. Synthetic Generation Hallucinations: The generation step could produce generic or uninformative text, resulting in overlapping or indistinguishable anchor embeddings.
   Diagnostic: Low variance in the generated synthetic markdown embeddings and minimal Euclidean distance between adjacent boundary anchors. 
   Fallback: Restrict the LLM generation explicitly to syntax-aware entity extraction (Bhattacharya and Gupta [21]) to force the inclusion of distinct variable names, API calls, and AST structures in the synthetic scaffolding.
   
3. Scalability on Extreme Contexts: Large notebooks with hundreds of cells may make the global distance matrix computation and algorithmic matching computationally prohibitive within the 9-hour Kaggle runtime limit.
   Diagnostic: Out-of-memory (OOM) errors or inference timeouts during validation on notebooks containing >200 cells. 
   Fallback: Implement hierarchical clustering (Zhu et al. [22]) to pre-group related subsets of adjacent code and markdown cells into macro-blocks before running the matching algorithm locally within those blocks.

## Critic justifications

- **novelty**: Translating the cross-modal ranking task into a uni-modal bipartite matching problem via synthetic scaffolding is a highly original approach to definitively circumvent transitive inconsistencies.
- **feasibility**: Using an LLM to summarize code cells into text and applying standard bipartite matching algorithms (like the Hungarian algorithm) over text embeddings is computationally straightforward.
- **excitement**: The ML and NLP communities would appreciate the elegance of bypassing complex cross-modal listwise ranking with a robust generate-then-match pipeline that mathematically guarantees cycle-free assignments.
- **impact**: By strictly enforcing global consistency through 1-to-1 bipartite matching, this method directly neutralizes the Kendall tau penalties associated with transitive cycles and misorderings.

## Candidate ranking

| Rank | Mode | Title | Overall | Nov | Feas | Exc | Imp | Viable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | unconventional | Synergistic Bipartite Alignment via Synthetic Markdown Scaffolding | 7.75 | 8 | 8 | 7 | 8 | True |
| 2 | unconventional | Narrative-to-Transition Mapping using Dual-Encoder Fusion | 6.95 | 7 | 6 | 8 | 7 | True |
| 3 | unconventional | Skeleton Alignment via One-Pass Code Context Sampling | 6.75 | 6 | 8 | 6 | 7 | True |
| 4 | unconventional | Dynamic Notebook Slicing via Dependency-Aware Context Pruning | 6.15 | 5 | 7 | 6 | 7 | True |
| 5 | unconventional | Block-Level Narrative Sequencing via Leiden Hierarchical Clustering | 5.9 | 6 | 7 | 5 | 5 | True |
| 6 | unconventional | Iterative Sequence Refinement via FeatureSHAP Attention Reweighting | 5.0 | 6 | 4 | 5 | 5 | True |
| 7 | unconventional | State-Delta Compression via Collapsed History Management | 4.95 | 5 | 4 | 6 | 5 | True |
