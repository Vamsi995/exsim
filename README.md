# MoE Align: Quantifying Expert Similarity Across Transformer Layers


## 🧠 Overview

Modern **Mixture-of-Experts (MoE)** transformers activate a subset of expert functions per token, enabling large-scale models with efficient inference. These expert layers are usually assumed to be depth-specific and non-interchangeable. This project explores whether experts across layers perform **functionally similar transformations**, making cross-layer **expert reuse** feasible.



## 🔍 Core Idea

We propose a method to **quantify the functional similarity** between experts in different transformer layers by introducing a **lightweight adapter**. This adapter aligns the input distribution of one expert to another, allowing us to evaluate output similarity using **Mean Squared Error (MSE)**.

If such an adapter achieves low MSE, it implies functional similarity (up to input transformation) between the two experts.



## 🧪 Problem Formulation

Let:

- $\( f_{\ell, e} : \mathbb{R}^d \rightarrow \mathbb{R}^d \)$ be the transformation implemented by expert $\( e \)$ in layer $\( \ell \)$
- $\( D_{\ell,e} \)$ be the empirical input distribution of expert $\( (\ell, e) \)$
- $\( A : \mathbb{R}^d \rightarrow \mathbb{R}^d \)$ be an adapter function

We aim to find:

$\[
f_{\ell_2, e_2}(A(x)) \approx f_{\ell_1, e_1}(x), \quad \forall x \sim D_{\ell_1, e_1}
\]$

The adapter is defined as:

$\[
A(x) = \text{LayerNorm}(Wx + b), \quad W \in \mathbb{R}^{d \times d}, b \in \mathbb{R}^d
\]$

The training objective is:

$\[
\mathcal{L}\_{\text{MSE}} = \mathbb{E}\_{x \sim D_{\ell_1,e_1}} \left[ \| f_{\ell_2,e_2}(A(x)) - f_{\ell_1,e_1}(x) \|_2^2 \right]
\]$



## ⚙️ Experimental Setup

- **Model**: [Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen) (decoder-only MoE transformer)
- **Tokenization**: QwenTokenizer
- **Hardware**: 1x NVIDIA A100 GPU
- **Framework**: HuggingFace Transformers
- **Data**: Subset of English Wikipedia
- **Precision**: float16
- **Adapter**: Linear + LayerNorm trained to minimize MSE



## 📊 Results

### 🔁 Cross-Layer Expert Alignment

![activations (1)](https://github.com/user-attachments/assets/14f0fae5-af10-4dfe-b1d0-18ea3f786f2e)



We compare outputs of Layer 1 and Layer 2 experts using learned adapters. Example MSE alignment losses:

| L1→L2 | E0    | E4    | E8    | E12   |
|-------|-------|-------|-------|-------|
| E0    | 8.34  | 7.71  | 7.81  | 7.87  |
| E4    | 0.017 | 0.010 | 0.018 | 0.014 |
| E8    | 0.017 | 0.011 | 0.016 | 0.014 |
| E12   | 0.015 | 0.009 | 0.014 | 0.011 |

📌 **Observation**: Expert 0 at Layer 1 shows consistently high MSE with all Layer 2 experts, indicating a functionally distinct behavior.



## 🧩 Additional Experiment: Layer Swapping

To test broader functional similarity, we swapped sparse-FFN blocks across layers in the `Switch-Base-8` model and evaluated on **MNLI**:

| Swapped Layers | Accuracy |
|----------------|----------|
| 1 <-> 3        | 0.74     |
| 1 <-> 5        | 0.728    |
| 1 <-> 7        | 0.654    |
| 1 <-> 9        | 0.378    |
| 1 <-> 11       | 0.406    |

📌 **Observation**: Functional similarity decays rapidly with layer distance.


<img width="649" alt="token_routing_heatmap" src="https://github.com/user-attachments/assets/c44a656f-8b43-4879-a9bd-dc9752fa99cf" />

Swapping entire MoE blocks (router + experts) between layers confirms that closer layers tend to be more functionally aligned. Deep swaps cause routing distributions to diverge significantly.



## 🔮 Future Work

- Extend adapter-based analysis across all layer pairs
- Measure perplexity degradation during expert swapping
- Leverage functional similarity for **dynamic expert routing**
- Apply results to **sparsity and compression**
- Evaluate generalization to **out-of-distribution** data



## 👥 Authors

- [Sai Vamsi Alisetti](https://github.com/Vamsi995)
- [Abhishek Kumar](https://github.com/Abhi12122000)




## 📎 Resources

- 📄 [Project Report](https://drive.google.com/file/d/1OdAHZ2vjIHe34CT3864xsk_MvYrVxhoh/view?usp=sharing)
- 📊 [Slides](https://docs.google.com/presentation/d/1JK6G4--TupMm0Y_bpSLhs93HhFvzDuwrkUw5C7wjdfM/edit?usp=sharing)




## 📚 References

- Switch Transformer: https://arxiv.org/abs/2101.03961  
- Universal Transformers: https://arxiv.org/abs/1807.03819  
- MoE-UT: https://arxiv.org/abs/2405.16039
- Expert Merging: https://arxiv.org/pdf/2310.01334  




## 💡 Conclusion

Our findings suggest that **many MoE experts are functionally redundant across layers**, and functional specialization is unevenly distributed. These insights pave the way for more **efficient and interpretable** MoE transformer architectures.






