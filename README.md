
# The Large Language Model Course

The Large Language Model (LLM) course is a collection of topics and educational resources for people to get into LLMs. It features two main roadmaps:

1. 🧑‍🔬 The LLM Scientist focuses on building the best possible LLMs using the latest techniques.
2. 👷 The LLM Engineer focuses on creating LLM-based applications and deploying them.


## 🧑‍🔬 The LLM Scientist

This section of the course focuses on learning how to build the best possible LLMs using the latest techniques.

<img src="./image/1.png"> 


### 1. The LLM architecture
An in-depth knowledge of the Transformer architecture is not required, but it’s important to understand the main steps of modern LLMs: converting text into numbers through tokenization, processing these tokens through layers including attention mechanisms, and finally generating new text through various sampling strategies.


* Architectural Overview: Understand the evolution from encoder-decoder Transformers to decoder-only architectures like GPT, which form the basis of modern LLMs. Focus on how these models process and generate text at a high level.

* Tokenization: Learn the principles of tokenization — how text is converted into numerical representations that LLMs can process. Explore different tokenization strategies and their impact on model performance and output quality.

* Attention mechanisms: Master the core concepts of attention mechanisms, particularly self-attention and its variants. Understand how these mechanisms enable LLMs to process long-range dependencies and maintain context throughout sequences.

* Sampling techniques: Explore various text generation approaches and their tradeoffs. Compare deterministic methods like greedy search and beam search with probabilistic approaches like temperature sampling and nucleus sampling.


### 📚 References:

* <a href="https://www.youtube.com/watch?v=wjZofJX0v4M">Visual intro to Transformers</a> by 3Blue1Brown: Visual introduction to Transformers for complete beginners.

* <a href="https://bbycroft.net/llm">LLM Visualization</a> by Brendan Bycroft: Interactive 3D visualization of LLM internals.

* <a href="https://www.youtube.com/watch?v=kCc8FmEb1nY">nanoGPT</a> by Andrej Karpathy: A 2h-long YouTube video to reimplement GPT from scratch (for programmers). He also made a video about  <a href="https://www.youtube.com/watch?v=zduSFxRajkE">tokenization</a>.

* <a href="https://lilianweng.github.io/posts/2018-06-24-attention/">Attention? Attention!</a> by Lilian Weng: Historical overview to introduce the need for attention mechanisms.

* <a href="https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html">Decoding Strategies in LLMs</a> by Maxime Labonne: Provide code and a visual introduction to the different decoding strategies to generate text.


### 2. Pre-training models

Pre-training is a computationally intensive and expensive process. While it’s not the focus of this course, it’s important to have a solid understanding of how models are pre-trained, especially in terms of data and parameters. Pre-training can also be performed by hobbyists at a small scale with <1B models.


* Data preparation: Pre-training requires massive datasets (e.g., Llama 3.1 was trained on 15 trillion tokens) that need careful curation, cleaning, deduplication, and tokenization. Modern pre-training pipelines implement sophisticated filtering to remove low-quality or problematic content.

* Distributed training: Combine different parallelization strategies: data parallel (batch distribution), pipeline parallel (layer distribution), and tensor parallel (operation splitting). These strategies require optimized network communication and memory management across GPU clusters.

* Training optimization: Use adaptive learning rates with warm-up, gradient clipping and normalization to prevent explosions, mixed-precision training for memory efficiency, and modern optimizers (AdamW, Lion) with tuned hyperparameters.

* Monitoring: Track key metrics (loss, gradients, GPU stats) using dashboards, implement targeted logging for distributed training issues, and set up performance profiling to identify bottlenecks in computation and communication across devices.


### 📚 References:

<a href="https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1">DFineWeb</a> by Penedo et al.: Article to recreate a large-scale dataset for LLM pretraining (15T), including FineWeb-Edu, a high-quality subset.

<a href="https://www.together.ai/blog/redpajama-data-v2">RedPajama v2</a>  by Weber et al.: Another article and paper about a large-scale pre-training dataset with a lot of interesting quality filters.

<a href="https://github.com/huggingface/nanotron">nanotron</a> by Hugging Face: Minimalistic LLM training codebase used to make <a href="https://github.com/huggingface/smollm">SmolLM2</a>.

Parallel training by Chenyan Xiong: Overview of optimization and parallelism techniques.

Distributed training by Duan et al.: A survey about efficient training of LLM on distributed architectures.

OLMo 2 by AI2: Open-source language model with model, data, training, and evaluation code.

LLM360 by LLM360: A framework for open-source LLMs with training and data preparation code, data, metrics, and models.












<img src="./image/2.png"> 




