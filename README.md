
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

<a href="https://www.andrew.cmu.edu/course/11-667/lectures/W10L2%20Scaling%20Up%20Parallel%20Training.pdf">Parallel training</a>  by Chenyan Xiong: Overview of optimization and parallelism techniques.

<a href="https://arxiv.org/abs/2407.20018">Distributed training</a>  by Duan et al.: A survey about efficient training of LLM on distributed architectures.

<a href="https://allenai.org/olmo">OLMo 2</a> by AI2: Open-source language model with model, data, training, and evaluation code.

<a href="https://www.llm360.ai/">LLM360</a> by LLM360: A framework for open-source LLMs with training and data preparation code, data, metrics, and models.


### 3. Post-training datasets:


Post-training datasets have a precise structure with instructions and answers (supervised fine-tuning) or instructions and chosen/rejected answers (preference alignment). Conversational structures are a lot rarer than the raw text used for pre-training, which is why we often need to process seed data and refine it to improve the accuracy, diversity, and complexity of the samples. More information and examples are available in my repo <a href="https://github.com/mlabonne/llm-datasets">💾 LLM Datasets.</a>



* Storage & chat templates: Because of the conversational structure, post-training datasets are stored in a specific format like ShareGPT or OpenAI/HF. Then, these formats are mapped to a chat template like ChatML or Alpaca to produce the final samples the model is trained on.

* Synthetic data generation: Create instruction-response pairs based on seed data using frontier models like GPT-4o. This approach allows for flexible and scalable dataset creation with high-quality answers. Key considerations include designing diverse seed tasks and effective system prompts.

* Data enhancement: Enhance existing samples using techniques like verified outputs (using unit tests or solvers), multiple answers with rejection sampling, Auto-Evol, Chain-of-Thought, Branch-Solve-Merge, personas, etc.

* Quality filtering: Traditional techniques involve rule-based filtering, removing duplicates or near-duplicates (with MinHash or embeddings), and n-gram decontamination. Reward models and judge LLMs complement this step with fine-grained and customizable quality control.


### 📚 References:

<a href="https://huggingface.co/spaces/argilla/synthetic-data-generator">Synthetic Data Generator</a> by Argilla: Beginner-friendly way of building datasets using natural language in a Hugging Face space.

<a href="https://github.com/mlabonne/llm-datasets">LLM Datasets</a>  by Maxime Labonne: Curated list of datasets and tools for post-training.

<a href="https://github.com/NVIDIA/NeMo-Curator">NeMo-Curator</a>  by Nvidia: Dataset preparation and curation framework for pre and post-training data.

<a href="https://distilabel.argilla.io/dev/sections/pipeline_samples/">Distilabel</a>  by Argilla: Framework to generate synthetic data. It also includes interesting reproductions of papers like UltraFeedback.

<a href="https://github.com/MinishLab/semhash">Semhash</a> by MinishLab: Minimalistic library for near-deduplication and decontamination with a distilled embedding model.

<a href="https://huggingface.co/docs/transformers/main/en/chat_templating">Chat Template</a>  by Hugging Face: Hugging Face’s documentation about chat templates.


### 4. Supervised Fine-Tuning

SFT turns base models into helpful assistants, capable of answering questions and following instructions. During this process, they learn how to structure answers and reactivate a subset of knowledge learned during pre-training. Instilling new knowledge is possible but superficial: it cannot be used to learn a completely new language. Always prioritize data quality over parameter optimization.

* Training techniques: Full fine-tuning updates all model parameters but requires significant compute. Parameter-efficient fine-tuning techniques like LoRA and QLoRA reduce memory requirements by training a small number of adapter parameters while keeping base weights frozen. QLoRA combines 4-bit quantization with LoRA to reduce VRAM usage.

* Training parameters: Key parameters include learning rate with schedulers, batch size, gradient accumulation, number of epochs, optimizer (like 8-bit AdamW), weight decay for regularization, and warmup steps for training stability. LoRA also adds three parameters: rank (typically 16–128), alpha (1–2x rank), and target modules.

* Distributed training: Scale training across multiple GPUs using DeepSpeed or FSDP. DeepSpeed provides three ZeRO optimization stages with increasing levels of memory efficiency through state partitioning. Both methods support gradient checkpointing for memory efficiency.

* Monitoring: Track training metrics including loss curves, learning rate schedules, and gradient norms. Monitor for common issues like loss spikes, gradient explosions, or performance degradation.


### 📚 References:




<img src="./image/2.png"> 




