# 1.-Experiment-
## Name:Manoj kumar D
## reg.no :212223060152
## Aim
Comprehensive Report on the Fundamentals of Generative AI and Large Language Models (LLMs)
## 1.     Explain the foundational concepts of Generative AI.
## 1. Foundation Models
Foundation models are massive, deep learning models trained on broad, generally unlabeled datasets (e.g., the entire internet). They are "foundational" because they serve as a base for various downstream tasks rather than being trained for a single, specific purpose. 
Large Language Models (LLMs): A type of foundation model designed for language-based tasks, such as GPT-4.
Adaptability: These models can be fine-tuned to handle specific tasks (like medical diagnosis) using smaller, labeled datasets.
## 2. Deep Learning and Neural Networks
Generative AI relies on deep learning—a subfield of machine learning involving multi-layered artificial neural networks. These networks, loosely inspired by the human brain, identify complex patterns in data to generate new, similar outputs. 
## 3. Key Model Architectures
Transformers: The architecture behind most modern generative tools (like ChatGPT), utilizing a "self-attention" mechanism to process entire sequences of data (like sentences) at once, capturing context and relationships efficiently.
Diffusion Models: State-of-the-art models for image generation (e.g., DALL-E, Stable Diffusion). They work by adding noise to data and then learning to reverse the process, transforming random noise into coherent, high-quality visuals.
Generative Adversarial Networks (GANs): These models consist of two neural networks—a generator and a discriminator—competing against each other to produce highly realistic data. 
## 4. Training and Fine-Tuning
Self-Supervised Learning: The initial training method where the model learns by predicting missing or future parts of the data (e.g., "fill in the blank" exercises for text).
Fine-Tuning: The process of taking a pre-trained model and training it further on a smaller, specialized dataset to adapt it for a specific task.
Reinforcement Learning from Human Feedback (RLHF): A method used to align AI behavior with human values by using human feedback to rank or score outputs, allowing the model to refine its responses for better accuracy and relevance. 
## 5. Key Technical Concepts
Prompt Engineering: The technique of crafting specific inputs to guide the AI to produce the desired, high-quality, or accurate output.
Tokens: The basic units of data (words or parts of words) that a model reads and writes.
Embeddings & Vectors: Numerical representations of content that allow the model to understand the semantic meaning and relationships between data points.
Retrieval-Augmented Generation (RAG): A framework that enhances the accuracy of AI models by allowing them to retrieve relevant information from external knowledge sources (like databases or documents) during generation, rather than relying solely on their static training data.
Latent Space: A compressed, mathematical representation of data, which allows models like VAEs to generate new variations of content by manipulating this space. 
## 2.     Focusing on Generative AI architectures. (like transformers).
From RNNs to Attention

Early models used Recurrent Neural Networks (RNNs) and LSTMs for text. But RNNs were limited by:

Slow sequential computation

Poor long-range dependency handling

To solve this, researchers introduced attention mechanisms, culminating in the Transformer architecture.

The Transformer Architecture

The Transformer, introduced by Vaswani et al. (“Attention Is All You Need”, 2017), revolutionized generative AI.
~~~
Key Components:
🔹 Input Embeddings – Converts tokens (words/subwords) into vectors
🔹 Positional Encodings – Adds sequence order information
🔹 Self-Attention – Lets tokens “attend” to each other
🔹 Multi-Head Attention – Multiple attention processes in parallel
🔹 Feed-Forward Networks – Layered nonlinear transformation
🔹 Layer Norm & Residual Connections – Stabilize deep network learning
~~~
Two Main Paths:

Encoder-Decoder: Used for translation (e.g., text-to-text).

Decoder Only: Typical in LLMs like GPT models for generation.
How Self-Attention Works

Given input tokens, the model computes queries (Q), keys (K), and values (V). Attention is calculated as:
~~~



Attention(Q,K,V) = softmax(Q·Kᵀ / √d𝑘) · V
~~~

This lets the model focus on relevant words anywhere in the sequence — not just nearby ones.
## 3.     Generative AI architecture  and its applications
Common Generative Architectures
~~~
Architecture	  Typical Use
GPT (Decoder-only Transformer)	 Text generation/chatbots
BERT (Encoder-only Transformer)	Text understanding/search
VAE (Variational Autoencoder)	Image & latent representation generation
GANs (Generative Adversarial Networks)	High-res image/video synthesis
Diffusion Models	Image, audio synthesis (e.g., Stable Diffusion)
Applications Across Domains
~~~
✅ Text

Chat assistants (e.g., ChatGPT)

Summarization

Translation

Code writing & completion

✅ Vision

Image generation (DALL-E, Midjourney)

Super-resolution

Style transfer

✅ Audio

TTS (Text-to-Speech)

Music composition

Voice conversion

✅ Multimodal

Text-to-image

Image captioning

Vision-language interaction

✅ Science & Code

Drug design

Molecular generation

Code suggestion and debugging
## 4.     Generative AI impact of scaling in LLMs.
Why Scaling Matters

As models scale, they tend to:
~~~
✔ Learn more complex patterns
✔ Generalize better across tasks
✔ Perform “zero-shot” and “few-shot” learning
✔ Reduce reliance on labeled data
~~~
<p style="text-align:center;"><i>Performance often improves smoothly with scale</i></p>
Scaling Dimensions

Parameters: More weights increase representational capacity

Training Data: More diverse text/image sources

Compute: Larger training runs over longer periods

Effects on Capabilities

Smaller models: Good at simple tasks

Larger models: Better reasoning, language understanding, creativity

Very large models (>100B parameters): exhibit emergent abilities

Costs and Challenges
<img width="1100" height="360" alt="image" src="https://github.com/user-attachments/assets/32cc2bec-0a40-4677-aa13-782bda2405ee" />
# Key Impacts of Scaling LLMs:
# Performance & Efficiency: As models grow, they generally become more accurate, though returns diminish at extreme scales, requiring optimal, balanced scaling of data, compute, and parameters.
# Emergent Abilities: Larger models gain new capabilities (e.g., complex reasoning, coding) not seen in smaller models.
# Model Parameters: Increased size improves reasoning and knowledge capacity.
# Dataset Size: More training data enhances generalization.
# Compute: Greater compute power allows for larger models and more training, improving overall quality.
# Challenges: Increased scaling results in higher environmental, financial costs and potential training instabilities.
# Inference Scaling: Investing more resources during inference (e.g., chain-of-thought) can sometimes yield better results than solely increasing model size, especially for specialized tasks. 

## 5.     Explain about LLM and how it is build.


What Is an LLM?

A Large Language Model (LLM) is a neural network trained to understand and generate natural language. It predicts the next token based on prior context.

Training Pipeline (Step-by-Step)

Data Collection

Billions of tokens from books, articles, code, web text

Tokenization

Break text into smaller units (subwords/BPE/WordPiece)

Model Architecture

Mostly decoder-only Transformer stacks

Optimization

Use gradient descent / Adam optimizer

Minimize cross-entropy loss

Evaluation

Benchmarks, human ratings, safety evaluations

Core Building Blocks
Embeddings

Text is transformed into dense vectors that the model can process.

Transformer Blocks

Every block contains:

Self-Attention

Feed-Forward Network

Layer Normalization

Residual Paths

Stacked repeatedly — sometimes hundreds of layers.

Tokenization

Tokens can be:

Words

Subwords (common in GPT/BERT)

Characters

Subword tokenization (like BPE) balances vocabulary size and flexibility.

Training Signals

LLMs typically optimize predicting the next token (autoregressive):
~~~
Loss = −∑ log P(token t|previous tokens)
~~~
Fine-Tuning & Instruction Tuning

After base training:
~~~
🔹 Models can be fine-tuned on labeled examples
🔹 Models can learn to follow instructions (e.g., ChatGPT via RLHF — Reinforcement Learning from Human Feedback)
~~~
Inference

At runtime:
~~~
Input prompt → tokenize → pass through model → generate tokens
~~~
Sampling strategies include:
~~~
Greedy

Beam search

Top-k / Top-p (nucleus sampling)
~~~
Summary: Key Takeaways
Generative AI
~~~
Learns & generates new content

Uses deep learning and probabilistic models
~~~
Transformers
~~~
Self-attention enables context understanding

Foundation of modern generative architectures
~~~
Applications
~~~
Text, images, audio, code, multimodal tools
~~~
Scaling
~~~
Bigger models unlock new capabilities

Also bring cost and safety trade-offs
~~~
LLMs
~~~
Built on tokenization, transformer stacks, and massive text corpora

Trained to predict text and guided via fine-tuning
~~~


