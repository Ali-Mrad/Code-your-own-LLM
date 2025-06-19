# Project Overview


# Building your own LLM – ReImplementing & Fine-Tuning GPT-2  
### Master-Semester Project : 

 *“You don’t really understand something unless you can build it yourself .”*

This repository is the deliverable for my semester project at EPFL.  
I was guided by Sebastian Raschka’s book *Build a Large Language Model From Scratch*, I reproduced the GPT-2 architecture in PyTorch, experimented with genuine OpenAI weights, and fine-tuned the model for two concrete downstream tasks.

---

##  Project Purpose & Learning Goals
1. **Internalise the Transformer**  implement every component (multi-head attention, GELU, positional encoding, causal masking) *from scratch*, to gain a deep, practical understanding.
2. **Hands-on with real weights**  load the public GPT-2 checkpoints (124 M & 355 M) and inspect generation behaviour in real scenarios.
3. **Downstream adaptation**  finetune a GPT-2 for:
   * **SMS spam detection** (a binary classification)  
   * **Instruction following**   (text generation with contextual constraints)
4. **Modular engineering** migrate notebook code into a clean, well-structured Python package for maintainability and reuse.
5. **Design an advanced use case** Develop and implement a more complex, real-world application to extend and challenge the current system

The outcome is a folder **(`modular_v/`)** cointaining ready-to-run scripts and the original chapter notebooks remain for theory exploration notebooks, that document each step of the journey.

---





## Table of Contents

1. [Repository Layout](#repository-layout)
2. [Setup](#setup)
3. [Core Model Quick Demo](#core-model-quick-demo)
4. [Pipeline ① – Spam Classifier](#pipeline-①--sms-spam-classifier)
5. [Pipeline ② – Instruction Fine-Tune](#pipeline-②--instruction-sft)
6. [Chapter Notebooks](#chapter-notebooks)
7. [Results and conclusion](#results)
8. [Credits](#Credits)


---

##  Repository Layout

```
main/                                         #cointains ch02–ch07 on Jupyter notebooks format
│
├── modular_v/                 
│           ├── main_2/                       # Core GPT model components
│           │    │
│           │    ├── attention.py             # Causal multi-head attention with dropout & optional QKV bias
│           │    ├── config.py                # Configurations for different model sizes
│           │    ├── data.py                  # Prepares tokenized sequences for training and create dataloader
│           │    ├── feedforward.py           # layer FFN
│           │    ├── gelu.py	              # custom GELU activation fonction
│           │    ├── generation.py            # Generates text tokens autoregressively from context                        
│           │    ├── layernorm.py             # Applies learnable normalization over embedding dimension
│           │    ├── transformer_block.py     # Transformer block architecture
│           │    ├── weights.py               # Download and load GPT-2 weights into our model 
│           │    ├── model.py                 # GPT-2 model architecture
│           │    └── train.py                 # Basic training loop and helper functions
│           │
│           ├── data/
│           │      ├── instruction-data-with-response.json
│           │      ├── instruction-data.json
│           │      ├── sms_spam/              # data used for the classification 
│           │      └── ...
│           │
│           ├── models/
│           │       ├── gpt2-medium355M-sft.pth
│           │       ├── review_classifier.pth
│           │       └── ...                  # other models
│           │
│           ├── notebooks/                   # Jupyter notebooks theoretical build-up
│           │       ├── ch02.ipynb
│           │       ├── ch03.ipynb
│           │       ├── ch04.ipynb
│           │       ├── ch05.ipynb
│           │       ├── ch06.ipynb
│           │       ├── ch07.ipynb
│           │       └── previous_chapters.py
│           │
│           ├── scripts/
│           │       └──generate_text.py     #quick GPT-2 generation tool
│           │
│           └── task/
│               ├── spam_classification/    # pipeline GPT for spam classification
│               │   ├── data.py
│               │   ├── dataloader.py
│               │   ├── evaluation.py
│               │   ├── inference.py
│               │   ├── model_setup.py
│               │   ├── run_classification_model.py
│               │   └── training.py
│               │
│               └── instruction_finetune/   # pipeline GPT for instruction
│                       ├── data.py
│                       ├── dataloader.py
│                       ├── inference.py
│                       ├── model_setup_simple.py
│                       ├── run_instruction_model.py
│                       └── training.py
│
├── requirements.txt                        # Required Python packages
│
...

```
---

## Setup

``` bash
# clone and jump in
git clone <repo-url> && cd <repo-name>

# create a fresh env
python -m venv .venv && source .venv/bin/activate

# install all deps (CUDA PyTorch auto-detects your GPU)
pip install -r requirements.txt
```

Requirements.txt contains: torch, tiktoken, numpy, pandas, matplotlib ...
No TensorFlow required we use PyTorch checkpoints only.

---
## Core Model Quick Demo



A five-line command that proves the library works end-to-end: it downloads the official GPT-2 checkpoints, rebuilds the architecture from our own source code, moves the model to your GPU / CPU, and finally generates text with temperature & top-k sampling.
this is hiw to call the high-level script, how the weights.py utilities abstract download + loading, and how to adjust sampling hyper-parameters without touching any internals.


```bash 
python modular_v/scripts/generate_text.py \
       --size small \
       --prompt "Every effort moves you" \
       --tokens 60 \
       --temperature 0.8 \
       --top_k 5
```

```
output: 
toward finding an ideal new way to practice something! What makes us want to be on top of that?
```

---


## Pipeline ① – Spam Classifier

Classic “hello-world” for NLP classification: decide whether an SMS is spam or ham. This demonstrates that the same decoder-only GPT can be adapted to supervised tasks by attaching a small head and fine-tuning only the final Transformer block.



###  the Spam Classification Pipeline 

#### 1. Dataset Preparation (`task/spam_classification/data.py`)
- Downloads the SMS Spam Collection dataset (or reuses the local copy in `data/sms_spam/`).
- Converts the raw data into a `pandas.DataFrame` with two columns: `Label` and `Text`.
- Creates a balanced dataset with equal numbers of “spam” and “ham” messages to prevent bias.
- Splits the data into **train (70%)**, **validation (10%)**, and **test (20%)**, then saves each as a CSV.



#### 2. Dataloader (`task/spam_classification/dataloader.py`)
- Defines a `SpamDataset(torch.utils.data.Dataset)` class that:
  - Tokenizes each message using the GPT-2 BPE tokenizer (`tiktoken`).
  - Truncates or left-pads each sequence to match the maximum training length.
  - Returns `(input_ids, label)` pairs, ready for batching.
- Provides a `get_dataloaders(...)` function that returns `train`, `val`, and `test` `DataLoaders` with shuffling and `drop_last` for training stability.


#### 3. Model Construction (`task/spam_classification/model_setup.py`)
- Loads a base GPT-2 model from `modular_v.main_2` using `download_weights`.
- Freezes all layers **except**:
  - The **last Transformer block** and final **LayerNorm**.
  - A newly added **linear classification head** (2 units: spam / ham) on the last token's hidden state.
- Exposes `build_spam_model_simple()` → returns a model ready for training on the selected device.


#### 4. Training (`task/spam_classification/training.py`)
- Implements `train_classifier_simple(...)`:
  - Uses **Cross-Entropy Loss** on the logits of the final token.
  - Includes periodic loss evaluation (`eval_freq`, `eval_iter`).
  - Computes accuracy at the end of each epoch.



#### 5. Evaluation (`task/spam_classification/evaluation.py`)
- `calc_loss_loader(...)` and `calc_accuracy_loader(...)` compute average metrics over any given loader.
- These are called automatically during training and are also useful for manual evaluation or benchmarking.


#### 6. Inference (`task/spam_classification/inference.py`)
- Exposes a simple one-liner:

  ```python
  classify_review(text, model, tokenizer, ...)
  ```

  → Returns `"spam"` or `"not spam"` in a single line 


```bash 
python -m modular_v.task.spam_classification.run_classification_model \
       --load modular_v/models/review_classifier.pth \
       --epochs 2

```

You’ll see a confusion-free test accuracy plus two demo predictions printed to the console.



If we want to test or use another text to be classified we should :

```python
from main.modular_v.task.spam_classification.inference import classify_review

# here we assume that we are using `model`, `tokenizer`, and `device` that are already loaded

text = "your the winner of our lottery! send us your credit card details to claim your prize and win 3000 dollars in cash""
label = classify_review(text, model, tokenizer, device, max_length=train_dataset.max_length)
print(label) 



```
output: 
 spam
```

```python
text_2 = "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
label = classify_review(text_2, model, tokenizer, device, max_length=train_dataset.max_length)
print(label) 
```


```
output: 
 not spam
```
---

## Pipeline ② – Instruction Fine-Tune


### The Instruction Fine-Tuning Pipeline 

#### 1. Dataset Preparation (`task/instruction_finetune/data.py`)
- Downloads `instruction-data.json` if not already present .
- Defines `format_input()` to wrap each record using the InstructGPT-style prompt:

  ```
  ### Instruction:
  ...
  ### Input:
  ...
  ### Response:
  ...
  ```


#### 2. Dataloader (`task/instruction_finetune/dataloader.py`)
- `InstructionDataset` encodes each `[prompt + expected response]` as a single contiguous sequence of token IDs
- Custom `custom_collate_fn`:
  - Pads all samples in a batch to the length of the longest
  - Shifts targets by +1 for next-token prediction
  - Masks all non-response tokens using `ignore_index` — loss is only computed on the response
- `create_dataloaders(...)` returns `train`, `val`, and `test` loaders with dynamic batch-level padding


#### 3. Model Setup (`task/instruction_finetune/model_setup_simple.py`)
- Initializes from the GPT-2 “medium” (355M parameters) backbone
- Optionally loads a checkpoint with `load_into(...)`
- Retains the standard language modeling head (predicting next-token probabilities)
- Exposes `build_sft_model(ckpt_path, model_size="medium", device=...)` for training-ready setup



#### 4. Training (`task/instruction_finetune/training.py`)
- Lightweight wrapper around the generic `train_model_simple` from `main_2.train`:
  - Uses AdamW optimizer with learning rate = **5e-5**
  - Trains for **2 epochs** by default — enough to demonstrate alignment without heavy GPU needs
  - Prints sample generations after each epoch


#### 5. Generation & Evaluation

- **Quick generation**:  
  `generate_response(prompt, model, device="cuda")`  
  → Formats the instruction, feeds it to the model, and decodes only the response section.

- **Full test set evaluation**:  
  Using the `--gen_test_set` CLI flag generates `instruction-data-with-response.json`.  
  `Note :` it Can be evaluated with `ollama_evaluation.py` against an external LLM to get a 0–100 alignment score. Not present here due to implementation and memory problems


You can run it that way :


```bash 
python -m modular_v.task.instruction_finetune.run_instruction_model \
       --ckpt models/gpt2-medium355M-sft.pth \
       --epochs 0 \
       --ask "Summarise the rules of chess in bullet-points."


```
output: 
Model response:
Input:
The game is played by placing the pieces on the board.


The rules of chess are:
1. The king must be in the center of the board.
2. The queen must be on the opposite side of the board.
3. The rook must be on the first square.
4. The king must be on the square with the largest number of pieces.


Another example :

```bash
--ask "list benefits of eating vegetables "
```
```
Model response:
>> 1. They are rich in nutrients.
2. They are low in calories.
3. They are high in potassium.
4. They are rich in fiber.
5. They are low in sugar.
6. They are rich in iron.
7. They are rich in zinc.
8. They are rich in manganese.
9. They are rich in copper.
10. They are rich in manganese dioxide.
```

Note that some instruction takes much more time to produce an answer, this is directly due to the complexity of the request, a simpler instruction lead to a fast answer

```
### Instruction:
 finish the sentence

### Input:
france is well known for its

Model response:
>> beauty.
france is known for its beauty.
```

#### Key component of the Supervised Fine-Tuning 

| Component | What we did (and why) |
|-----------|----------------------|
| **Prompt template** | Implemented in `format_input()` – keeps Sebastian Raschka’s style but adds an explicit *double line-break* before the assistant section. Guarantees identical offsets across the batch. |
| **Dynamic batch builder** | `custom_collate_fn` pads to *the longest sample of the current batch* and masks every token before `### Response:` with `ignore_index (-100)`. No wasted compute on instructions . |
| **Optimisation recipe** | `AdamW (β₁=0.9, β₂=0.95)` + LR `5e-5`, weight-decay 0.1, *gradient-clipping 1.0* – two epochs on GPT-2-medium (355 M)  |
| **Checkpointing** | We save `gpt2-medium355M-sft.pth` **and** a `instruction-data-with-response.json` for later external grading (e.g. Ollama). |
| **Next steps** | add LoRA adapters to fine-tune. |


## Chapter Notebooks

#### Overview: What We Built & Why It Matters

| **Chapter** | **What We Built & Learnt** | **Why It Matters for the Final Pipelines** |
|-------------|-----------------------------|---------------------------------------------|
| **ch02 · Working with Text Data** | Implemented a full tokenization pipeline using GPT-2 BPE, created a sliding-window dataset and DataLoader, and built a manual word embedding demo. | Developed a solid understanding of how raw text is transformed into integer IDs and tensors — the foundation for all subsequent models. |
| **ch03 · Self-Attention** | Built scaled dot-product attention with causal masking, wrapped it into Multi-Head Attention, and visualized the learned attention weights. | Explained how Transformers model long-range dependencies and how causal masks enable autoregressive generation. |
| **ch04 · Building the GPT Block** | Combined attention and feedforward layers into a `TransformerBlock` with LayerNorm, residuals, and dropout; constructed a full `GPTModel` with configurable depth and width. | Delivered GPT-2 replica capable of loading OpenAI weight checkpoints. |
| **ch05 · Mini Pre-Training & Generation** | Wrote a generic training loop, implemented `generate()` with greedy, top-k, and temperature sampling; added utilities to import OpenAI GPT-2 weights. | Validated our GPT architecture (loss decreases, coherent outputs) before applying it to downstream tasks. |
| **ch06 · Spam Classification Fine-Tune** | Froze 95% of GPT-2 layers, added a 2-class classification head, balanced the SMS dataset, tracked accuracy, and plotted training curves. | Showed how LMs can be adapted for practical tasks like spam filtering with minimal additional weights. |
| **ch07 · Instruction Supervised Fine-Tune** | Introduced InstructGPT-style prompts, implemented custom loss-masking in the collate function, fine-tuned a medium-sized (355M) GPT-2 model, and evaluated responses (optionally with Ollama). | Demonstrated how to align LMs to follow instructions effectively, preparing them for structured user interaction. |


## Results and conclusion

The spam classification pipeline, based on a frozen GPT-2-small backbone with only the final Transformer block and a two-unit linear head left trainable, achieves 95% test accuracy on the balanced SMS-Spam dataset after just two training epochs (batch size 8).

For instruction-following tasks, a GPT-2-medium model fine-tuned over two epochs attains an average score of 48 ± 11 out of 100, as evaluated by an external LLaMA-3 model. Inference times vary depending on the complexity of the request.

This  project shows that rebuilding a Transformer step by step from tokenization to full GPT-2 blocks and downstream fine-tuning can lead to practical, more or less performing models without relying on large external frameworks. The spam classifier reaches over 95% accuracy with just a few megabytes of additional weights, and the instruction-tuned model turns GPT-2 into a task following assistant.

Re-implementing each part helped deepen our understanding of attention, residual connections, and loss masking. The modular codebase (`modular_v/`) now serves as a simple and flexible starting point for future work.

The next phase of the project will build on these foundations to create more concrete, end-to-end applications potentially using recent techniques like LoRA for efficient fine-tuning.



## Credits


**Sebastian Raschka**  :  For his book *Build a Large Language Model From Scratch*, which served as a major reference throughout this work.

**OpenAI and Hugging Face**  : For the GPT-2 model weights, for providing all the necessary packages and data.

**Yousra El Bachir** and **Oleg Bakhteev**  : my EPFL supervisors, who provided essential guidance and support during the course of this project.


