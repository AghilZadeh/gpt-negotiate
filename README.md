# Negotiate-Gpt

This is the primary repo for the negotiate-GPT project, with the primary emphasis on developing a customized GPT that can provide more factual information on salary and benefits given a customized dataset.  

It's a fork from Alpaca-LoRA, which will be used as the basis for training our initial set of models.  This is where we'll setup project workflows and monitor on going activity.


### Local Setup

1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

1. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

### Training (`finetune.py`)

This file contains a straightforward application of PEFT to the LLaMA model,
as well as some code related to prompt construction and tokenization.
PRs adapting this code to support larger models are always welcome.

Example usage:

```bash
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca'
```

We can also tweak our hyperparameters:

```bash
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length
```

### Inference (`generate.py`)

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `tloen/alpaca-lora-7b`, and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.

Example usage:

```bash
python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
```

### Official weights

The most recent "official" Alpaca-LoRA adapter available at [`tloen/alpaca-lora-7b`](https://huggingface.co/tloen/alpaca-lora-7b) was trained on March 26 with the following command:

```bash
python finetune.py \
    --base_model='decapoda-research/llama-7b-hf' \
    --num_epochs=10 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='./lora-alpaca' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=8
```

### Checkpoint export (`export_*_checkpoint.py`)

These files contain scripts that merge the LoRA weights back into the base model
for export to Hugging Face format and to PyTorch `state_dicts`.
They should help users
who want to run inference in projects like [llama.cpp](https://github.com/ggerganov/llama.cpp)
or [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp).

### Docker Setup & Inference

1. Build the container image:

```bash
docker build -t alpaca-lora .
```

2. Run the container (you can also use `finetune.py` and all of its parameters as shown above for training):

```bash
docker run --gpus=all --shm-size 64g -p 7860:7860 -v ${HOME}/.cache:/root/.cache --rm alpaca-lora generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
```

3. Open `https://localhost:7860` in the browser

### Docker Compose Setup & Inference

1. (optional) Change desired model and weights under `environment` in the `docker-compose.yml`

2. Build and run the container

```bash
docker-compose up -d --build
```

3. Open `https://localhost:7860` in the browser

4. See logs:

```bash
docker-compose logs -f
```

5. Clean everything up:

```bash
docker-compose down --volumes --rmi all
```

### Resources

- [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp), a native client for running Alpaca models on the CPU
- [Alpaca-LoRA-Serve](https://github.com/deep-diver/Alpaca-LoRA-Serve), a ChatGPT-style interface for Alpaca models
- [AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned), a project to improve the quality of the Alpaca dataset
- [GPT-4 Alpaca Data](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) a project to port synthetic data creation to GPT-4
- [dolly-15k-instruction-alpaca-format](https://huggingface.co/datasets/c-s-ale/dolly-15k-instruction-alpaca-format), an Alpaca-compatible version of [Databricks' Dolly 15k human-generated instruct dataset](https://github.com/databrickslabs/dolly/tree/master/data) (see [blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm))
- [Alpaca-LoRA MT](https://github.com/juletx/alpaca-lora-mt), a project to finetune models with [machine-translated Alpaca data](https://huggingface.co/datasets/HiTZ/alpaca_mt) in 6 Iberian languages: Portuguese, Spanish, Catalan, Basque, Galician and Asturian.
- Various adapter weights (download at own risk):
  - 7B:
    - 3️⃣ <https://huggingface.co/tloen/alpaca-lora-7b>
    - 3️⃣ <https://huggingface.co/samwit/alpaca7B-lora>
    - **4️⃣ <https://huggingface.co/chansung/gpt4-alpaca-lora-7b>**
    - 🚀 <https://huggingface.co/nomic-ai/gpt4all-lora>
    - 🇧🇷 <https://huggingface.co/22h/cabrita-lora-v0-1>
    - 🇨🇳 <https://huggingface.co/qychen/luotuo-lora-7b-0.1>
    - 🇨🇳 <https://huggingface.co/ziqingyang/chinese-alpaca-lora-7b>
    - 🇯🇵 <https://huggingface.co/kunishou/Japanese-Alapaca-LoRA-7b-v0>
    - 🇫🇷 <https://huggingface.co/bofenghuang/vigogne-lora-7b>
    - 🇹🇭 <https://huggingface.co/Thaweewat/thai-buffala-lora-7b-v0-1>
    - 🇩🇪 <https://huggingface.co/thisserand/alpaca_lora_german>
    - 🇵🇱 <https://huggingface.co/mmosiolek/polpaca-lora-7b>
    - 🇵🇱 <https://huggingface.co/chrisociepa/alpaca-lora-7b-pl>
    - 🇮🇹 <https://huggingface.co/teelinsan/camoscio-7b-llama>
    - 🇷🇺 <https://huggingface.co/IlyaGusev/llama_7b_ru_turbo_alpaca_lora>
    - 🇺🇦 <https://huggingface.co/robinhad/ualpaca-7b-llama>
    - 🇮🇹 <https://huggingface.co/mchl-labs/stambecco-7b-plus>
    - 🇪🇸 <https://huggingface.co/plncmm/guanaco-lora-7b>
    - 🇬🇧 🇪🇸 🇵🇹 <https://huggingface.co/HiTZ/alpaca-lora-7b-en-pt-es-ca-eu-gl-at>
  - 13B:
    - 3️⃣ <https://huggingface.co/Angainor/alpaca-lora-13b>
    - 3️⃣ <https://huggingface.co/chansung/alpaca-lora-13b>
    - 3️⃣ <https://huggingface.co/mattreid/alpaca-lora-13b>
    - 3️⃣ <https://huggingface.co/samwit/alpaca13B-lora>
    - **4️⃣ <https://huggingface.co/chansung/gpt4-alpaca-lora-13b>**
    - 🇯🇵 <https://huggingface.co/kunishou/Japanese-Alapaca-LoRA-13b-v0>
    - 🇰🇷 <https://huggingface.co/chansung/koalpaca-lora-13b>
    - 🇨🇳 <https://huggingface.co/facat/alpaca-lora-cn-13b>
    - 🇨🇳 <https://huggingface.co/ziqingyang/chinese-alpaca-lora-13b>
    - 🇪🇸 <https://huggingface.co/plncmm/guanaco-lora-13b>
    - 🇮🇹 <https://huggingface.co/mchl-labs/stambecco-13b-plus>
    - 🇬🇧 🇪🇸 🇵🇹 <https://huggingface.co/HiTZ/alpaca-lora-13b-en-pt-es-ca-eu-gl-at>
  - 30B:
    - 3️⃣ <https://huggingface.co/baseten/alpaca-30b>
    - 3️⃣ <https://huggingface.co/chansung/alpaca-lora-30b>
    - **4️⃣ <https://huggingface.co/chansung/gpt4-alpaca-lora-30b>**
    - 🇯🇵 <https://huggingface.co/kunishou/Japanese-Alapaca-LoRA-30b-v0>
    - 🇬🇧 🇪🇸 🇵🇹 <https://huggingface.co/HiTZ/alpaca-lora-30b-en-pt-es-ca-eu-gl-at>
  - 65B
    - <https://huggingface.co/chansung/alpaca-lora-65b>
    - 🇬🇧 🇪🇸 🇵🇹 <https://huggingface.co/HiTZ/alpaca-lora-65b-en-pt-es-ca>
- [alpaca-native](https://huggingface.co/chavinlo/alpaca-native), a replication using the original Alpaca code
- [llama.onnx](https://github.com/tpoisonooo/llama.onnx), a project to inference alpaca with onnx format


