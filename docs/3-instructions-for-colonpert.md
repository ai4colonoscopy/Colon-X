# ColonPert: Benchmarking MLLMs on Reliability in Colonoscopy Tasks

> [!IMPORTANT]
> 📢 To test the MLLMs' performance under challenging types of human perturbation, we developed a subset of tests called ColonPert. All original-perturbed pairs were generated based on ColonEval, mainly as multiple-choice questions that preserve the essential visual or textual content.

## 1. Run ColonPert with Your MLLMs

If you want to use ColonPert to evaluate various models, please first refer to the official code of each model for inference. 
Here we provide two examples for demonstration: one [📝open-source](#open-source-demo) and one [📝closed-source](#closed-source-demo) model. For the models we evaluated and the results in this article, please refer to [📝here](#4-benchmarking-results).

We assume that you have followed the [📝instructions](./1-how-to-download-data.md) to download the ColonPert dataset and organized it as follows:

```text
📁 cache/
└── 📁 data/
    ├── 📁 JSON/                                  # all annotation *.json files
        ├── 📁 ColonPert/                         # evaluation JSONs for benchmarking MLLM generalizability
            ├── TestA_on_image_text_masking.json
            ├── TestB_on_image_misleading_text.json
            ├── TestC_case_contradicting_instruction.json
            └── TestD_emotion_driven_decision_bias.json
```

### 1.1. open-source demo

We demonstrate the workflow using MedGemma. If you wish to test other models, you may adapt the script accordingly.

- Firstly, download the MedGemma-4B checkpoints from [🤗HuggingFace](https://huggingface.co/google/medgemma-4b-it) and place them in `cache/exp/ColonPert`.
- Secondly, set `EXP_MODEL_ID` to the path of your model checkpoints, such as `cache/exp/ColonPert/medgemma-4b-it`.
- Configure `IMAGE_BASE_PATH` and `ROOT_PATH` for the images and JSON files, respectively.
- Prepare the environment following the [🔗MedGemma's instructions](https://github.com/google-health/medgemma).
- Finally, run inference `bash ColonPert/infer_open_source_demo.sh`

### closed-source demo

Here we use [🔗o4-mini](https://platform.openai.com/docs/quickstart) as an example.

- Obtain the API key from OpenAI.
- Set `MODEL=o4-mini-2025-04-16` as the model name.
- Then set `IMAGE_PATH` and `ROOT_PATH` for the images and JSON files.
- Run inference: `bash ColonPert/infer_closed_source_demo.sh`


## 2. One-click Evaluation

To evaluate model reliability on ColonPert, simply modify a few parameters in `ColonPert/eval_reliability.sh` as follows:

- Place your checkpoint under `cache/checkpoints/pert-exp`.
- Set `EXP_MODEL_ID` to the path of the model you want to evaluate.
- Start evaluation:
  
    ```shell
    conda activate coloneval
    bash ColonPert/eval_reliability.sh
    ```
- An example case from our evaluation script is as follows:

    ```shell
    #!/bin/bash

    EXP_MODEL_ID=cache/exp/robust-exp/medgemma-4b-it
    EVAL_MODE=pert

    python ColonEval/eval_engine.py \
        --task_id A \
        --eval_mode $EVAL_MODE \
        --input_file $EXP_MODEL_ID/pred/TestA_on_image_text_masking.json \
        --output_file $EXP_MODEL_ID/pred/Task_A.txt > $EXP_MODEL_ID/pred/eval_task_A_log.txt 2>&1 &
    ```


## 3. Data statistics

Here we present the ColonPert statistics, including the categories and VQA entries for each task.

| Test  | VQA entries |
| :---  | :--- |
| Test.A - On image text masking | 57 |
| Test.B - On image misleading text | 100 |
| Test.C - Case contradicting instruction | 20 |
| Test.D - Emotion driven decision bias | 80 |


## 4. Benchmarking Results

ColonPert evaluates the reliability of MLLMs by analyzing accuracy variations under four human-induced perturbations. 
All prediction files and per-task scores are available on [🔗Google Drive](https://drive.google.com/drive/folders/1EYdf60GQmDhV7NZ7CemrONyc-yX5Ewax?usp=sharing).
Table 1 reports the performance of six leading MLLMs.


<p align="center">
    <img src="../assets/benchmarking_results_colonpert.jpg"/> <br />
    <em> 
    Table 1:  Reliability test of six multimodal large language models (MLLMs) on ColonPert.
    </em>
</p>


## 5. Visual cases

Figure 1 illustrates representative examples from ColonPert and the model's performance under four human-induced perturbations, including on-image text in visual prompts (Test.A & Test.B) and explicitly through textual prompts (Test.C & Test.D).

- Test.A: Obscure embedded texts (e.g., device information) in images to test text bias.
- Test.B: Overlay erroneous texts in image corners to test resistance to visual textual interference.
- Test.C: Inject case-contradicting descriptions (e.g., describing malignant as "benign") into textual prompts.
- Test.D: Incorporate patient emotional states (e.g., anxiety, fear) into prompts to test for emotional bias.

<p align="center">
    <img src="../assets/ColonPert_cases.jpg"/> <br />
    <em> 
    Figure 1: Illustration of four types of human-induced perturbations.
    </em>
</p>

