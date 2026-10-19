# POPE Reproduction and Extension with InternVL3-2B

[繁體中文](README.md) · [Paper summary (zh-TW)](PAPER_SUMMARY_zh-TW.md) · [Results](EXPERIMENT_RESULTS_zh-TW.md) · [Limitations](LIMITATIONS_zh-TW.md)

This project applies the evaluation methods of [Li et al., EMNLP 2023](https://aclanthology.org/2023.emnlp-main.20/) to **OpenGVLab/InternVL3-2B**. It does not retrain or rerun the original baseline models. The paper did not evaluate InternVL3-2B.

See the [D process and same-sample A/B/C/D comparison](VQA_D_PROCESS_AND_COMPARISON_zh-TW.md) for the exploratory semantic audit and its failed reliability checks.

## Main results

COCO POPE F1: Random **93.0403%**, Popular **87.1491%**, Adversarial **84.4875%**, using the authors' fixed questions (3,000 each). These exceed the historical InstructBLIP Table 3 values, but the models, training data and native interfaces differ.

![COCO POPE](figures/coco_pope_f1.png)

CHAIR on the authors' 2,000 COCO image IDs: I1 CHAIRi/CHAIRs **2.8634/4.9500%**; I2 **4.7814/18.0000%**. Both are worse than the original InstructBLIP values, so we do not claim overall superiority. Four-prompt POPE F1 mean **91.9855%**, population SD **0.9340 percentage points**. Published SEEM-derived A-OKVQA/GQA POPE questions (18,000 total), caption metrics, frequency/co-occurrence analyses and caption/POPE consistency were also evaluated.

## Settings and incomplete reproduction

Windows, RTX 3070 Ti 8GB, Python 3.12.14, torch 2.6.0+cu124, transformers 4.52.1; BF16 without quantization. FP16 produced non-finite language logits, **not an OOM**. Greedy decoding, batch 1, seed 0, native chat interface and official 448px dynamic image preprocessing. Model revision `899155015275a9b7338c7f4677e19c784e0e5a21`; POPE commit `08d957b917e5a378a2f99d35b6293c536a66298b`.

- Original COCO SEEM Table 5 was **not reproduced**: the original fixed questions could not be verified.
- Original ChatGPT-assisted VQA Table 6 / Appendix E grading was **not reproduced**.
- Supplementary local VQA A/B/C scores: A-OKVQA **0 / 1.0204 / 9.7959%** (490 eligible, 25 excluded); GQA **0 / 0.7196 / 36.3798%** (10,701). These are format diagnostics, not validated semantic accuracy or the paper's grader.
- Exploratory symbolic D evaluated only the fixed 240-item review sample. It falsely accepted 5 of 17 post-freeze challenge cases and must **not** be adopted as formal accuracy or expanded to all items. No independent human labels or successful open-source LLM judgments exist.
- Single-model, POPE-image-derived VQA subsets cannot establish cross-model correlation.

## Reproduce and inspect

Read [reproducibility instructions](docs/REPRODUCIBILITY.md), [aggregate data dictionary](public_data/README.md), [data policy](docs/DATA_POLICY.md), and [third-party notices](THIRD_PARTY_NOTICES.md). With Python 3.12, create an isolated `.venv` and run `.venv/Scripts/python.exe release_tools/check_public.py` for CPU-only aggregate/package checks. Charts use `release_tools/plot_public.py` after installing analysis requirements. Full raw verification requires separately acquired local resources.

The public repository includes original research code, aggregate JSON/CSV, figures, source hashes and reports. It excludes weights, images, caches, private paths, raw third-party question/reference corpora and uncertain-license vendored code. Local original artifacts remain preserved. Original contributions are [MIT licensed](LICENSE); third-party assets retain their own terms.
