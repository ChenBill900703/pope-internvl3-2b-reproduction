# Third-party sources and rights

The root MIT license applies only to this project's original contributions. It does **not** relicense upstream code, datasets, annotations, model weights, the source paper, or generated text rights controlled by other terms.

| Source | Local evidence / public treatment |
| --- | --- |
| [RUCAIBox/POPE](https://github.com/RUCAIBox/POPE/tree/08d957b917e5a378a2f99d35b6293c536a66298b) | Upstream LICENSE is MIT, copyright 2024 Yifan Li. No full upstream repository or question/annotation corpus is redistributed. Answer parsing implements the same documented rule. |
| [CHAIR / LisaAnne/Hallucination](https://github.com/LisaAnne/Hallucination/tree/6e4d33c4bedf6dfcd24c61bf527f40714717be47) | No explicit top-level license was found in the locally acquired snapshot. Upstream code/synonym data are **not bundled or relicensed**. The local adapter loads separately acquired sources; users must establish appropriate rights before using them. |
| [Pattern](https://github.com/clips/pattern) | Locally acquired LICENSE.txt is BSD 3-clause, University of Antwerp 2011–2013. Inflection source not redistributed here; its original license must accompany separately obtained copies. |
| [A-OKVQA evaluation](https://github.com/allenai/aokvqa/tree/83786de0b24c4fef0bfe39420684f6f73d9f3581) | Apache-2.0 license confirmed locally. Only the evaluator needed for offline grading is included under research_extensions/vqa/sources, with full license and fixed source metadata. This does not grant rights to COCO images or all annotations. |
| [InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B/tree/899155015275a9b7338c7f4677e19c784e0e5a21) | Model card/license and component terms apply separately; weights, native model code and images are not bundled. Runtime reads the official preprocessing from the separately acquired fixed model card. No blanket MIT license of the model is asserted. |
| [COCO](https://cocodataset.org/), [GQA](https://cs.stanford.edu/people/dorarad/gqa/), A-OKVQA annotations | No images, full questions, reference answer corpora or annotation files redistributed. Obtain data from the original sources subject to their terms and original image rights. |
| [coco-caption / pycocoevalcap](https://github.com/tylin/coco-caption) and METEOR | Installed separately; scoring algorithms and bundled Java/paraphrase resources retain their licenses. Our wrapper changes Windows process startup, deadlines and cleanup, not metric formulas. |
| [POPE paper](https://aclanthology.org/2023.emnlp-main.20/) | Cited and summarized in original explanatory prose. Selected published numerical statistics are identified by table/page. PDF, extracted full text, screenshots and original paper figures are not redistributed. New figures are generated from this project's aggregates plus explicitly marked published statistics. |

Dependency packages in requirements retain their own licenses. This is a distribution inventory, not a claim that all upstream assets share one license. The public repository deliberately excludes assets whose redistribution terms have not been verified.
