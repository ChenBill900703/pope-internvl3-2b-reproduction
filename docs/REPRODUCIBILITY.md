# 重現與重新計分說明

## 先選擇查核層級

1. **公開彙整查核**：不需資料集、模型或GPU。檢查公開JSON/CSV、混淆矩陣、圖表與Markdown連結。這只能驗證公開數據內部一致性。
2. **本機逐筆再核對**：需要研究者保留的 `results/`、`analysis/`、資料及第三方原程式。執行 `release_tools/verify_local.py`，只讀取既有來源，結果寫到 `public_data/`；不啟動模型、不重寫正式輸出。SQLite使用 `mode=ro`，保留WAL可見性，不能任意加 `immutable=1` 而漏掉已提交WAL頁。
3. **新實驗重新推論**：在獨立的新工作目錄取得合法資料與模型後執行原研究腳本。這需要時間、GPU及來源資源，並未在這次公開整理中重新執行。不要在原始封存工作目錄啟動新的推論來覆蓋歷史進度。

## 環境

實測環境為Windows、Python 3.12.14、RTX 3070 Ti 8192 MiB、torch 2.6.0+cu124、torchvision 0.21.0+cu124、Transformers 4.52.1。沿用已有Python、CUDA與Driver；不重裝系統工具。

在新工作副本使用既有Python 3.12建立獨立venv：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-analysis.txt
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe release_tools/check_public.py
.venv\Scripts\python.exe release_tools/plot_public.py
```

`requirements.lock.txt`記錄實際環境；CPU彙整查核本身只需要標準函式庫。重新繪圖需matplotlib/numpy。Captioning工具另需既有Java，METEOR路徑／逾時處理見 `scripts/evaluate_caption_metrics.py`，不更換指標公式。

## 外部來源與目錄

第三方碼與資料不隨公開庫整包散布。使用者須自行從來源取得，遵守[權利說明](../THIRD_PARTY_NOTICES.md)；其中CHAIR來源未確認明確授權，完整重現須先確認使用條件。

| 資源 | 固定來源／必要本機路徑 |
| --- | --- |
| POPE | [RUCAIBox/POPE](https://github.com/RUCAIBox/POPE)，commit `08d957b917e5a378a2f99d35b6293c536a66298b`，放 `third_party/POPE/` |
| 模型 | [InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B/tree/899155015275a9b7338c7f4677e19c784e0e5a21)，放 `models/InternVL3-2B/` |
| COCO | 原val2014影像放 `data/coco/val2014/`；人工caption註記見prepare_followup腳本 |
| GQA | 原官方影像放 `data/gqa/images/`，官方balanced問答來源與選取見prepare_gqa_vqa |
| A-OKVQA | 官方v1p0驗證註記；保留全部十個direct answers、difficult_direct_answer |
| CHAIR | [LisaAnne/Hallucination](https://github.com/LisaAnne/Hallucination)，commit `6e4d33c4bedf6dfcd24c61bf527f40714717be47`；`third_party/CHAIR/utils/chair.py` 與 `data/synonyms.txt` |
| Pattern | 原英語inflect.py與其BSD授權；放 `third_party/pattern/`；來源雜湊見設定 |

在新副本取得POPE後才能準備資料；不要重下載已存在且雜湊正確的檔案：

```powershell
git clone https://github.com/RUCAIBox/POPE third_party/POPE
git -C third_party/POPE checkout 08d957b917e5a378a2f99d35b6293c536a66298b
.venv\Scripts\python.exe scripts/prepare_data.py
.venv\Scripts\python.exe scripts/prepare_extended.py
.venv\Scripts\python.exe scripts/fetch_model.py
.venv\Scripts\python.exe scripts/prepare_nltk.py
```

下載腳本會先檢查現有檔案。模型執行需要 `trust_remote_code=True`；使用固定且自行審閱的官方snapshot。模型與資料來源權利各自獨立，本專案MIT不授權重新散布這些資源。

## 實際提示、生成與保存

設定來源：[experiment_settings.json](../public_data/experiment_settings.json)。每題加模型所需 `<image>\n` 與原生聊天模板，`history=None`，不加「只答Yes/No」。BF16、無量化、無FlashAttention、batch1、seed0、greedy、num_beams1、max_new_tokens1024。正式採BF16是因FP16語言logits NaN，並非OOM。部分早期128等較小上限回答在正常EOS且相同圖／提示時重用，來源保存於本機逐題記錄；不宣稱所有列都以相同上限重新生成。

SQLite每題提交，另保留JSONL與token IDs；原腳本按既有有效列斷點續跑。程式雜湊會改變結果目錄，因此新程式版本不應偷偷將歷史列併為相同設定。Watchdog用鎖防止同時啟動兩個GPU工作，有限重試，錯誤另存；檔案鎖短暫衝突使用atomic_io重試。這些機制不保證斷電、硬碟故障或任何環境下永不失敗。

## 新工作副本的逐項執行

以下是已保存腳本的入口，**不要在封存原實驗目錄重跑**。第一次跑需準備 `STATUS.md`、`logs/`、`results/`，且先檢查所有相依資源。準備與執行請逐項確認完成再往下，不同GPU工作不要並行。

```powershell
New-Item -ItemType Directory -Force logs,results | Out-Null
if (-not (Test-Path STATUS.md)) { Set-Content STATUS.md '# Local experiment status' }
.venv\Scripts\python.exe scripts/run_pope.py --pilot-only
.venv\Scripts\python.exe scripts/run_pope.py
.venv\Scripts\python.exe scripts/run_captions.py
.venv\Scripts\python.exe scripts/prepare_followup.py
.venv\Scripts\python.exe scripts/prepare_gqa.py
.venv\Scripts\python.exe scripts/run_stability.py
.venv\Scripts\python.exe scripts/run_appendix_d.py
.venv\Scripts\python.exe scripts/prepare_gqa_vqa.py
.venv\Scripts\python.exe scripts/run_vqa.py
```

`prepare_followup.py`會準備Table4題目、COCO caption註記、A-OKVQA子集及AppendixD問題；`--table4-only`則只準備Table4。須先準備相依資源再執行各runner；CHAIR原碼亦須先備妥。上列不是承諾公開庫不需外部資源便可一鍵重現全部實驗。不同runner的progress檔由其主程式建立；資料下載、計分及模型執行是不同工作。

## 既有輸出的CPU重新計分

```powershell
.venv\Scripts\python.exe release_tools/verify_local.py
.venv\Scripts\python.exe scripts/analyze_captions.py --folder results/captions-78966472f82d508d
.venv\Scripts\python.exe scripts/analyze_consistency.py --help
.venv\Scripts\python.exe scripts/evaluate_caption_metrics.py --help
```

第一個新唯讀核對器供本機來源再驗證，後三個原計分入口會寫入其指定輸出，請只在**副本**上重新生成報表。公開版不隨附原始逐題資料，僅有aggregate檢查；來源缺失時應明確失敗，不能產生示範分數冒稱實測。

VQA A/B/C規則與可攜重新計分程式在 `research_extensions/vqa/`。若已合法取得本機私有審查ZIP的兩個prediction CSV，可使用：

```powershell
.venv\Scripts\python.exe research_extensions/vqa/run_analysis.py --csv-dir PATH_TO_PRIVATE_EXPORTS --output tmp/vqa-cpu-rerun
```

其中保留原參考答案與排除旗標，擷取器不讀正解。D規則及凍結後挑戰程式在 `research_extensions/semantic_audit/`，可在新副本執行挑戰，應重現其失誤，**不是通過可靠性認證**。沒有人工語意真值、沒有原文ChatGPT-assisted分數。

## 查核紀錄與原始保存

新公開版查核見 [local_validation.json](../public_data/local_validation.json)；原始來源逐檔SHA見 [private_source_hashes.json](../public_data/private_source_hashes.json)。原技術報告、所有完成輸出及VQA A/B/C/D資料在本機原位置保留；原README與可編輯檔另存本機 `local_backups/release-2026-10-19/`。此備份不含大型模型／影像的第二份複製，原檔仍原地保留。
