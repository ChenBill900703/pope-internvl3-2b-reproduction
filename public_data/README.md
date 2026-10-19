# 公開彙整資料欄位

這裡不包含逐題官方問題、參考答案或原始模型回答。所有本次分數由本機正式結果重新核對，歷史論文值另存 `paper_reference.json`，不可當成本專案模型的輸出。

| 檔案 | 範圍與欄位 |
| --- | --- |
| coco_pope.json / .csv | 三策略各3000題；n、TP/TN/FP/FN及0–1 accuracy/precision/recall/f1/yes_ratio |
| chair.json / .csv | I1/I2各2000圖；CHAIRi/s為0–1，次數保留重複，長度有空白與NLTK兩種 |
| prompt_stability.json | P1–P4問答、C1–C4描述，分母與指標分開 |
| prompt_stability_summary.json | 四提示平均與母體標準差；比例轉百分比須乘100，標準差以百分點表示 |
| aokvqa_gqa_pope.json / .csv | 六條件各3000題，是SEEM衍生POPE不是VQA |
| hra_hrc.json / .csv | prompt_id、anchor、k、images及有幻覺圖數；條件平均、補0平均、合併提及比均0–1 |
| object_frequencies.json / .csv | prompt/anchor/rank/object；image_frequency為GT排序頻率；hallucination_count為本次提及次數 |
| appendixB_cumulative.json | top_k及合併幻覺次數累積比例 |
| caption_metrics.json / .csv | I1/I2各500圖；BLEU1/2、METEOR、ROUGE-L均0–1 |
| caption_pope_consistency.json | 六條件下No與caption提及配對之分子、分母與比例 |
| vqa_supplementary.json / .csv | A/B/C有效題、排除、score_sum、score_percent（此欄已是百分比）；模糊／未解析數為C的解析狀態 |
| semantic_audit_challenges.json | D凍結後17個人工設計挑戰案例，自動判分結果，非人工標註準確率 |
| vqa_sample_abcd.json / .csv | 原固定240題的A/B/C/D未加權樣本信用表；不同於全量成績，D不作正式準確率 |
| vqa_semantic_summary.json | D全240題判定與C/D差異方向之原始彙整，無逐題官方問題 |
| vqa_c_state_d_crosstab.json | C各解析狀態與D四種判定交叉計數，含7道排除題 |
| vqa_D_export_validation.json | 凍結D的240筆判斷重播及八列樣本A/B/C/D信用重算；驗證匯出忠實，並非認可D的語意正確性 |
| paper_reference.json | 原文Table3等已發表數值、頁碼與限制；數值已為百分比 |
| experiment_settings.json | 固定版本、生成、影像處理、提示及第三方來源 |
| local_validation.json | 2026-10-19本機唯讀查核範圍；不代表公開原始資料 |
| private_source_hashes.json | 本機正式結果相對路徑及SHA-256；原始檔未隨公開版散布 |

CSV由同名JSON生成，不經人工改數字。欄位內複雜值採JSON字串。檔案雜湊見[公開清單](RELEASE_MANIFEST.json)，該清單不含自身及會因提交狀態更新的發布稽核文件。
