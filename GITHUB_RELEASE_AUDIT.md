# GitHub 公開前查核

查核日期：2026-10-19。使用者已授權由研究助理選擇名稱並建立**公開**倉庫。已建立 `ChenBill900703/pope-internvl3-2b-reproduction`；本文件區分內容查核、倉庫建立與完整上傳，不以空倉庫表示內容已發布。

## 研究內容

- 已產生七份指定Markdown文件及VQA D流程／比較專章，另有重現說明、資料政策、來源與授權說明、彙整JSON/CSV及八張新圖（各含PNG/SVG）。
- 原論文Table 1–12、Figure 2與Appendix A–F已核對，原作者數值與本次新模型數值分開；歷史比較圖註明架構、年代、訓練資料與原生介面差異。
- 新的唯讀查核確認五個正式資料庫共63,182筆保存列（含跨實驗重用，**不是同數獨立新推論**），SQLite與JSONL一致。
- 27,000筆COCO／A-OKVQA／GQA POPE原題、標籤及圖片SHA核對；9條件各3,000題的混淆矩陣與sklearn吻合。
- 12,966筆描述以原CHAIR解析器重新解析（含各表重用），24列HRA/HRC與632列物件頻率重新計算；分母政策明列，不冒稱完全重現Table 2。
- 11,216筆VQA回答、完整參考、圖／題對應及GQA 5,461筆前導零ID核對；A/B/C規則與共識彙總可重算，25道AOK排除依原旗標。未重跑GPU。
- Captioning BLEU/METEOR/ROUGE仍引用已實際執行並保存的評分器輸出；本次查核不宣稱重新執行Java評分器。
- VQA D凍結後挑戰17題的5次錯誤接受可重現；明列不得採為正式準確率。沒有人工一致率、沒有成功執行的開源LLM判分，也沒有付費API。
- 另新增VQA D流程與同樣本A/B/C/D比較專章；240筆凍結D判斷逐筆重播一致、八列樣本信用重新計算一致，見public_data/vqa_D_export_validation.json。這是忠實匯出查核，不是認可D語意正確性。
- Table 5與原Table 6評判未完成，清楚標記；FP16異常是NaN而非OOM，正式BF16無量化。

## 原始資料與公開範圍

- 本機舊檔保護清單共855檔，僅原README依使用者要求更新；其餘854檔內容SHA均保持一致，包括正式結果、原技術報告與VQA A/B/C/D。
- 原README及可編輯程式／文件另有本機備份。模型、圖片原地保留，沒有為備份而重複下載或複製大型權重。
- 根目錄 `.gitignore` 預設排除所有非允許項目；不包含原始SQLite/JSONL、權重、影像、第三方clone、舊本機日誌／報告、PDF全文、含官方逐題問題／答案的審查ZIP與個人資料。
- 公開資料只有彙整結果、原創程式、必要版本／來源SHA，以及授權已確認的A-OKVQA evaluator（附Apache-2.0原授權）。CHAIR原碼未確認明確授權，沒有再散布；本專案MIT只適用原創貢獻。
- 新Git庫沒有匯入原第三方倉庫的歷史。使用GitHub noreply提交身分，不公開私人email。提交前檢查實際Git檔案清單、檔案大小、敏感字串與私有路徑。
- `.gitattributes`保留原始位元組，避免Windows換行轉換破壞凍結規則與公開SHA。公開清單不含自身與本發布狀態文件，避免自我引用雜湊。

## 執行過的查核與證據

| 查核 | 執行／證據 | 結論 |
| --- | --- | --- |
| 本機逐筆唯讀核對 | `release_tools/verify_local.py`；[local_validation.json](public_data/local_validation.json) | 通過；不改原始結果 |
| 公開彙整數據、CSV及程式語法 | `release_tools/check_public.py`及`--staged` | 通過；129檔的Git暫存內容與允許清單逐位元組相同，不需GPU |
| VQA v2.0規則 | 20個原開發控制案例、凍結SHA比對 | 開發檢查通過，非人工準確率驗證 |
| D可靠性反例 | 17個凍結後合成挑戰；重現5次錯誤接受 | **D評判可靠性未通過**，不能混稱整個研究全部通過 |
| Markdown與圖表 | 檢查本地連結均在公開檔案清單；PNG/SVG實際生成並檢視 | 不以本機存在但Git忽略的檔案充當公開連結 |
| 公開SHA | [RELEASE_MANIFEST.json](public_data/RELEASE_MANIFEST.json) | 逐檔內容清單；原始私有成果另有來源SHA |

## 發布狀態

**主內容已發布至[公開倉庫](https://github.com/ChenBill900703/pope-internvl3-2b-reproduction)的main分支。這份發布稽核為最後一項提交；原始私有成果仍完整保留本機。**

主內容128檔已逐一用GitHub遠端遞迴檔案樹與本機Git blob SHA核對：缺漏0、額外檔案0、內容不一致0。此處128檔不含最後提交的本稽核；含本稽核的正式公開範圍共129檔。最終提交SHA與完整129檔核對記錄另外保存於本機，避免本文件內嵌自身提交SHA造成循環。

先前自動核准審查曾因「Selected model is at capacity」失敗，當時Git與瀏覽器動作沒有執行，並非GitHub拒絕帳號。使用者要求重試後，正常核准恢復，Git暫存與GitHub網頁建立倉庫成功。沒有選用第三方POPE上游作為推送目標。

Git Credential Manager的標準裝置登入要求私人倉庫完整控制、gist與workflow等超出本次範圍的權限，自動審查拒絕該授權。已在GitHub取消，程序回報access_denied，未取得該登入憑證。現有GitHub連接器對新倉庫寫入亦回報403，因此改用既有登入的GitHub網頁，僅提交此公開倉庫的已核對內容；沒有繞過前述授權拒絕或取得私人倉庫權限。

可公開內容129檔、約1.92 MB，本機 `deliverables/GITHUB_PUBLIC_RELEASE_2026-10-19.zip`提供同一公開範圍，正式交付以最新同名 `.sha256`及解壓後CPU檢查為準。封裝程式每次先做公開檢查，再重新打包、驗證CRC、實際解壓並重播離線檢查。可攜VQA評分程式已在新輸出目錄重算全部A/B/C，與正式診斷完全一致，未更動原始結果。依賴套件 `pip check` 通過。

GitHub已啟用私人email隱藏，頁面明示網頁Git操作採GitHub noreply地址，已從實際公開提交確認。資料字典經網頁編輯時曾出現換行差異，已改用原始檔上傳補正，未因此修改正式來源雜湊。發布採多筆有意義的網頁Git提交，並在本機建立公開範圍的乾淨Git提交；沒有force push、沒有覆蓋既有遠端歷史，沒有開啟超出本次需要的Git登入權限。
