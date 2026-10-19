"""Build reader documents and figures from verified aggregate JSON, no GPU.

Published paper values below are explicitly transcribed historical results,
never presented as outputs of this project's model.
"""
from pathlib import Path
import csv,json,shutil
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'public_data';F=ROOT/'figures'
def read(n):return json.loads((D/n).read_text(encoding='utf-8'))
def write(n,text):
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text.strip()+'\n',encoding='utf-8')
def table(rows,cols):
    return '| '+' | '.join(title for key,title in cols)+' |\n| '+' | '.join('---' for _ in cols)+' |\n'+'\n'.join('| '+' | '.join(str(r.get(key,'')) for key,title in cols)+' |' for r in rows)
def pct(v):return f'{100*v:.4f}'
def paper_data():
    # PDF p7 / printed p298, Table 3, F1 in percent, visually checked.
    models={'mPLUG-Owl':[68.06,66.79,66.82],'LLaVA':[68.65,67.72,66.98],'MultiModal-GPT':[66.68,66.67,66.67],'MiniGPT-4':[78.86,72.21,71.37],'InstructBLIP':[89.29,83.45,78.45]}
    paper={'source':'https://aclanthology.org/2023.emnlp-main.20/','table3_f1_percent':models,'table3_source':'PDF page 7 / printed 298, Table 3; published rounded values, not rerun','table1_instructblip_percent':{'I1':{'CHAIRi':2.6,'CHAIRs':3.7},'I2':{'CHAIRi':2.5,'CHAIRs':3.4}},'table4_llava':{'pope_f1_percent':[68.65,66.83,66.67,67.58],'chairi_percent':[10.5,18.8,14.6,11.6]},'table11_instructblip_f1_percent':{'aokvqa':[87.20,79.53,73.75],'gqa':[85.32,76.89,72.39]},'table12_percent':{'InstructBLIP':{'POPE':89.29,'BLEU-1':59.5,'BLEU-2':45.2,'METEOR':22.6,'ROUGE-L':42.3},'LLaVA':{'POPE':68.65,'BLEU-1':22.0,'BLEU-2':13.9,'METEOR':19.8,'ROUGE-L':22.4},'MiniGPT-4':{'POPE':78.86,'BLEU-1':41.1,'BLEU-2':28.8,'METEOR':25.6,'ROUGE-L':44.7}},'other_sources':{'table1':'PDF p3 / printed 294','table4':'PDF p8 / printed 299','table11_table12':'PDF p14 / printed 305'},'comparison_policy':'Historical context only. Different model generations, architectures, training corpora and native chat/image processors. Original baselines not rerun.'}
    write('public_data/paper_reference.json',json.dumps(paper,ensure_ascii=False,indent=2));return paper
def charts(paper):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    F.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    colors=['#2563a0','#21958b','#d89a35'];labels=['Random','Popular','Adversarial']
    def finish(fig,name,source):
        fig.text(.02,.015,source,fontsize=8,ha='left',va='bottom');fig.tight_layout(rect=(0,.09,1,1))
        fig.savefig(F/f'{name}.png',dpi=180);fig.savefig(F/f'{name}.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));yy=[100*r['f1'] for r in read('coco_pope.json')]
    bars=ax.bar(labels,yy,color=colors);ax.bar_label(bars,fmt='%.2f%%',padding=4);ax.set(ylim=(0,104),ylabel='F1 (%)',title='InternVL3-2B: COCO POPE (3,000 questions / strategy)')
    finish(fig,'coco_pope_f1','Source: verified coco_pope.json; author-provided fixed questions; BF16, greedy decoding.')
    fig,ax=plt.subplots(figsize=(11,6));xx=np.arange(3)
    for model,values in paper['table3_f1_percent'].items():ax.plot(xx,values,'o--',label=model+' (paper, 2023)',alpha=.75)
    ax.plot(xx,yy,'o-',color='#111827',linewidth=3,label='InternVL3-2B (this project)')
    ax.set(xticks=xx,xticklabels=labels,ylabel='F1 (%)',ylim=(0,103),title='COCO POPE: published baselines and this project')
    ax.legend(fontsize=9,ncol=2,loc='lower left')
    finish(fig,'coco_historical_comparison','Source: paper Table 3 (p298) and coco_pope.json. Historical comparison, not controlled model replication.\nDifferent architectures, training data, native prompts and image preprocessing; baseline models not rerun.')
    fig,axes=plt.subplots(1,2,figsize=(10,5));ch=read('chair.json')
    for ax,key,name in zip(axes,['CHAIRi','CHAIRs'],['Object instances','Captions with hallucination']):
        bars=ax.bar(['I1','I2'],[100*r[key] for r in ch],color=colors[:2]);ax.bar_label(bars,fmt='%.2f%%',padding=4)
        ax.set(ylabel=key+' (%) - lower is better',title=name,ylim=(0,23))
    fig.suptitle('InternVL3-2B CHAIR: 2,000 author-matched COCO images / prompt')
    finish(fig,'chair_prompts','Source: chair.json, recomputed with original CHAIR parser. I1 short caption; I2 brief description of given image.')
    ss=read('prompt_stability.json');fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,rr,key,title in [(axes[0],ss[:4],'f1','POPE F1 (higher is better)'),(axes[1],ss[4:],'CHAIRi','CHAIRi (lower is better)')]:
        bars=ax.bar([r['prompt_id'] for r in rr],[100*r[key] for r in rr],color='#2563a0');ax.bar_label(bars,fmt='%.2f',padding=4)
        ax.set(ylabel='Percent (%)',title=title,ylim=(0,105 if key=='f1' else 7))
    fig.suptitle('Prompt stability: separate metrics and denominators')
    finish(fig,'prompt_stability','Source: prompt_stability.json; POPE 3,000 questions / prompt; CHAIR 2,000 captions / prompt.\nPopulation SD: POPE F1 0.9340 percentage points; CHAIRi 1.0791 percentage points. Not repeated-run error bars.')
    # Intentionally different figures, never merge VQA or different datasets into one accuracy curve.
    for dataset in ['aokvqa','gqa']:
        rr=[r for r in read('aokvqa_gqa_pope.json') if r['split'].startswith(dataset+'_')]
        fig,ax=plt.subplots(figsize=(8,5));bars=ax.bar(labels,[100*r['f1'] for r in rr],color=colors);ax.bar_label(bars,fmt='%.2f%%',padding=4)
        ax.set(ylabel='POPE F1 (%)',ylim=(0,104),title=f'InternVL3-2B: {dataset.upper()} POPE (not open-ended VQA)')
        finish(fig,dataset+'_pope','Source: aokvqa_gqa_pope.json; author SEEM-derived fixed questions; 3,000 / strategy.\nSEEM annotations reused from authors; SEEM model not rerun.')
    objects=read('object_frequencies.json')
    for prompt in ['I1','I2']:
        fig,axes=plt.subplots(2,1,figsize=(11,7))
        for ax,anchor,title in [(axes[0],'','Top 10 by appearance in COCO ground truth'),(axes[1],'dining table','Top 10 by co-occurrence with dining table')]:
            rr=[r for r in objects if r['prompt_id']==prompt and r['anchor']==anchor][:10]
            ax.bar([r['object'] for r in rr],[r['hallucination_count'] for r in rr],color='#2563a0');ax.set(ylabel='Hallucinated mention count',title=title);ax.tick_params(axis='x',rotation=20)
        fig.suptitle(f'InternVL3-2B {prompt}: object hallucination frequency (Figure 2 extension)')
        finish(fig,'object_hallucinations_'+prompt,'Source: object_frequencies.json; 2,000 COCO captions; rank corpus = 40,504 validation images.\nRanked high to low, left to right. Counts preserve repeated mentions; not the InternVL training distribution.')
def documents():
    pope=read('coco_pope.json');ch=read('chair.json');vqa=read('vqa_supplementary.json');ext=read('aokvqa_gqa_pope.json');stab=read('prompt_stability_summary.json');cap=read('caption_metrics.json');hr=read('hra_hrc.json')
    pt=table([{**r,**{k:pct(r[k]) for k in ['accuracy','precision','recall','f1','yes_ratio']}} for r in pope],[('split','策略'),('n','題數'),('accuracy','Accuracy %'),('precision','Precision %'),('recall','Recall %'),('f1','F1 %'),('yes_ratio','Yes %')])
    ct=table([{**r,**{k:pct(r[k]) for k in ['CHAIRi','CHAIRs']}} for r in ch],[('prompt_id','提示'),('n','圖數'),('CHAIRi','CHAIRi % ↓'),('CHAIRs','CHAIRs % ↓'),('mentioned_instances','物件提及次數'),('hallucinated_instances','幻覺次數'),('average_length_whitespace','平均空白分詞長度')])
    vt=table([{**r,'score_percent':f'{r["score_percent"]:.4f}'} for r in vqa],[('dataset','資料集'),('scheme','方案'),('n_eligible','計分題數'),('n_excluded','排除'),('score_percent','本地分數 %')])
    et=table([{**r,**{k:pct(r[k]) for k in ['accuracy','precision','recall','f1','yes_ratio']}} for r in ext],[('split','資料集／策略'),('n','題數'),('accuracy','Accuracy %'),('precision','Precision %'),('recall','Recall %'),('f1','F1 %'),('yes_ratio','Yes %')])
    ft=table([{**r,**{k:pct(r[k]) for k in ['BLEU-1','BLEU-2','METEOR','ROUGE-L']}} for r in cap],[('prompt_id','提示'),('n','圖數'),('BLEU-1','BLEU-1 %'),('BLEU-2','BLEU-2 %'),('METEOR','METEOR %'),('ROUGE-L','ROUGE-L %')])
    abstract='大型視覺語言模型能同時處理圖片與文字，例如看圖回答問題或描述場景。不過，模型有時會說出圖片裡根本沒有的物件，這就是物件幻覺。它可能依賴文字裡常見的搭配，例如看到餐桌就猜測有杯子，而沒有充分根據圖片判斷。這篇論文先以圖片描述分析幻覺，發現評分會受到提示寫法、描述長度及物件解析規則影響，並觀察到常見物件及經常一起出現的物件較容易被誤認。作者因此提出 POPE，直接詢問某個物件是否存在，搭配三種不存在物件的選取方式，統計模型答對與誤判的情形。實驗顯示，當時部分模型傾向一直回答「有」，而 POPE 對提示變化較穩定，也能延伸到自動標註的資料。這項工作的貢獻是提供容易操作、可以比較模型物件存在判斷的評估方法；它並沒有解決所有幻覺，也不能取代完整的視覺理解測試。'
    assert 200<=len(abstract)<=350,len(abstract)
    write('PAPER_SUMMARY_zh-TW.md',f'''# POPE 論文與 InternVL3-2B 重現摘要

原文：[Evaluating Object Hallucination in Large Vision-Language Models](https://aclanthology.org/2023.emnlp-main.20/)，Li 等人，EMNLP 2023，頁 292–305。以下表號均指原論文；原作者沒有測試 InternVL3-2B。

## 一、論文摘要

{abstract}

## 二、本篇論文目的

**確認幻覺是否普遍。** 作者讓五種大型視覺語言模型看圖產生描述，檢查它們是否寫出不存在的物件，並與已發表的小型視覺語言模型結果對照。

**找出容易被誤認的物件。** 除了看總分，也統計哪些物件反覆出現在幻覺描述中，避免總分掩蓋具體行為。

**檢查出現頻率與共現關係。** 常見物件是否容易被猜成存在？看到餐桌時，模型是否更容易補出杯子或椅子？這是關聯性分析，不能據此證明訓練資料造成幻覺。

**檢查 CHAIR 的不足。** 讓模型「簡短描述」或「提供說明」可能產生不同長度的文字，進而改變評分；以字詞規則判斷物件也會出錯。

**建立容易使用且較穩定的評估。** POPE 不要求先產生整段描述，而是直接問物件是否存在，再把回答轉成可計分的 Yes/No。

**觀察與其他視覺任務的關係。** 作者將 POPE 與 VQA、圖片描述及描述中的物件提及對照，研究存在判斷是否與其他能力一致，而非假定分數越高就全面越強。

## 三、研究方法

### 資料、模型與 CHAIR

作者使用 MSCOCO 驗證資料及物件、人工描述標註。CHAIR 實驗取 2,000 張圖，限 COCO 的 80 類物件。五種 LVLM 為 mPLUG-Owl、LLaVA、MultiModal-GPT、MiniGPT-4、InstructBLIP；較小模型 OSCAR、VinVL、OFA、BLIP 的結果引用自 Dai 等人，並非全部重新執行。模型結構見原 Table 7。

CHAIRi 是「描述中幻覺物件次數／全部物件提及次數」；CHAIRs 是「至少含一次幻覺的描述數／全部描述數」，兩者越低越好。它以物件標註與描述標註為依據，再由同義詞及解析規則對應物件，因此不能直接視為人工語意判斷。

作者按物件出現頻率及共現頻率排序，繪出幻覺次數；另以 HRA@k、HRC@k 計算幻覺集中於前 k 名物件的比例，分別觀察全體及指定物件出現的圖片。

### POPE 與三種負樣本

例如圖片實際只有一隻狗，卻問「圖片中有貓嗎？」；回答 Yes 就是對不存在物件的誤判。這是教學例子，正式 COCO POPE 使用超過三種標註物件的圖，每圖六題，三題存在、三題不存在。500 圖各策略 3,000 題；正題依圖片中的標註物件，負題只從不存在的物件選。

| 策略 | 如何選不存在的物件 | 簡單例子（僅示意，非聲稱實際排序） |
| --- | --- | --- |
| Random | 隨機挑選 | 狗的圖中隨機問沒有出現的飛機 |
| Popular | 優先挑整體資料中常見的物件 | 問圖中沒有的人或椅子 |
| Adversarial | 優先挑與已出現物件經常共現的物件 | 狗常與沙發一起出現，因此問這張實際沒有沙發的圖是否有沙發 |

上述三組負例是不同選法，並非把任意較難問題都稱為 Adversarial。本專案直接使用作者固定題目，沒有依新模型重新出題。

### 指標與延伸驗證

以 Yes 為正類：TP 是實際存在且答 Yes；FP 是不存在卻答 Yes；FN 是存在卻答 No；TN 是不存在且答 No。Accuracy=(TP+TN)/N；Precision=TP/(TP+FP)；Recall=TP/(TP+FN)；F1=2PR/(P+R)；Yes Ratio=(TP+FP)/N。Recall 依官方程式定義為正類召回率，不能誤寫成 No 題答對率。高 Recall 若同時幾乎都答 Yes，仍可能有大量幻覺。

作者使用四種提示比較穩定性，並用 SEEM 分割結果取代人工物件標註，延伸到 COCO、A-OKVQA 與 GQA。VQA 開放回答則以 Appendix E 的 ChatGPT 輔助對應參考答案，再做計分；圖片描述另評 BLEU、METEOR、ROUGE-L。物件存在、開放問答與描述品質的分母及意義不同，不能合成一個準確率。

## 四、呈現的成果

### 原論文成果

| 原圖表／位置 | 作者實際呈現的發現與數值 | 解讀邊界 |
| --- | --- | --- |
| Table 1，頁 294 | I1 的 LLaVA CHAIRs 32.7%，InstructBLIP 3.7%；I2 分別 62.7%、3.4%。mPLUG-Owl I2 CHAIRi 30.2%、CHAIRs 76.8% | 幻覺普遍，但程度不同；描述提示與長度會影響結果 |
| Figure 2，頁 295 | 四模型中，常見及與餐桌共現的物件有較多幻覺提及 | 圖為描述次數，不是每類物件的條件錯誤率，也非因果證據 |
| Table 2，頁 296 | mPLUG-Owl HRA@10/@30 為 0.5455/0.7533；餐桌 HRC@10/@30 為 0.6608/0.8253 | 排序與分母處理會影響分析 |
| Table 3，頁 298 | InstructBLIP 三策略 F1 為 89.29/83.45/78.45%；MiniGPT-4 為 78.86/72.21/71.37%；其他三模型約 66–69% | 後三者多數題都答 Yes；不是 99% Recall 就代表表現好 |
| Table 4，頁 299 | LLaVA 四提示 POPE F1 平均 67.43%、標準差 0.78 百分點；CHAIRi 13.88%、3.22 百分點 | POPE 在這四提示中變化較小，不能推論對所有提示皆穩定 |
| Table 5，頁 299 | COCO SEEM 的 MiniGPT-4 Adversarial F1 62.70%，人工標註版為 71.37%；InstructBLIP SEEM 為 71.96% | 自動標註版與人工標註版有差距；不應視為同一測試集 |
| Table 6，頁 300 | A-OKVQA MiniGPT-4 POPE/VQA 為 72.47/38.69%，LLaVA 66.64/50.51%；GQA 分別 67.13/42.24%、66.56/47.60% | 較高 POPE 不保證較高 VQA；InstructBLIP 使用相關訓練資料的影響亦須注意 |
| Table 7，頁 304；Appendix A | 列視覺編碼器、對齊網路與 LLM，例如 LLaVA 為 ViT-L/14、Linear、LLaMA 13B，InstructBLIP 為 ViT-G/14、Q-Former、Vicuna 13B | 是模型及可訓練元件設定表，沒有單獨性能分數 |
| Table 8，頁 304；Appendix B | 按出現頻率排前 30 類，四模型涵蓋幻覺提及的 77.34/69.74/68.85/70.68% | 分別為 mPLUG、LLaVA、MultiModal、MiniGPT；不是前 30 類每類都高錯誤率 |
| Table 9，頁 304；Appendix B | 餐桌共現前 20 類涵蓋 82.82/73.79/75.16/80.00% 幻覺提及 | 同上四模型順序，屬集中程度 |
| Table 10，頁 305；Appendix C | LLaVA 椅子 HRC@10/@30=0.5206/0.8152，汽車=0.6870/0.9188 | 換共現錨點仍可觀察集中現象 |
| Table 11，頁 305；Appendix D | SEEM 固定題下 InstructBLIP A-OKVQA F1=87.20/79.53/73.75%，GQA=85.32/76.89/72.39%；另有 VQAv2 微調 BLIP 作監督式基線 | 每個資料集與策略分開比較，不是開放式 VQA 分數 |
| Appendix E，頁 304 | ChatGPT 將開放回答對應到候選參考答案，否則輸出 NAN | 未完整公布可重建的服務版本等設定，本專案沒有重現此評判 |
| Table 12，頁 305；Appendix F | InstructBLIP BLEU-1/2=59.5/45.2、METEOR=22.6、ROUGE-L=42.3；MiniGPT-4 METEOR=25.6 | 各指標排序並非完全相同，不能宣稱 POPE 決定所有描述品質 |

上述數值逐表核對使用者提供 PDF。原 Table 5 的 InstructBLIP Random「Truth」印為 89.27%，與 Table 3 的 89.29% 不一致；本專案歷史對照採 Table 3，沒有擅自更改原表。

### 本次 InternVL3-2B 實驗成果

我們沒有重新訓練或重跑五個原模型，而是換成固定版本 InternVL3-2B，在 RTX 3070 Ti 8GB 上以 BF16、無量化、貪婪解碼執行。FP16 曾出現語言 logits NaN，這是數值異常，不是顯示記憶體不足。COCO POPE 固定題與 Table 1 原 2,000 圖可核對；其他條件差異見[比較表](REPRODUCTION_COMPARISON_zh-TW.md)。

{pt}

![COCO POPE F1](figures/coco_pope_f1.png)

{ct}

POPE F1 高於原 Table 3 的 InstructBLIP，但 CHAIR I1 的 2.8634/4.9500% 比原 2.6/3.7% 差，I2 也比原 2.5/3.4% 差。這顯示問答存在判斷與自由描述有不同失誤，不能寫成 InternVL3-2B 全面較好。

Figure 2、HRA/HRC 與 Appendix B/C 已由自己的描述重算。I1 常見前 30 類涵蓋幻覺提及 71.1538%；排名依作者公布的 40,504 張 COCO 驗證標註，不代表 InternVL 訓練分布。四提示 POPE F1 平均 {pct(stab['POPE_F1_mean'])}%，標準差 {pct(stab['POPE_F1_population_std'])} 百分點；CHAIRi 平均 {pct(stab['CHAIRi_mean'])}%，標準差 {pct(stab['CHAIRi_population_std'])} 百分點。

Table 11 的作者 SEEM 題目共 18,000 題已完成；我們使用題檔，沒有重新執行 SEEM。A-OKVQA 三策略 F1=91.8002/85.6640/81.2570%，GQA=89.1874/87.0483/83.9605%。Appendix F 兩提示各 500 圖已計分，正文描述／POPE 一致性也完成；完整數據見[實驗結果](EXPERIMENT_RESULTS_zh-TW.md)。

VQA 11,216 筆原始回答已核對。A/B/C 本地補充評分如下，全部不是 Appendix E ChatGPT-assisted 分數，也不是人工驗證的語意準確率。

{vt}

短答案擷取後仍有 3,025 道有效題為模糊或未解析；解析成功卻不匹配的題也不能全部當成模型真錯。方案 D 只審查固定 240 題，17 個凍結後挑戰案例有 5 次錯誤接受；不採為正式準確率、不擴充全資料。COCO SEEM Table 5 原固定題無法核對、Table 6 原 VQA 評判未完成，均明列未重現。
''')
    write('README.md',f'''# POPE Reproduction and Extension with InternVL3-2B

[English](README_EN.md) · [教授版論文摘要](PAPER_SUMMARY_zh-TW.md) · [完整結果](EXPERIMENT_RESULTS_zh-TW.md) · [原文與本次差異](REPRODUCTION_COMPARISON_zh-TW.md) · [限制](LIMITATIONS_zh-TW.md)

VQA另附[方案D完整過程與A/B/C比較](VQA_D_PROCESS_AND_COMPARISON_zh-TW.md)：使用相同240題比較，保留失敗反例；D不是正式準確率。

## 專案簡介

這個專案想了解：AI 看圖片時，會不會認為圖片裡有實際不存在的物件？我們沿用 EMNLP 2023 [POPE 論文](https://aclanthology.org/2023.emnlp-main.20/)的評估方法，把受測模型換成 **OpenGVLab/InternVL3-2B**，保存實際回答並核對分數。

**這是新模型的評估與方法延伸，並非完整重新訓練或重跑原論文所有模型。** 原始結果留在本機；公開版提供研究程式、可追溯的彙整表、圖表及限制，不含影像、權重與未確認可再散布的逐題資料。

## 原論文摘要與研究目的

模型會利用文字裡常見的關係補出畫面，例如看到餐桌就回答有杯子。這類猜測有時合理，卻不符合圖片。作者先分析圖片描述中的物件幻覺，再提出 POPE：直接問「圖中有這個物件嗎？」。研究關心幻覺與物件常見程度、共現關係、提示寫法及其他視覺任務的關聯。四個重點及原 Table 1–12 的發現整理於[論文摘要](PAPER_SUMMARY_zh-TW.md)。

## 研究方法

COCO POPE 使用作者公開的固定 500 圖題集，每組 3,000 題、正負各半。Random 隨機選不存在物件；Popular 優先選常見但不存在的物件；Adversarial 優先選常與畫面物件共現、但本圖不存在的物件。以 Yes 為正類計算 Accuracy、Precision、Recall、F1 及 Yes Ratio。自由描述另以 CHAIRi/CHAIRs 評估，兩者越低越好。原始 Yes/No 解析規則照官方程式，沒有加上「只答 Yes/No」。

## 本次重現設定

| 項目 | 實際設定 |
| --- | --- |
| 模型 | InternVL3-2B，revision `899155015275a9b7338c7f4677e19c784e0e5a21` |
| 硬體 | NVIDIA RTX 3070 Ti，8GB；Windows |
| 環境 | Python 3.12.14、PyTorch 2.6.0+cu124、Transformers 4.52.1；專案獨立 `.venv` |
| 精度 | BF16、無量化；FP16 語言 logits NaN，**不是 OOM** |
| 生成 | batch 1、seed 0、greedy、beam 1、1024 token 上限、每題無歷史對話 |
| 影像 | 官方動態切圖，448px、max_num=12、thumbnail；最多 13 tiles |
| 來源 | POPE commit `08d957b917e5a378a2f99d35b6293c536a66298b` |

部分早期小上限輸出因已正常 EOS、圖片與提示一致而保留重用；不是每一列都重新以 1024 token 生成。截斷與重複描述保留在主要統計，並附敏感度分析。詳細條件見[重現說明](docs/REPRODUCIBILITY.md)。

## 實驗成果

{pt}

![COCO POPE F1](figures/coco_pope_f1.png)

CHAIR 原 2,000 圖：I1 CHAIRi **2.8634%**／CHAIRs **4.9500%**；I2 **4.7814%**／**18.0000%**。四提示 POPE F1 平均 **91.9855%**、母體標準差 **0.9340 百分點**。作者 SEEM 衍生固定題 A-OKVQA/GQA 各 9,000 題亦完成，見[完整結果與圖表](EXPERIMENT_RESULTS_zh-TW.md)。

## 與原論文的比較

![歷史模型對照](figures/coco_historical_comparison.png)

InternVL3-2B 的 COCO POPE 三組 F1 高於原文 InstructBLIP 的 89.29/83.45/78.45%；但 CHAIR 的 I1/I2 比該模型的原文數值差。模型年代、架構、訓練資料及原生影像／對話處理不同，這是歷史對照，**不是控制所有條件後的模型優劣實驗**。完整差異見[逐項比較表](REPRODUCTION_COMPARISON_zh-TW.md)。

## 未完成的實驗與限制

- **Table 5：未完整重現。** 原 COCO SEEM 固定題無法取得並核對；沒有自行重抽題集冒充原測試。
- **Table 6／Appendix E：未重現原評判。** VQA 11,216 筆回答已保存，但沒有原 ChatGPT-assisted grader。本地 A/B/C 是補充診斷：A-OKVQA 0/1.0204/9.7959%，GQA 0/0.7196/36.3798%。不能把這些分數解讀為原文 VQA 語意準確率。
- **方案 D 不可採為正式準確率。** 僅固定 240 題的探索性符號規則；凍結後 17 個挑戰案例有 5 次錯誤接受，沒有獨立人工真值，也未執行開源文字模型評判。
- HRA/HRC 排序資料、零幻覺分母及 Captioning 提示等存在差異；單一模型不能驗證原文跨模型相關性。詳見[研究限制](LIMITATIONS_zh-TW.md)。

## 專案安裝與執行

不需要為閱讀結果下載模型。先看 `public_data/`，可在現有 Python 3.12 的獨立 venv 中執行純 CPU 公開資料查核：

```powershell
python -m venv .venv
.venv\\Scripts\\python.exe release_tools/check_public.py
```

重新繪圖需依賴套件；重新推論另需自行依法取得資料、固定版本模型及第三方評分程式。完整安裝、資料路徑、逐項執行與離線重算方式見[重現說明](docs/REPRODUCIBILITY.md)。**本次整理沒有重跑 GPU 推論，也沒有改動原始正式結果。**

## 專案檔案結構

```text
README.md / README_EN.md           閱讀入口
PAPER_SUMMARY_zh-TW.md             原論文四重點＋本次結果
REPRODUCTION_COMPARISON_zh-TW.md   比較條件與未重現項目
EXPERIMENT_RESULTS_zh-TW.md        實測數據、分母、圖表
LIMITATIONS_zh-TW.md              研究限制
GITHUB_RELEASE_AUDIT.md           公開前查核
scripts/                         原研究程式（不含本機盤點工具）
research_extensions/             VQA 補充規則及評分程式
release_tools/                   唯讀核對、圖表、公開版檢查
public_data/                     彙整 JSON/CSV、來源雜湊與設定
figures/                         自動生成 PNG/SVG
docs/                            重現與授權說明
```

本機另保留 `results/`、`analysis/`、舊技術報告及備份，未納入公開 Git。公開版的彙整查核不等於讀者已取得原始逐題資料，資料取得限制詳見[資料政策](docs/DATA_POLICY.md)。

## 參考文獻與授權

- Li et al. (2023), [POPE，EMNLP，292–305](https://aclanthology.org/2023.emnlp-main.20/)，[官方程式](https://github.com/RUCAIBox/POPE)。
- [InternVL3-2B 固定版本](https://huggingface.co/OpenGVLab/InternVL3-2B/tree/899155015275a9b7338c7f4677e19c784e0e5a21)。
- [CHAIR](https://github.com/LisaAnne/Hallucination)、[COCO Caption 評分](https://github.com/tylin/coco-caption)、[A-OKVQA](https://github.com/allenai/aokvqa)、[GQA](https://cs.stanford.edu/people/dorarad/gqa/)。

本專案原創程式與整理文字採 [MIT](LICENSE)，第三方權利不在此授權範圍；見[來源與授權說明](THIRD_PARTY_NOTICES.md)。本專案不是原作者官方實作或模型官方評測。
''')
    write('EXPERIMENT_RESULTS_zh-TW.md',f'''# 實際完成的實驗結果

整理日期：2026-10-19。來源為本機保存的正式 SQLite、JSONL、JSON/CSV 與評分紀錄；新查核程式只讀取來源，不重跑模型。以下百分比由未四捨五入資料產生。原論文值只出現在[歷史比較](REPRODUCTION_COMPARISON_zh-TW.md)。

## COCO POPE：對應 Table 3

{pt}

| 策略 | TP | TN | FP | FN |
| --- | --- | --- | --- | --- |
| Random | 1397 | 1394 | 106 | 103 |
| Popular | 1397 | 1191 | 309 | 103 |
| Adversarial | 1397 | 1090 | 410 | 103 |

來源：[coco_pope.json](public_data/coco_pope.json)。各組 500 圖、3,000 題、正負各 1,500；原題與圖片雜湊已核對，Yes 為正類。負題難度增加時 FP 從 106 增至 410，正題相同，因此 Recall 保持 93.1333%。這是實際誤判增加，不是指標格式問題。

![COCO三策略](figures/coco_pope_f1.png)

## CHAIR：對應 Table 1

{ct}

I1：`Generate a short caption of the image.`；I2：`Provide a brief description of the given image.`。兩者同為作者原 2,000 圖，總 4,000 描述。物件解析沿用 CHAIR 與 Pattern singularize，GT 採作者公布的 segmentation＋caption 物件。CHAIRi 為全部幻覺次數除以全部提及次數，不是逐圖 CHAIRi 平均。

I1 有 1 筆達長度上限、2 筆重複標記；I2 有 3、4 筆。主要分數沒有刪除這些輸出。僅保留正常 EOS 的敏感度分數見 [chair.json](public_data/chair.json)。較長的 I2 描述有更多物件提及及幻覺，但這不是對「長文字必然造成幻覺」的因果測試。

![CHAIR](figures/chair_prompts.png)

## 物件頻率：Figure 2、Table 2、Appendix B/C

排名依作者公開之 40,504 張 COCO 驗證圖的物件集合，並列頻率按標準物件名稱排序。這與分析原模型實際 instruction 訓練語料的頻率不同。計數單位是提及次數，保留重複；圖中的高次數不能直接解讀為該物件較高條件錯誤率。

![I1物件頻率](figures/object_hallucinations_I1.png)

![I2物件頻率](figures/object_hallucinations_I2.png)

原論文沒有清楚說明「無幻覺圖片」除以零如何處理，因此三種分母全部保留。下表先列 HRA@10/20/30；完整 HRC 的餐桌、椅子、汽車與 I1/I2 共 24 列見 [hra_hrc.json](public_data/hra_hrc.json)。

{table([{**r,**{k:f'{r[k]:.6f}' for k in ['mean_conditional_nonzero_denominator','mean_all_images_zero_filled','pooled_instance_ratio']}} for r in hr if r['metric']=='HRA'],[('prompt_id','提示'),('k','k'),('images','全部圖'),('images_with_hallucinations','有幻覺圖'),('mean_conditional_nonzero_denominator','有幻覺圖平均'),('mean_all_images_zero_filled','無幻覺補0後平均'),('pooled_instance_ratio','全部提及合併比例')])}

這三欄都是 0–1 比例，不能任選一個當作精確重現原 Table 2。Appendix B 的累積比例屬合併提及比例：I1 前 30 類涵蓋 74/104=71.1538%；餐桌子集前 20 類涵蓋 10/17=58.8235%。見 [累積表](public_data/appendixB_cumulative.json)及[每物件次數](public_data/object_frequencies.json)。

## 提示穩定性：對應 Table 4

{table([{**r,'score':pct(r['f1'] if 'f1' in r else r['CHAIRi'])} for r in read('prompt_stability.json')],[('prompt_id','提示'),('n','題／圖數'),('score','F1或CHAIRi %（各自解讀）')])}

POPE P1–P4 F1 平均 {pct(stab['POPE_F1_mean'])}%，母體標準差 {pct(stab['POPE_F1_population_std'])} 百分點；CHAIR C1–C4 平均 {pct(stab['CHAIRi_mean'])}%，母體標準差 {pct(stab['CHAIRi_population_std'])} 百分點。這是四個固定提示間的離散程度，**不是多次隨機執行的信賴區間**，兩種指標也不是同一尺度的能力。C2 的 `of the image` 與 Table 1 I2 的 `of the given image` 不同。

![提示穩定性](figures/prompt_stability.png)

## A-OKVQA／GQA 的 POPE：對應 Table 11、Appendix D

{et}

來源：[aokvqa_gqa_pope.json](public_data/aokvqa_gqa_pope.json)。每資料集 500 圖、每策略 3,000 題，合計 18,000 題。使用作者已公布的 SEEM 衍生問題與標籤，沒有重新跑 SEEM。這些是物件存在二元問答，**不是 VQA 開放回答準確率**。

![A-OKVQA POPE](figures/aokvqa_pope.png)

![GQA POPE](figures/gqa_pope.png)

## 描述／POPE 一致性：正文 5.3

相同 500 圖的 I1/I2 描述配對三策略，共 18,000 個條件內唯一 image/object 配對（不是 18,000 次新模型推論）。例如 I1 Random 的 1,497 個 No 配對中，描述仍提及 11 個；描述提及的 737 個被問物件中，726 個答 Yes。I2 Random 對應為 18/1,497 與 944/962。完整六條件分母見[一致性表](public_data/caption_pope_consistency.json)。原文沒有完整公布所選策略與提示，因此不能當作同條件數值重現。

## Captioning：對應 Appendix F

{ft}

同原 500 張 COCO POPE 圖，PTB tokenizer、BLEU、METEOR、ROUGE 原評分工具已實際執行；這次整理核對已保存的來源與彙整，沒有再跑 Java 評分器。BLEU 為語料層級分數，不能拿逐圖 BLEU 平均代替。原文沒有完整明示 Table 12 提示，故 I1/I2 分開報告。I2 較長且分數較差，照實保留。[來源](public_data/caption_metrics.json)。

## VQA：補充診斷，不是完成 Table 6

方案D的完整流程、同一批240題A/B/C/D比較及凍結後失敗案例，見[專章](VQA_D_PROCESS_AND_COMPARISON_zh-TW.md)。

{vt}

A 是原始整句與標準答案精確比較；B 處理大小寫、標點、冠詞與少量數字格式；C 使用凍結的 v2.0 規則，僅根據問題與模型回答擷取短答案，**不讀正解選詞**，之後才計分。A-OKVQA 對十個人工答案使用 `min(相符數/3,1)` 共識分數；排除 25 道官方 difficult_direct_answer 題。GQA 單一答案做 Accuracy，保留全部題號前導零（5,461 筆）。未解析／模糊仍保留在有效題分母，不用只計可解析題提高分數。

| 資料集 | 原始回答 | 有效題 | 排除 | C模糊 | C未解析 | 合計未可靠解析 |
| --- | --- | --- | --- | --- | --- | --- |
| A-OKVQA | 515 | 490 | 25 | 126 | 141 | 267 |
| GQA | 10701 | 10701 | 0 | 961 | 1797 | 2758 |

0% 的主要原因包含完整句子對短答案的格式落差，例如應答短詞卻產生描述或解釋；但也存在真正答非所問、錯誤物件／屬性及解析器失誤。C 的 4,218 道「已解析但零分」仍混有語意改寫、參考答案及擷取問題，**不能直接稱為 4,218 道模型真錯**。本次不靠調整規則美化分數。

固定分層 240 題、seed 20261019 已保存於本機人工核對表，但沒有獨立人工標註。方案 D-symbolic-2.0 使用問題、完整回答與參考答案，與 C 分開；240 題中 81 equivalent、9 not_equivalent、12 ambiguous、138 unassessable，其中含 7 道不計分題。其 17 個凍結後挑戰案例有 5 次錯誤接受，因此**不能採為正式準確率**。沒有開源文字模型或付費 API 的語意評判結果。

目前只有一個受測模型，VQA 也只是原 POPE 圖片對應的官方題目延伸子集，不能宣稱完成跨模型 POPE/VQA 相關性驗證。公開版只提供彙整與規則；含問題／標準答案的逐題 ZIP 保留本機，避免未確認的再散布。

## 本次重新查核範圍

[local_validation.json](public_data/local_validation.json)記錄逐筆及彙整查核範圍。[private_source_hashes.json](public_data/private_source_hashes.json)為本機正式來源檔的相對路徑與 SHA-256，供取得原始成果的人核對；公開倉庫本身沒有這些原始檔。各表重用相同結果，63182 筆保存列數不能稱為 63182 次獨立新推論。數值、程式碼及先前正式結果均保留。
''')
    write('REPRODUCTION_COMPARISON_zh-TW.md','''# 原論文與本次實驗的差異

這裡的「完成」是方法套用與實際資料查核的完成，並非宣稱所有原模型、訓練資料與硬體完全一致。原文數值來源為 [POPE 論文](https://aclanthology.org/2023.emnlp-main.20/)，本次來源為 [public_data](public_data/README.md)。

| 項目 | 原論文模型與方法 | 本次做法／實測結果 | 原文結果例子 | 可比性與限制 |
| --- | --- | --- | --- | --- |
| 受測模型 | 五種 2023 LVLM；另引用小模型基線 | 固定 InternVL3-2B，未重新訓練 | Table 7 有編碼器／對齊網路／LLM差異 | 不同年代、架構、訓練資料，不能單憑分數宣稱全面優於原模型 |
| Table 3 COCO POPE | 500圖、三組各3000題；原生模型介面 | 作者原固定題；F1 93.0403/87.1491/84.4875% | InstructBLIP 89.29/83.45/78.45% | 題目、標籤及官方回答解析可對照；模型內部聊天模板、影像前處理與訓練不同 |
| Table 1 CHAIR | 原2000圖，I1/I2；80類 | 作者相同圖ID；I1 i/s=2.8634/4.9500%，I2=4.7814/18.0000% | InstructBLIP I1=2.6/3.7%，I2=2.5/3.4% | 新模型分數較差須照實呈現；描述長度、同義詞規則及標註有限 |
| Figure 2 | 原四模型的幻覺提及次數與頻率／共現排序 | InternVL I1/I2，各2000圖 | 圖示常見物件有較多幻覺次數 | 我們排序採作者公開COCO驗證集GT，不是原訓練語料，也非InternVL訓練分布 |
| Table 2；Tables 8–10 | HRA/HRC與累積集中度 | 三種分母並列；餐桌／椅子／汽車，I1/I2皆完成 | LLaVA餐桌HRC@30=0.8595 | 原文未明示零幻覺圖分母處理，不能指定其中一欄為完全重現 |
| Table 4 | LLaVA四提示；POPE均67.43%、std0.78pp | 新模型四提示POPE均91.9855%、std0.9340pp；CHAIRi均4.1010%、std1.0791pp | CHAIRi均13.88%、std3.22pp | 提示可對照；不是重複訓練或seed的誤差範圍，指標不同不能直接等同 |
| Table 5 COCO SEEM | SEEM重新標註後建立COCO問題 | **未完成**；原固定題與原抽樣無法取得並核對 | MiniGPT-4 Adversarial F1=62.70% | 沒有自行替代題集，沒有額外SEEM實驗分數 |
| Table 11／Appendix D | 作者SEEM之A-OKVQA/GQA固定題；另BLIP監督基線 | 原題共18000問答；AOK F1 91.8002/85.6640/81.2570%，GQA 89.1874/87.0483/83.9605% | InstructBLIP AOK 87.20/79.53/73.75%；GQA 85.32/76.89/72.39% | 固定題可對照；沒有重新跑SEEM，也沒有重跑BLIP基線 |
| Table 6／Appendix E | ChatGPT將開放回答對應候選答案，再做VQA評分 | **原方法未完成**。原POPE圖片對應子集11216筆回答 | InstructBLIP AOK VQA59.68%、GQA62.12% | 原精確抽樣與grader服務設定無法重建；不能拿本地分數直接對比 |
| VQA A/B/C | 原文無此三方案 | AOK=0/1.0204/9.7959%；GQA=0/0.7196/36.3798% | 無對應原文值 | 本地格式診斷；有3025有效題模糊／未解析，沒有人工語意準確率 |
| VQA D | 原文ChatGPT輔助，與本地D不同 | 固定240題有限符號語意規則；凍結後挑戰5次錯誤接受 | 無對應原文值 | **未充分驗證且可靠性檢查失敗**；不作正式準確率、不擴展全資料，非開源LLM評判 |
| 正文5.3一致性 | InstructBLIP/MiniGPT-4 的No與caption提及配對 | I1/I2×三策略，18000配對 | InstructBLIP 1303個No中0個被提及；提及664中664答Yes | 原文未完整指定策略／提示；我們完整分母公開，屬延伸 |
| Table 12／Appendix F | 三模型COCO描述品質 | 原500圖I1/I2；I1 BLEU1/2=61.3197/44.6260%，METEOR28.8915%、ROUGE49.4507% | InstructBLIP59.5/45.2/22.6/42.3% | 原文未明示完整caption提示，I2結果另列；不能因某些指標高就概括全面較佳 |
| 數值精度 | 原文未完整提供可對應本機的精度設定 | 正式BF16無量化；FP16語言logits NaN | 無同硬體對照 | 未發生OOM；沒有把NaN說成記憶體不足；也未證明BF16與FP16輸出等價 |

`pp` 為百分點。所有本次詳細指標、分母與原始來源雜湊見[實驗結果](EXPERIMENT_RESULTS_zh-TW.md)與[重現說明](docs/REPRODUCIBILITY.md)。本專案沒有測試訓練資料重疊／汙染是否造成分數差異；無法排除也不將其當成已證實原因。
''')
    write('LIMITATIONS_zh-TW.md','''# 研究限制與尚未完成項目

## 研究定位

本專案依 POPE 方法評估一個新模型，沒有重訓原模型，也沒有重跑原文五種 LVLM 或全部小型基線。模型年代、原生對話模板、影像處理、訓練來源與資料重疊未受控制。COCO固定題有較好的題目可比性，但不能因此宣稱模型全面較強或公平比較所有架構。

## 尚未重現

- 原 Table 5 COCO SEEM 的固定題與抽樣無法取得並核對，未執行；未自行產生不同題集補分數。
- 原 Table 6／Appendix E 的 ChatGPT-assisted VQA 評判未完成，原精確抽樣與服務版本等亦未能確認。
- Table 7 是原模型設定整理，並非本專案完成原模型訓練；跨模型相關性、因果消融與完整視覺幻覺均未完成。

## 測量限制

POPE測的是特定物件是否存在，無法涵蓋計數、顏色、位置與所有關係。Yes/No解析沿用原官方規則：只看句點前文字及特定否定詞，對複雜回答仍可能誤判。固定題不是新的隱藏測試集。

CHAIR依COCO 80類、同義詞與字詞規則，不是看圖的人工裁決。例如 flower beds 可能被解析成 bed；這是已保存的真實解析風險，不能把全部規則失誤當成模型幻覺。長描述、重複與截斷保留在主要結果，EOS-only敏感度另列，不因較差而刪除。

HRA/HRC原文未明示無幻覺圖片的除零處理；三種分母均公開。排序使用作者公布的40,504張COCO驗證GT，不是原模型訓練語料，亦不代表InternVL訓練分布。頻率高的物件有更多幻覺提及，不等於其條件錯誤率更高，更不能證明訓練資料頻率造成幻覺。

四提示的母體標準差只描述四個固定提示間差異；沒有估計模型重複執行變異、信賴區間或統計顯著性。各表共用相同圖與回答，不是互相獨立的重複實驗。

## VQA補充評分限制

A/B/C固定本地規則分開報告。A原始0%不是語意正確率0%；C提高也不證明提高的每題都被語意正確評分。擷取器不讀參考答案，但不能充分處理序數、同義詞、修飾語、物件角色、空間關係與多句說明。3,025道有效題模糊／未解析仍留在分母；其他零分也混合模型、解析、格式及參考標註差異。

固定分層240題不是簡單隨機樣本；沒有獨立人工標註，不能估計人工一致率或把樣本分數直接外推到11,216題。D-symbolic-2.0是同一研究流程撰寫的有限語意規則，不是訓練過的語言模型；信心標記未校準。17個凍結後挑戰中有5次錯誤接受，涉及錯主體、二元回答與動作矛盾、所有者、食物細節及否定範圍。故不採D正式準確率，也不擴展全資料。開源文字模型只做過可行性評估／取得嘗試，沒有成功執行的判分結果。全程沒有付費API。

目前只有InternVL3-2B一個模型，VQA採POPE圖片所對應的延伸子集；任何圖層級探索關聯都不是原文跨模型相關性驗證。

## 推論與公開重現限制

FP16曾出現logits NaN，正式使用BF16無量化，未出現OOM。未證明不同精度、驅動或GPU可逐token一致；固定seed也不保證跨平台位元相同。部分已正常EOS的早期小上限回答按相同圖片／提示條件重用，保留來源，不能聲稱全部重新生成。

公開版不含第三方問題全集、參考答案、圖片、權重或未確認再散布權利的CHAIR原碼。讀者可查聚合值及程式，完整逐題重現仍須從來源取得資料並遵守各自條款；不得把公開版的聚合查核稱為公開原始資料的獨立再驗證。原始結果與ChatGPT審查ZIP仍完整保留於研究者本機。

適合與教授討論的結論是：新模型在固定POPE存在問答有較少誤判，但自由描述仍有幻覺；評估方法、提示與回答格式會明顯影響可報告的數字。這支持分開報告各項任務與測量限制，不支持「新模型已解決幻覺」的結論。
''')
    write('README_EN.md',f'''# POPE Reproduction and Extension with InternVL3-2B

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
''')
    write('public_data/README.md','''# 公開彙整資料欄位

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
''')
def main():
    assert read('local_validation.json')['passed']
    p=paper_data();charts(p);documents()
    for f in D.glob('*.json'):
        if f.name=='RELEASE_MANIFEST.json':continue
        data=json.loads(f.read_text(encoding='utf-8'))
        if isinstance(data,list) and data and isinstance(data[0],dict):
            keys=list(dict.fromkeys(k for r in data for k in r))
            with f.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in data)
    print('Reader documents and figures generated from validated aggregates.')
if __name__=='__main__':main()
