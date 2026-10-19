"""D symbolic semantic audit v1.0: independent of the v2 answer extractor.

Inputs: question, complete student answer, complete references. No image, C
status/answer/score, metadata, prior AI notes or pretrained model is read.
Finite transparent semantic equivalences + role-preserving propositions.
Unsupported semantic comparisons abstain. Confidence is NOT calibrated.
"""
import re,unicodedata
VERSION='D-symbolic-2.0'
COLORS=set('red blue green yellow orange black white brown gray grey pink purple beige tan silver gold khaki cream violet maroon turquoise cyan'.split())
SYNONYMS=[['sofa','couch'],['donut','doughnut'],['bicycle','bike'],['motorcycle','motorbike'],['television','tv'],['airplane','aeroplane','plane'],['cell phone','cellphone','mobile phone'],['gray','grey'],['refrigerator','fridge'],['trash can','garbage can','rubbish bin','trash bin'],['cabinet','cupboard'],['jacket','coat'],['sneakers','trainers'],['child','kid'],['children','kids'],['photograph','photo','picture']]
ALIASES={x:g[0] for g in SYNONYMS for x in g}
SMALL='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
TENS={'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90}
ORDINALS={'first':'one','second':'two','third':'three','fourth':'four','fifth':'five','sixth':'six','seventh':'seven','eighth':'eight','ninth':'nine','tenth':'ten','eleventh':'eleven','twelfth':'twelve','thirteenth':'thirteen','fourteenth':'fourteen','fifteenth':'fifteen','sixteenth':'sixteen','seventeenth':'seventeen','eighteenth':'eighteen','nineteenth':'nineteen','twentieth':'twenty','thirtieth':'thirty','fortieth':'forty','fiftieth':'fifty','sixtieth':'sixty','seventieth':'seventy','eightieth':'eighty','ninetieth':'ninety'}
PLURALS={'people':'person','men':'man','women':'woman','children':'child','kids':'child','mice':'mouse','teeth':'tooth','feet':'foot','knives':'knife','leaves':'leaf','pizzas':'pizza','lamps':'lamp','lights':'light','cats':'cat','dogs':'dog','horses':'horse','hats':'hat','shoes':'shoe','birds':'bird','cows':'cow','zebras':'zebra','elephants':'elephant','cars':'car','buses':'bus','trains':'train','chairs':'chair','tables':'table','vehicles':'vehicle','animals':'animal','vests':'vest','workers':'worker','bottles':'bottle','gloves':'glove','remotes':'remote','keyboards':'keyboard'}
OBJECTS=set('dog cat horse elephant zebra giraffe cow sheep goat bear bird duck chicken pig rabbit fish person man woman child helmet cigarette cigar apple orange banana carrot onion pizza hamburger sandwich donut bread cake cup bowl plate bottle glass chair table sofa desk cabinet television refrigerator freezer bed bench car bus train truck motorcycle bicycle boat airplane hat jacket shirt sweater pants skirt shoe umbrella racket kite skateboard laptop keyboard phone clock book fork knife spoon scissors brush'.split())
NEG=re.compile(r"\b(no|not|never|cannot|can't|isn't|aren't|doesn't|don't|wasn't|weren't|without)\b",re.I)
UNCERTAIN=re.compile(r"\b(cannot|can't|unable to|unclear|uncertain|not sure|not enough|insufficient|could be|might be|may be|hard to tell|difficult to determine|cannot be determined)\b",re.I)
META=re.compile(r'<\|[^>]+\|>|\[/?INST\]|\ufffd')
NUMPAT=r'(?:\d+(?:\.\d+)?(?:st|nd|rd|th)?|'+ '|'.join(sorted(SMALL+list(TENS)+list(ORDINALS),key=len,reverse=True))+r')(?:[- ](?:'+ '|'.join(SMALL[1:10])+r'))?'
RELATIONS={'left':'right','right':'left','above':'below','below':'above','behind':'in front of','in front of':'behind','on':'under','under':'on','inside':'outside','outside':'inside'}
STOP=set('what which who whom whose where when why how is are was were do does did the a an this that these those in on of to from at by with as it they image picture photo shown visible called look looks think there any either both or and color colour side relative'.split())
def number(x):
    x=x.lower().replace('-',' ').strip()
    x=re.sub(r'(?<=\d)(?:st|nd|rd|th)\b','',x)
    x=' '.join(ORDINALS.get(w,w) for w in x.split());ww=x.split()
    if re.fullmatch(r'\d+(?:\.\d+)?',x):return str(int(float(x))) if float(x).is_integer() else x
    if x in SMALL:return str(SMALL.index(x))
    if len(ww)==1 and ww[0] in TENS:return str(TENS[ww[0]])
    if len(ww)==2 and ww[0] in TENS and ww[1] in SMALL[1:10]:return str(TENS[ww[0]]+SMALL.index(ww[1]))
    return x
def norm(s,numeric=False):
    s=unicodedata.normalize('NFKC',s).lower().replace('’',"'")
    s=re.sub(r"\b(\w+)'s\b",r'\1',s);s=re.sub(r'[^\w\s.\-]',' ',s)
    if numeric:s=re.sub(r'\b'+NUMPAT+r'\b',lambda m:number(m[0]),s)
    for value in sorted(ALIASES,key=len,reverse=True):s=re.sub(r'\b'+re.escape(value)+r'\b',ALIASES[value],s)
    return ' '.join(PLURALS.get(w,w) for w in s.strip(' .').split() if w not in {'a','an','the'})
def words(s):return set(norm(s).split())-STOP
def sentences(s):return [x.strip().strip(' .!?') for x in re.split(r'(?<!\d)[.!?]\s+|(?<=\d)[!?]\s+|\n+',s.strip()) if x.strip()]
def qkind(q):
    q=q.lower()
    if re.search(r'how many|how much|which number|what number|which birthday|birthday.*celebrat',q):return 'number_ordinal'
    choice=' or ' in q and not re.search(r'\b(there|either|any|both)\b',q)
    if re.match(r'^(is|are|was|were|do|does|did|has|have|can|could)\b',q) and not choice:return 'binary'
    if re.search(r'\bcolou?r\b',q) or (' or ' in q and len(set(q.split())&COLORS)>=2):return 'color'
    if re.search(r'which side|what side|where|left or|right or|relative to',q):return 'spatial_location'
    if re.match(r'^(why|how)\b',q):return 'causal_reasoning'
    if q.startswith('who'):return 'person'
    return 'object_attribute'
def trim(s):
    s=s.strip(' .!?;:')
    s=re.split(r',\s*(?:which|who|where|suggesting|indicating)|\s+(?:because|suggesting|indicating)\b',s,1,flags=re.I)[0]
    s=re.sub(r'\s+(?:in (?:the|this) (?:image|photo|picture|scene)|shown here|visible here)$','',s,flags=re.I)
    return s.strip(' .!?;:')
def target_subject(q):
    # Preserve the main target before a relation to another named object.
    text=q.lower().strip(' ?')
    for pattern in [r'^(?:what|which) colou?r (?:is|are) (.+)',r'^what is the colou?r of (.+)',r'^where (?:is|are) (.+?) relative to ',r'^(?:is|are) (.+?) (?:to|on) (?:the )?(?:left|right)\b',r'^(?:on )?(?:which|what) side of (?:the )?(?:image|picture|photo) (?:is|are) (.+)']:
        m=re.search(pattern,text)
        if m:return re.split(r'\b(?:that|which|relative to)\b',m[1])[0]
    return text
def judge(question,prediction,references):
    """Pure deterministic D: no dependency on the old extractor or C scores."""
    q=question.strip();raw=prediction.strip();kind=qkind(q);ss=sentences(raw)
    trace=[];proof=[]
    def result(verdict,reason,confidence='low',indices=None,answer='',category='unsupported_semantics'):
        return {'verdict':verdict,'compatible_reference_indices':indices or [],'confidence':confidence,'reason':reason,
          'answer_proposition':answer,'question_kind':kind,'semantic_features':sorted(set(proof)),'trace':trace,'audit_category':category}
    if not ss or META.search(raw):return result('unassessable','空輸出或特殊字元，無法建立可靠語意命題。',category='malformed')
    first=ss[0];target=words(target_subject(q));generic=not target or target<=set('animal object thing furniture food person man woman child'.split())
    # A data instruction never controls this code; ignore known injected command
    # as a source of truth, keeping the actual answer for comparison.
    if re.search(r'ignore (?:previous|all)|say match',raw,re.I):
        proof.append('instruction_in_answer_ignored')
        cleaned=[s for s in ss if not re.search(r'ignore (?:previous|all)|say match',s,re.I)]
        if not cleaned:return result('unassessable','回答含指令而無可辨認的問題答案。',category='non_answer')
        ss=cleaned;first=ss[0]
    explicit=[]
    for s in ss:
        m=re.match(r'^(yes|no)\b',s,re.I)
        if m:explicit.append(m[1].lower())
        explicit.extend(x.lower() for x in re.findall(r'\banswer is (yes|no)\b',s,re.I))
    if len(set(explicit))>1:return result('ambiguous','同一回答明示互相矛盾的 Yes 與 No。',category='contradictory_answers')
    if re.search(r'\b(?:actually|correction|on second thought|rather than|instead of)\b',raw,re.I):return result('ambiguous','回答含後續更正或替代敘述，需要完整語意裁定；不保留有利的早期答案。',category='correction_or_alternative')
    if UNCERTAIN.search(first):return result('ambiguous','首個回答命題對目標明示無法確定或提供猜測。',category='uncertain_target')
    candidate=None;confidence='high';category='literal_equivalence'
    if kind=='binary':
        if explicit:
            candidate=explicit[0];tail=re.sub(r'^(?:yes|no)[,\s]*','',first,flags=re.I)
            if candidate=='yes' and NEG.search(tail):return result('ambiguous','肯定詞後接否定子句，有限規則無法可靠排除語意矛盾。',category='contradictory_answers')
            if candidate=='no' and not NEG.search(tail) and re.search(r'\b(is|are)\s+',tail,re.I):
                # Reject inconsistent No + affirmative repetition of the same
                # queried predicate; a different object can support a valid No.
                predicate=re.split(r'\b(?:is|are|does|do)\b',q.lower(),maxsplit=1)[-1]
                pp=words(predicate);tt=words(tail)
                if pp and pp<=tt:return result('ambiguous','No 後完整肯定同一問題命題，回答自相矛盾。',category='contradictory_answers')
            if re.search(r'\b(but|however|although)\b',first,re.I):return result('ambiguous','二元回答有條件或轉折，未證明其滿足整個問題。',category='qualified_binary')
            for later in ss[1:]:
                if candidate=='yes' and re.match(r'^there (?:is|are) no\b',later,re.I):return result('ambiguous','Yes 後出現不存在的反向命題，有限規則不忽略矛盾。',category='contradictory_answers')
                if candidate=='no' and re.match(r'^there (?:is|are) (?:a|an|one|two|three)\b',later,re.I):return result('ambiguous','No 後出現存在的反向命題，需要確認同一目標。',category='contradictory_answers')
            trace.append('explicit_binary_answer')
        elif re.match(r'^there (?:is|are) no\b',first,re.I):candidate='no';trace.append('explicit_absence_statement')
        else:return result('unassessable','二元命題未給明示極性；不憑個別物件詞判定。',category='binary_scope_unresolved')
    elif kind=='number_ordinal':
        patterns=[r'^(?:there (?:is|are)|i (?:see|count)|we (?:see|count))\s+('+NUMPAT+r')\b',
          r'^(?:the )?number\s+('+NUMPAT+r')\b',r'^('+NUMPAT+r')\s+\w+',r'\b('+NUMPAT+r')\s+(?:birthday|years old)\b']
        found=[]
        for pattern in patterns:
            m=re.search(pattern,first,re.I)
            if m:found.append(number(m[1]));break
        if not found and re.fullmatch(NUMPAT,first,re.I):found=[number(first)]
        if re.search(r'\b(?:or|between|approximately|about|around)\b',first,re.I):return result('ambiguous','數量主句有替代值、範圍或近似數，未擇有利答案。',category='quantity_uncertainty')
        if not found:return result('unassessable','未找到與數量問題綁定的量詞；不搜尋後文任意數字。',category='quantity_scope_unresolved')
        count_target=re.search(r'how many\s+(?:(?:different|total|visible|small|large)\s+)?(\w+)',q,re.I)
        if count_target:
            quantity_noun=re.search(r'\b'+NUMPAT+r'\s+(\w+)',first,re.I)
            if quantity_noun:
                noun=lambda x:norm(x).rstrip('s')
                if noun(count_target[1])!=noun(quantity_noun[1]):return result('unassessable','數字後的名詞與所問計數目標不同，不能把任意相同數字判成相容。',category='quantity_target_unresolved')
        candidate=found[0];trace.append('target_quantity_from_answer_clause');proof.append('number_or_ordinal_canonicalization')
        for later in ss[1:]:
            m=re.match(r'^(?:there (?:is|are)|i see)\s+('+NUMPAT+r')\b',later,re.I)
            if m and number(m[1])!=candidate:return result('ambiguous','後續再次宣告不同總數，與主回答衝突。',category='contradictory_quantity')
    elif kind=='color':
        # A garment color may be expressed as the modifier of the correctly
        # named object, even when its wearer is the sentence subject.
        wearable=re.match(r'^.+?\s+(?:is|are)\s+wearing\s+(.+)$',first,re.I)
        garment_color=None
        if wearable:
            ww=norm(wearable[1]).split();cc=[w for w in ww if w in COLORS]
            if len(cc)==1 and words(wearable[1])&target:garment_color=cc[0]
        # Extract the outer predicate, never the color of a subordinate object.
        m=re.match(r'^(.+?)\s+(?:is|are|was|were)\s+(?:described as |(?:a )?colou?r(?:ed)? )?(.+)$',first,re.I)
        if not m:return result('unassessable','沒有可綁定目標主語的顏色命題。',category='color_scope_unresolved')
        subject,tail=m.groups();subjectwords=words(subject)
        pronoun=norm(subject) in {'it','they','this','that'}
        if garment_color:
            tail=garment_color;subject=target_subject(q);trace.append('question_bound_worn_object_color')
        elif not generic and not pronoun and not target&subjectwords:
            if len(ss)>1:return result('unassessable','首句主語與目標不符；後文可能另有回答，有限規則不擅自裁定。',category='wrong_target_or_later_answer')
            return result('unassessable','未證明主句主語或受詞與所問目標相連；句型不符本身不是答非所問的證據。',category='target_link_unresolved')
        if NEG.search(tail):
            alt=re.search(r'\bnot\s+\w+[,;]\s*(?:but\s+)?(.+)',tail,re.I)
            if alt:tail=alt[1]
            else:return result('unassessable','顏色命題含否定，無唯一肯定色彩。',category='negation_scope')
        if re.search(r'\bor\b',tail,re.I):return result('ambiguous','目標顏色包含未決替代答案。',category='alternative_answers')
        normalized=norm(trim(tail));ww=normalized.split()
        colorwords=[w for w in ww if w in COLORS]
        if not colorwords or any(w not in COLORS|{'dark','light','and','color','colour','colored','very'} for w in ww):return result('unassessable','顏色片語混有未解析的物件或關係，棄答。',category='color_scope_unresolved')
        candidate=' '.join(ww);trace.append('main_subject_color_predicate')
        for later in ss[1:]:
            mm=re.match(r'^(it|they|'+re.escape(subject)+r')\s+(?:is|are)\s+(.+)',later,re.I)
            if mm and set(norm(mm[2]).split())<=COLORS|{'dark','light','and'} and norm(mm[2])!=candidate:return result('ambiguous','後句對同一主語／代名詞宣告不同顏色。',category='contradictory_answers')
    elif kind=='spatial_location':
        relpat=r'(?:to |on )?(?:the )?(left|right)(?: side)?(?: of (.+))?|in front of (.+)|behind (.+)|above (.+)|below (.+)|under (.+)|on (.+)|in (.+)'
        m=re.match(r'^(.+?)\s+(?:is|are|was|were)\s+(?:located )?('+relpat+r')',first,re.I)
        if m:
            subject=m[1];tail=m[2];sm=words(subject);wanted=words(target_subject(q))
            direction=re.search(r'\b(left|right)\b',tail,re.I) or re.search(r'\b(in front of|behind|above|below|under|on|in)\b',tail,re.I)
            candidate=direction[1].lower() if direction else None
            if candidate in {'left','right'}:
                other=re.split(r'\bof\b',tail,1,flags=re.I)[-1];other=re.split(r'\bthat\b',other,1,flags=re.I)[0]
                if wanted and not wanted&sm and wanted&words(other):candidate=RELATIONS[candidate];proof.append('inverse_relation_roles');confidence='medium'
                trace.append('outer_subject_spatial_relation')
            else:
                candidate=norm(trim(tail));trace.append('outer_subject_location_phrase');confidence='medium'
        else:return result('unassessable','無法確定位置關係的主體與參照物。',category='relation_scope_unresolved')
    elif kind=='causal_reasoning':
        causal=[]
        for s in ss:
            mm=re.search(r'\b(?:because|in order to|to prevent|to ensure|to increase|helps? (?:them|people|workers)|for)\b(.+)',s,re.I)
            if mm:causal.append(s[mm.start():])
        if not causal:return result('unassessable','開放式原因未有受控語意映射可證；不在長文搜正解詞。',category='reasoning_requires_semantics')
        cause=' '.join(causal).lower();causekeys=set()
        if re.search(r'prevent.*(?:harm|accident|injur)|avoid.*(?:harm|accident|injur)|safety|protect',cause):causekeys.add('safety')
        if re.search(r'visible|visibility|(?:people|worker).*see|see.*(?:people|worker)',cause):causekeys.add('visibility')
        if re.search(r'eat|meal|snack|dining|enjoy.*food',cause):causekeys.add('eating')
        if re.search(r'carry.*(?:people|passenger|goods)|transport',cause):causekeys.add('transportation')
        if not causekeys:return result('unassessable','原因子句超出固定四類語意映射，不能判定同義。',answer=cause,category='reasoning_requires_semantics')
        candidate='|'.join(sorted(causekeys));proof.append('finite_causal_gloss');confidence='medium';trace.append('anchored_causal_clause')
    else:
        if re.search(r'\banimal\b',q,re.I):
            existential=re.match(r'^(?:below .+?,\s*)?there\s+(?:is|are)\s+(?:'+NUMPAT+r'\s+)?(?:a\s+herd\s+of\s+)?(.+?)(?:\s+(?:in front of|behind|under|above|below|in|on)|$)',first,re.I)
            if existential:
                value=norm(existential[1]);head=value.split()[0] if value else ''
                if head in OBJECTS:candidate=head;trace.append('animal_existential_object');confidence='medium'
        # Strong inverse subject questions: what is on/under/near X?
        inverse=re.match(r'^(?:what|which)\s+(?:is|are)\s+(?:likely )?(on|under|above|below|behind|beside|near|in front of|next to)\b',q,re.I)
        if inverse and candidate is None:
            mm=re.match(r'^(.+?)\s+(?:is|are)\s+(?:likely )?'+re.escape(inverse[1])+r'\b',first,re.I)
            if mm:candidate=mm[1];trace.append('inverse_relation_subject')
        if candidate is None and kind=='person':
            mm=re.match(r'^(.+?)\s+(?:(?:is|are)\s+)?(?:wearing|holding|sitting|standing|riding|sits|stands|rides)\b',first,re.I)
            if mm:candidate=mm[1];trace.append('requested_person_subject')
        if candidate is None:
            # Possession with a question-bound location, e.g. "X has Y in his
            # mouth". Do not select Y from an unrelated possession statement.
            if re.search(r'^what is in\b',q,re.I):
                mm=re.match(r'^.+?\s+has\s+(.+?)\s+in\s+.+$',first,re.I)
                if mm:candidate=mm[1];trace.append('question_bound_possession_location')
        if candidate is None:
            activities=[('wear','wearing'),('hold','holding'),('ride','riding'),('eat','eating'),('drink','drinking'),('play','playing'),('use','using')]
            for root,verb in activities:
                if re.search(r'\b'+root+r'(?:s)?\b|\b'+verb+r'\b',q,re.I):
                    mm=re.match(r'^.+?\s+(?:(?:is|are)\s+)?(?:'+verb+'|'+root+r's?)\s+(.+)',first,re.I)
                    if mm:candidate=mm[1];trace.append('question_bound_activity_object');break
        if candidate is None:
            mm=re.match(r'^(.+?)\s+(?:contains?|depicts?|shows?)\s+(.+)',first,re.I)
            if mm:candidate=mm[2];trace.append('containment_or_depiction_object')
        if candidate is None:
            mm=re.match(r'^(.+?)\s+(?:is|are|was|were|appears to be|seems to be)\s+(.+)',first,re.I)
            if mm:
                subject,tail=mm.groups()
                if re.search(r'\b(?:is|are|was|were)\b',tail):return result('unassessable','複合命題有多個主體／copula，有限規則不擅選子句。',category='relation_scope_unresolved')
                if re.search(r'\banimal\b',q,re.I) and not words(subject)&set('animal cat dog horse elephant zebra giraffe cow sheep goat bear bird duck chicken pig rabbit fish'.split()):
                    if re.match(r'^the weather\b',first,re.I) and len(ss)==1:return result('not_equivalent','唯一命題只描述天氣，沒有回答所問動物。','medium',category='wrong_target')
                    return result('unassessable','未解析出動物目標與命題的連結，不能因語法失敗宣稱答非所問。',category='target_link_unresolved')
                candidate=re.sub(r'^(?:likely |probably |made (?:out )?of |described as |called |located )','',tail,flags=re.I);trace.append('outer_copular_predicate')
        if candidate is None and len(norm(first).split())<=6 and len(ss)==1:candidate=first;trace.append('short_answer_phrase');confidence='medium'
        if candidate is None:return result('unassessable','沒有可證明與問題目標相連的答案命題。',category='answer_scope_unresolved')
        candidate=trim(candidate)
        if re.search(r'\bor\b|\bbut\b|;',candidate,re.I):return result('ambiguous','候選片語含替代或轉折，未任選有利答案。',answer=candidate,category='alternative_answers')
        if NEG.search(candidate):return result('unassessable','名詞答案含未解的否定範圍。',answer=candidate,category='negation_scope')
        if len(norm(candidate).split())>10:return result('unassessable','候選命題過長，語意比對能力不足。',answer=candidate,category='long_answer_scope')
    if candidate is None:return result('unassessable','沒有形成可比較的答案命題。')
    canonical=norm(candidate,numeric=kind=='number_ordinal')
    trace.append('candidate:'+canonical)
    # Full-answer consistency: only repeated assertions with same subject or
    # explicit pronoun/answer are compared, not unrelated explanatory examples.
    if kind not in {'binary','number_ordinal','color','causal_reasoning'}:
        initial=re.match(r'^(.+?)\s+(?:is|are)\s+',first,re.I)
        if initial:
            subj=initial[1]
            for later in ss[1:]:
                mm=re.match(r'^(?:'+re.escape(subj)+r'|it|they)\s+(?:is|are)\s+(.+)',later,re.I)
                if mm:
                    laterkey=norm(trim(mm[1]))
                    if laterkey!=canonical and len(laterkey.split())<=3 and (set(laterkey.split())&OBJECTS or set(laterkey.split())&COLORS):return result('ambiguous','後句對同一主體給出另一短答案，未選有利一項。',answer=candidate,category='contradictory_answers')
    matches=[];matchtypes=[]
    for i,ref in enumerate(references):
        refkey=norm(ref,numeric=kind=='number_ordinal');equal=False;why=''
        if canonical==refkey:equal=True;why='canonical_exact'
        elif kind=='number_ordinal' and re.fullmatch(NUMPAT,ref.strip(),re.I) and number(ref)==canonical:equal=True;why='number_ordinal_equivalent'
        elif kind=='causal_reasoning':
            refkeys=set()
            if re.search(r'safety|protect|prevent.*(?:harm|escape)|security',refkey):refkeys.add('safety')
            if re.search(r'visibility|visible|see',refkey):refkeys.add('visibility')
            if re.search(r'eat|meal|food',refkey):refkeys.add('eating')
            if refkey in {'transportation','transport','travel','traveling','travelling','riding','ride','passenger','to travel','get ride'}:refkeys.add('transportation')
            if refkeys&set(candidate.split('|')):equal=True;why='finite_causal_equivalence'
        elif kind in {'object_attribute','person'}:
            # Harmless modifiers may be removed ONLY from a candidate already
            # bound to the question target and only for a noun identity question.
            identity_question=not re.search(r'colou?r|what kind|which kind|what type|which type|what material|what shape|what size',q,re.I)
            allowed_mod=COLORS|{'small','large','big','little','young','old','wooden','metal','plastic','white','striped','red','short','long'}
            ww=canonical.split()
            while ww and ww[0] in allowed_mod:ww=ww[1:]
            if identity_question and ww and ' '.join(ww)==refkey:equal=True;why='harmless_modifier_entailment'
            # Explicit activity phrase equivalences, not bag-of-words overlap.
            activity={'learning to skateboard':'skateboarding','riding skateboard':'skateboarding','skateboarding':'skateboarding','playing tennis':'tennis','playing baseball':'baseball','playing soccer':'soccer','playing football':'football'}
            if re.search(r'doing|activity|sport|play',q,re.I) and activity.get(canonical,canonical)==activity.get(refkey,refkey):equal=True;why='activity_phrase_equivalence'
        elif kind=='color':
            ww=canonical.split()
            if len(ww)==2 and ww[0] in {'dark','light'} and ww[1]==refkey and not re.search(r'shade|light or dark|dark or light',q,re.I):equal=True;why='base_color_entailment'
        if equal:matches.append(i);matchtypes.append(why)
    if matches:
        if any(x!='canonical_exact' for x in matchtypes) or norm(candidate)!=unicodedata.normalize('NFKC',candidate).lower().strip(' .'):confidence='medium'
        proof.extend(matchtypes)
        for group in SYNONYMS:
            if any(re.search(r'\b'+re.escape(x)+r'\b',candidate,re.I) for x in group):proof.append('finite_synonym_lexicon')
        return result('equivalent','已綁定的答案命題經列明數字、同義詞或受控修飾詞規則，與完整參考中的對應答案相容；未在原長文任意搜尋正解。',confidence,matches,canonical,category='supported_semantic_equivalence')
    refkeys=[norm(r,numeric=kind=='number_ordinal') for r in references]
    exclusive=(kind=='binary' and all(r in {'yes','no'} for r in refkeys)) or (kind=='number_ordinal' and all(re.fullmatch(r'\d+(?:\.\d+)?',r) for r in refkeys)) or (kind=='color' and canonical in COLORS and all(r in COLORS for r in refkeys)) or (kind=='spatial_location' and canonical in RELATIONS and all(r in RELATIONS for r in refkeys))
    if exclusive:return result('not_equivalent','目標命題在極性／數值／顏色／方向上與所有參考不相容。','high',answer=canonical,category='explicit_value_mismatch')
    person_words={'person','man','woman','child'}
    if canonical in person_words and any(x in person_words for x in refkeys):return result('unassessable','人物名稱具有上下位或身分關係；不是互斥物件，有限詞彙不據此宣告矛盾。',answer=canonical,category='person_hierarchy_unresolved')
    if canonical in OBJECTS and all(x in OBJECTS for x in refkeys):return result('not_equivalent','問題目標的明確物件名稱與有限詞彙中的全部參考物件不同；仍未看圖確認標註。','medium',answer=canonical,category='object_identity_mismatch')
    return result('unassessable','候選與參考的表面形式不同，有限語意詞彙無法證明同義或矛盾；不把未命中自動視為答錯。',answer=canonical,category='unsupported_semantic_relation')
