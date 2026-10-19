"""Frozen, reference-blind VQA parser v1.0. No ground truth argument or import.

This is a conservative text transformation, not a semantic oracle. A parsed
answer may still be wrong. Unparsed/ambiguous records remain in the denominator.
"""
import re,unicodedata
VERSION='2.0'
NUMBERS=dict(zip('zero one two three four five six seven eight nine ten'.split(),map(str,range(11))))
NUMBERS['none']='0'
COLORS=set('black white red green blue brown yellow orange pink purple gray grey silver gold beige tan maroon turquoise cyan violet khaki cream'.split())
UNCERTAIN=re.compile(r"\b(cannot|can't|unable to|unclear|uncertain|not sure|not enough|insufficient|cannot be determined|can't be determined|could be|might be|may be|hard to tell|difficult to determine|does not provide enough|doesn't provide enough|not possible to determine)\b",re.I)
SPECIAL=re.compile(r'<\|[^>]+\|>|</?(?:s|im_start|im_end|image|img|think)>|\[/?INST\]|^\s*(?:assistant|user|system)\s*:',re.I)
def normalize(s):
    s=unicodedata.normalize('NFKC',s).lower().replace('’',"'").replace('\n',' ').replace('\t',' ')
    s=re.sub(r'(?<=\d),(?=\d)','',s)
    s=re.sub(r'(?<!\d)\.|\.(?!\d)',' ',s)
    s=re.sub(r"[^\w\s'.:]",' ',s)
    words=[]
    contractions={"dont":"don't","doesnt":"doesn't","isnt":"isn't","arent":"aren't","cant":"can't","wont":"won't"}
    for word in s.split():
        if word not in {'a','an','the'}:words.append(contractions.get(word,NUMBERS.get(word,word)))
    return ' '.join(words)
def sentences(s):return [x.strip() for x in re.split(r'(?<=[.!?])\s+|[\r\n]+',s.strip()) if x.strip()]
def question_kind(q):
    q=q.lower().strip()
    if re.search(r'\b(how many|how much|which number|what number|number of)\b',q):return 'number'
    # Or-questions naming alternatives are not binary; existential/logical or is.
    choice=bool(re.search(r'\bor\b',q)) and not bool(re.search(r'\b(there|either|any|both)\b',q))
    if re.match(r'^(is|are|was|were|do|does|did|has|have|had|can|could|will|would)\b',q) and not choice:return 'yes_no'
    if 'color' in q or 'colour' in q:return 'color'
    if re.search(r'\b(which side|what side|on which side|where|left or|right or|on the left or)\b',q):return 'spatial'
    if choice:return 'choice'
    if re.match(r'^(why|how)\b',q):return 'reasoning'
    if re.match(r'^who\b',q):return 'person'
    if re.search(r'\b(what|which)\b',q):return 'object_attribute'
    return 'other'
def clean_phrase(s):
    s=s.strip().strip(' .!?;:"“”')
    s=re.sub(r'^(?:likely|probably|apparently|clearly)\s+','',s,flags=re.I)
    # Stop only at explicit explanatory boundaries. Never select an embedded
    # reference word or drop adjectives to improve a match.
    s=re.split(r',\s*(?:which|who|where|as|because|suggesting|indicating)\b|\s+(?:because|since|suggesting|indicating)\b',s,1,flags=re.I)[0]
    s=re.sub(r'\s+(?:in (?:the|this) (?:image|picture|photo|photograph|scene)|shown (?:here|in the image)|visible (?:here|in the image))\s*$','',s,flags=re.I)
    return s.strip(' .!?;:')
def extract(question,raw_prediction):
    """Inputs deliberately limited to question and raw prediction (no refs)."""
    q=question.strip();s=raw_prediction.strip();kind=question_kind(q)
    def fail(status,reason):return {'answer':'','status':status,'rule':reason,'question_kind':kind}
    def ok(answer,rule):
        answer=clean_phrase(answer)
        if not answer:return fail('unparseable','empty_candidate')
        if re.search(r'\b(?:or|but|however|alternatively|either)\b|;',answer,re.I):return fail('ambiguous','alternative_or_contrast_in_answer')
        if len(normalize(answer).split())>8:return fail('unparseable','candidate_over_8_words')
        if UNCERTAIN.search(answer):return fail('ambiguous','uncertain_candidate')
        return {'answer':normalize(answer),'status':'parsed','rule':rule,'question_kind':kind}
    if not s or SPECIAL.search(s) or '\ufffd' in s:return fail('unparseable','empty_special_token_or_replacement_character')
    # A hedge that explicitly denies certainty anywhere is retained as ambiguous.
    if UNCERTAIN.search(s):return fail('ambiguous','explicit_uncertainty_anywhere')
    ss=sentences(s);first=ss[0].strip().rstrip('.!?')
    # Contradictory explicit final/initial answers, independent of reference.
    explicit=[]
    for sent in ss:
        m=re.match(r'^(yes|no)\b',sent,re.I)
        if m:explicit.append(m[1].lower())
        explicit.extend(m.lower() for m in re.findall(r'\b(?:the\s+)?answer\s+is\s+(yes|no)\b',sent,re.I))
    if len(set(explicit))>1:return fail('ambiguous','contradictory_yes_no_assertions')
    if kind=='yes_no':
        if explicit:
            # Do not turn a qualified "yes, but ..." into an unconditional yes.
            if re.search(r'\b(but|however|although)\b',first,re.I):return fail('ambiguous','qualified_binary_answer')
            if explicit[0]=='yes' and re.search(r"\b(no|not|never|isn't|aren't|doesn't|don't)\b",first,re.I):return fail('ambiguous','yes_followed_by_negative_clause')
            return ok(explicit[0],'explicit_binary')
        # No implicit positivity/negativity: subject and scope need semantics.
        return fail('unparseable','binary_without_explicit_yes_no')
    # Preserve all of an already short non-sentential answer.
    if len(normalize(first).split())<=5 and len(ss)==1 and not re.search(r'\b(is|are|was|were|has|have|shows|depicts|appears|see|sees|look|looks|know|think)\b',first,re.I) and (kind!='number' or re.fullmatch(r'\d+(?:\.\d+)?',normalize(first))):
        return ok(first,'short_direct_answer')
    candidate=None;rule=None
    m=re.match(r'^(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)\s+(.+)$',first,re.I)
    if m:candidate=m[1];rule='explicit_answer_marker'
    elif kind=='number':
        quantities=re.findall(r'\b(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b',s,re.I)
        if len({normalize(x) for x in quantities})>1:return fail('ambiguous','multiple_distinct_quantities')
        m=re.match(r'^(?:there\s+(?:is|are)\s+|(?:the\s+)?number(?:\s+of\s+.+?)?\s+is\s+)(\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b',first,re.I)
        if m:candidate=m[1];rule='anchored_numeric_answer'
        else:return fail('unparseable','number_without_anchored_quantity')
    elif kind=='reasoning':return fail('unparseable','open_reasoning_requires_semantics')
    elif re.match(r'^who\b',q,re.I):
        m=re.match(r'^(.+?)\s+(?:is|are|was|were)\s+(?:wearing|holding|sitting|standing|riding|playing|walking|eating|looking)\b',first,re.I)
        if m:candidate=m[1];rule='who_subject'
    if candidate is None:
        # Inverse relation question: "What is on the island?" --> subject,
        # only if the answer repeats the same initial relation word.
        m=re.match(r'^(?:what|which)(?:\s+\w+)?\s+(?:is|are)\s+(on|under|above|below|behind|beside|near|inside|outside|in front of|next to)\b',q,re.I)
        if m:
            am=re.match(r'^(.+?)\s+(?:is|are)\s+'+re.escape(m[1])+r'\b',first,re.I)
            if am:candidate=am[1];rule='inverse_relation_subject'
    if candidate is None:
        # Only remove an activity whose lexical root occurs in the question.
        for root,verb in [('wear','wearing'),('hold','holding'),('ride','riding'),('eat','eating'),('drink','drinking'),('use','using'),('play','playing')]:
            if re.search(r'\b'+root+r'(?:s|ing)?\b',q,re.I) or verb in q.lower():
                m=re.match(r'^.+?\s+(?:is|are)\s+'+verb+r'\s+(.+)$',first,re.I)
                if m:candidate=m[1];rule='question_matched_activity_object';break
    if candidate is None:
        m=re.match(r'^(?:the\s+)?(?:image|picture|photo|scene)\s+(?:shows|depicts)\s+(.+)$',first,re.I)
        if m and re.search(r'\b(room|place|scene|setting)\b',q,re.I):candidate=m[1];rule='scene_description'
    if candidate is None:
        # V2: do not guess syntactic scope when a relative clause contains a
        # second copula. V1's last-copula rule is preserved in development_v1.
        matches=list(re.finditer(r'\b(?:is|are|was|were|appears to be|seems to be)\s+',first,re.I))
        if len(matches)>1:return fail('unparseable','multiple_copulas_scope_requires_semantics')
        if matches:
            m=matches[-1];candidate=first[m.end():];rule='copular_complement'
            candidate=re.sub(r'^(?:likely|probably|apparently)\s+','',candidate,flags=re.I)
            candidate=re.sub(r'^(?:made (?:out )?of|described as|called|located)\s+','',candidate,flags=re.I)
            # A what/which-subject relation answered by a prepositional
            # complement requires an inverse rule; don't return the relation.
            if re.match(r'^(?:what|which)\s+(?:is|are|was|were)\b',q,re.I) and re.match(r'^(?:in|on|under|above|below|behind|beside|near|inside|outside|next to|to the)\b',candidate,re.I):
                return fail('unparseable','inverse_relation_not_resolved')
    if candidate is None:return fail('unparseable','no_supported_syntactic_rule')
    candidate=clean_phrase(candidate)
    if re.search(r'\b(or|but|however|alternatively|either)\b',candidate,re.I):return fail('ambiguous','multiple_or_contrasting_candidates')
    # Reject explicit later correction/assertion about the same subject.
    head=first.split(' is ')[0].split(' are ')[0].lower()
    if head:
        for later in ss[1:]:
            if later.lower().startswith(head+' is ') or later.lower().startswith(head+' are '):
                tail=re.split(r'\b(?:is|are)\s+',later,maxsplit=1,flags=re.I)[-1]
                if normalize(clean_phrase(tail))!=normalize(candidate):return fail('ambiguous','later_subject_assertion_differs')
    # A subsequent pronoun stating a different color is also a contradiction.
    if kind in {'color','choice'} and set(normalize(candidate).split())<=COLORS|{'light','dark','and'}:
        for later in ss[1:]:
            m=re.match(r'^(?:it|they)\s+(?:is|are)\s+(.+?)[.!?]?$',later,re.I)
            if m:
                tail=normalize(clean_phrase(m[1]))
                if tail and set(tail.split())<=COLORS|{'light','dark','and'} and tail!=normalize(candidate):return fail('ambiguous','later_pronoun_color_differs')
    if kind=='color':
        words=normalize(candidate).split()
        if not words or not any(w in COLORS for w in words) or any(w not in COLORS|{'light','dark','and','color','colored','colour'} for w in words):return fail('unparseable','color_not_a_color_phrase')
        candidate=re.sub(r'\s+(?:color|colour)$','',candidate,flags=re.I)
    if kind=='spatial':
        # Canonical side names, only when the entire candidate is a side phrase.
        m=re.fullmatch(r'(?:on\s+)?(?:the\s+)?(left|right)(?:\s+side)?(?:\s+of\s+(?:the\s+)?(?:image|picture|photo))?',candidate,re.I)
        if m:candidate=m[1];rule+=':side_phrase'
    return ok(candidate,rule)

def format_flags(q,s,parsed):
    tokens=normalize(s).split();ss=sentences(s);a=parsed['answer'];kind=parsed['question_kind']
    return {'A_single_word':len(tokens)==1,'B_sentence':bool(re.search(r'\b(is|are|was|were|has|have|shows|depicts|can|does|wearing|holding)\b',s,re.I)),
      'C_yes_no':bool(re.match(r'^\s*(yes|no)\b',s,re.I) or re.search(r'\banswer is (yes|no)\b',s,re.I)),
      'D_numeric':kind=='number' or bool(a and re.fullmatch(r'\d+(?:\.\d+)?',a)),
      'E_color':kind=='color' or bool(a and set(a.split())<=COLORS|{'and','dark','light'}),
      'F_object_name_candidate':parsed['status']=='parsed' and kind in {'object_attribute','person'} and bool(a) and len(a.split())<=5,
      'G_long_explanatory':len(tokens)>=40 or len(ss)>=3,'H_uncertain_refusal':bool(UNCERTAIN.search(s)),
      'I_abnormal':not s.strip() or bool(SPECIAL.search(s)) or '\ufffd' in s,
      'template_residue':bool(SPECIAL.search(s)),'multiple_sentences':len(ss)>1,'word_count':len(tokens)}

def self_test():
    cases=[('Is the jacket brown or green?','The jacket is brown. Brown is common.','brown','parsed'),
      ('What color is the cup?','The cup is blue.','blue','parsed'),
      ('What color is the cup?','The cup is red or blue.','','ambiguous'),
      ('Is there a dog?','Yes, there is a dog. The answer is no.','','ambiguous'),
      ('Is there a dog?','Yes, there is not a dog.','','ambiguous'),
      ('How many cats are there?','There are two cats.','2','parsed'),
      ('What is on the table?','An apple is on the table.','apple','parsed'),
      ('What is the animal?','The animal is a horse. A dog is not shown.','horse','parsed'),
      ('What color is it?','It might be red.','','ambiguous'),
      ('Why is it there?','It is there for protection.','','unparseable'),
      ('Who is sitting?','The man is sitting on a chair.','man','parsed'),
      ('Is there a cat?','The cat is visible.','','unparseable'),
      ('What is it?','<|im_start|>assistant: cat','','unparseable'),
      ('What animal is it?','The animal is a horse. The animal is a dog.','','ambiguous')]
    for q,s,a,status in cases:
        r=extract(q,s);assert (r['answer'],r['status'])==(a,status),(q,s,r)
    assert normalize('The TWO cats, 1,000!')=='2 cats 1000'
    assert normalize('3.14')=='3.14'
    # Candidate stays unchanged irrespective of possible hidden ground truth.
    assert extract('What is the animal?','The animal is a horse. A dog is not shown.')['answer']=='horse'
    more=[('Is the cabinet to the left or right of the freezer that is on the left?','The cabinet is to the right of the freezer that is on the left.','','unparseable'),
      ('How many pizzas do you see?','I see two pizzas.','','unparseable'),
      ('What color is the cup?','The cup is blue. It is red.','','ambiguous')]
    for q,s,a,status in more:
        r=extract(q,s);assert (r['answer'],r['status'])==(a,status),(q,s,r)
    return len(cases)+len(more)+3
if __name__=='__main__':print('Synthetic rule tests passed:',self_test())
