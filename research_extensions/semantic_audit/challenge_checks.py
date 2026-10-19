"""Post-freeze counterexample checks. Never tune D using these results."""
import json,hashlib,csv
from pathlib import Path
import semantic_rules_d as D
HERE=Path(__file__).resolve().parent
def main():
    frozen=json.loads((HERE/'FROZEN_D_RULES_v2.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((HERE/'semantic_rules_d.py').read_bytes()).hexdigest()==frozen['configuration']['code_sha256']['semantic_rules_d.py']
    cases=[
      ('wrong_counted_object','How many pizzas are there?','I see two people.',['two'],'not_equivalent'),
      ('wrong_actor','What is the man wearing?','The woman is wearing a white hat.',['hat'],'not_equivalent'),
      ('later_absence_contradiction','Is there a dog?','Yes, there is a dog. There are no dogs in the scene.',['yes'],'ambiguous'),
      ('later_correction','What color is the cup?','The cup is red. Actually, the cup is blue.',['red'],'ambiguous'),
      ('wrong_color_subject','What color is the jacket?','The bag is brown.',['brown'],'not_equivalent'),
      ('inverse_role_positive','Where is the cabinet relative to the freezer?','The freezer is to the left of the cabinet.',['right'],'equivalent'),
      ('unhandled_action_subject','What animal is chasing the cat?','The dog is chasing the cat.',['dog'],'equivalent'),
      ('opposite_purpose_food','Why did the boy go to the market?','He went to the market for selling food.',['buy food'],'not_equivalent'),
      ('currency','How much does it cost?','It costs three pounds.',['three dollars'],'not_equivalent'),
      ('later_relevant_answer','What color is the jacket?','The hat is green. The jacket is brown.',['brown'],'equivalent'),
      ('negated_distractor_same_clause','What animal is shown?','The animal is a dog, not a cat.',['dog'],'equivalent'),
      ('binary_incompatible_action','Is the dog running?','Yes, the dog is sleeping.',['yes'],'ambiguous'),
      ('all_quantifier_exception','Are all cups empty?','No, one cup is full.',['no'],'equivalent'),
      ('wrong_possessor_color','What color is the man\'s hat?','The woman is wearing a black hat.',['black'],'not_equivalent'),
      ('food_detail_conflict','Why are the cows lowering their heads?','They lower their heads because they are eating hay.',['eating grass'],'not_equivalent'),
      ('qualified_reference','Why is the person riding the bus?','The person is riding the bus for transportation.',['paying passenger'],'not_equivalent'),
      ('negated_causal_reference','Why are the cows lowering their heads?','They lower their heads because they are not eating any grass.',['eating grass'],'not_equivalent'),
    ]
    rr=[]
    for name,q,answer,refs,expected in cases:
        result=D.judge(q,answer,refs)
        rr.append({'challenge_id':name,'question':q,'raw_answer':answer,'references':json.dumps(refs),'expected_verdict':expected,
          'D_verdict':result['verdict'],'D_confidence':result['confidence'],'candidate':result['answer_proposition'],'reason':result['reason'],
          'passed':result['verdict']==expected,'false_accept':result['verdict']=='equivalent' and expected!='equivalent','synthetic_only':True})
    with (HERE/'POST_FREEZE_CHALLENGES.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]),quoting=csv.QUOTE_ALL);w.writeheader();w.writerows(rr)
    summary={'after_rules_freeze':True,'challenge_n':len(rr),'passed':sum(r['passed'] for r in rr),'false_accept_n':sum(r['false_accept'] for r in rr),
      'abstentions_n':sum(r['D_verdict'] in ['ambiguous','unassessable'] for r in rr),'not_human_or_dataset_accuracy':True,
      'passed_cases':[r['challenge_id'] for r in rr if r['passed']], 'failed_cases':[r['challenge_id'] for r in rr if not r['passed']],
      'do_not_adopt_for_full_dataset':True,'no_rule_changes_in_response_to_these_challenges':True}
    (HERE/'POST_FREEZE_CHALLENGE_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))
if __name__=='__main__':main()
