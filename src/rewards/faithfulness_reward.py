# ~/medfaith/src/rewards/faithfulness_reward.py
def compute_reward(rec, weights, biobert, overlap_func):
    # rec expects: valid_json, answer_correct, question, caption, delta_cf_roi, delta_cf_rand
    w_fmt, w_ans, w_cons, w_cf = weights["fmt"], weights["ans"], weights["cons"], weights["cf"]

    fmt = 1.0 if rec.get("valid_json") else 0.0
    acc = 1.0 if rec.get("answer_correct") else 0.0

    q = rec.get("question","")
    c = rec.get("caption","")
    if q and c:
        q_emb = biobert.encode(q, convert_to_tensor=True, normalize_embeddings=True)
        c_emb = biobert.encode(c, convert_to_tensor=True, normalize_embeddings=True)
        cos = float((q_emb @ c_emb.T).cpu().item())
        ov  = float(overlap_func(q, c))
        cons = 0.8*cos + 0.2*ov
    else:
        cons = 0.0

    faith = float(rec.get("delta_cf_roi", 0.0) - rec.get("delta_cf_rand", 0.0))

    R = w_fmt*fmt + w_ans*acc + w_cons*cons + w_cf*faith
    return R
