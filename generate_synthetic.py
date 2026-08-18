#!/usr/bin/env python3
import json,random,sqlite3,time,sys
from pathlib import Path
DB=Path(sys.argv[1] if len(sys.argv)>1 else "pilot.db");rng=random.Random(7)
traits=["state_dependence","info_value","resilience","control","goal_definition","tail_risk",
        "causal_discipline","reversibility","option_value","model_synthesis","systems_orientation","performance"]
models=["ChatGPT synthetic","Claude synthetic","Gemini synthetic"]
con=sqlite3.connect(DB)
for i in range(18):
    pid=f"synthetic{i:02d}"; model=models[i%3]
    h={t:round(rng.uniform(-1,1),3) for t in traits}
    # Family offset simulates a base-model signature; personal vector remains the larger component.
    fam_i=i%3
    family_bias=(fam_i-1)*0.08
    a={t:round(max(-1,min(1,h[t]+family_bias+rng.gauss(0,.20))),3) for t in traits}
    con.execute("INSERT OR REPLACE INTO participants(id,alias,age_confirmed,consent,created_at,finished_human_at,finished_ai_at) VALUES(?,?,?,?,?,?,?)",
                (pid,pid,1,1,time.time(),time.time(),time.time()))
    con.execute("INSERT OR REPLACE INTO fingerprints(participant_id,vector_json,created_at) VALUES(?,?,?)",(pid,json.dumps(h),time.time()))
    con.execute("INSERT OR REPLACE INTO ai_fingerprints(participant_id,model_name,vector_json,created_at) VALUES(?,?,?,?)",
                (pid,model,json.dumps(a),time.time()))
    con.execute("INSERT OR REPLACE INTO ai_metadata(participant_id,model_name,memory_status,custom_instructions,usage_duration,fresh_chat,created_at) VALUES(?,?,?,?,?,?,?)",
                (pid,model,"on","unknown","1-2y",1,time.time()))
con.commit();con.close();print("Inserted 18 synthetic pairs into",DB)
