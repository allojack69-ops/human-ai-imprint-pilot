
let pid = localStorage.getItem("pilot_pid_v14") || null;
let current = null;
let shownAt = 0;
const $ = id => document.getElementById(id);

function show(id){
  ["start","test","humanDone","aiStage","finish"].forEach(x=>$(x).classList.add("hidden"));
  $(id).classList.remove("hidden");
}
async function api(url,opts={}){
  const r=await fetch(url,{headers:{"Content-Type":"application/json"},...opts});
  const j=await r.json().catch(()=>({}));
  if(!r.ok) throw new Error(j.detail || "Помилка");
  return j;
}
$("startBtn").onclick=async()=>{
  $("startErr").textContent="";
  try{
    const j=await api("/api/start",{method:"POST",body:JSON.stringify({
      alias:$("alias").value.trim(),age_confirmed:$("adult").checked,consent:$("consent").checked
    })});
    pid=j.participant_id; localStorage.setItem("pilot_pid_v14",pid); await loadNext();
  }catch(e){$("startErr").textContent=e.message}
};
async function loadNext(){
  show("test");
  const j=await api(`/api/next/${pid}`);
  if(j.done){show("humanDone");return}
  current=j.question; shownAt=performance.now();
  $("progress").textContent=`${j.progress.answered+1}/${j.progress.target}`;
  $("barFill").style.width=`${Math.min(100,(j.progress.answered/j.progress.target)*100)}%`;
  $("level").textContent=current.level.toUpperCase();
  $("qtext").textContent=current.text;
  $("options").innerHTML="";
  $("freeText").classList.add("hidden");$("submitFree").classList.add("hidden");
  if(current.free_text){
    $("freeText").value="";$("freeText").classList.remove("hidden");$("submitFree").classList.remove("hidden");
  }else{
    current.options.forEach(o=>{
      const b=document.createElement("button");
      b.className="option";b.textContent=o.text;b.onclick=()=>sendAnswer(o.id,"");
      $("options").appendChild(b);
    });
  }
}
async function sendAnswer(answerId,text){
  const ms=Math.round(performance.now()-shownAt);
  await api("/api/answer",{method:"POST",body:JSON.stringify({
    participant_id:pid,question_id:current.id,answer_id:answerId,answer_text:text,reaction_ms:ms,changed_answer:false
  })});
  await loadNext();
}
$("submitFree").onclick=async()=>{
  const t=$("freeText").value.trim(); if(t.length<3)return; await sendAnswer(null,t);
};
$("loadAIPack").onclick=async()=>{
  const j=await api(`/api/ai-pack/${pid}`);$("aiPrompt").value=j.copy_prompt;show("aiStage");
};
$("copyPrompt").onclick=async()=>{
  await navigator.clipboard.writeText($("aiPrompt").value);
  $("copyPrompt").textContent="Скопійовано";setTimeout(()=>$("copyPrompt").textContent="Скопіювати prompt",1200);
};
$("submitAI").onclick=async()=>{
  $("aiErr").textContent="";
  try{
    const raw=$("aiResult").value.trim();
    const obj=JSON.parse(raw);
    const answers=obj.answers||obj;
    const j=await api("/api/ai-submit",{method:"POST",body:JSON.stringify({
      participant_id:pid,
      model_name:$("modelName").value.trim()||"unknown",
      memory_status:$("memoryStatus").value,
      custom_instructions:$("customInstructions").value,
      usage_duration:$("usageDuration").value,
      fresh_chat:true,
      answers
    })});
    $("pidShow").textContent=pid; show("finish");
  }catch(e){$("aiErr").textContent="Перевір JSON: "+e.message}
};
(async()=>{
  if(pid){
    try{
      const j=await api(`/api/next/${pid}`);
      if(j.done)show("humanDone"); else await loadNext();
    }catch(e){localStorage.removeItem("pilot_pid_v14");pid=null;show("start")}
  }
})();