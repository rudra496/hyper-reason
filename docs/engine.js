/*
 * HyperReason engine — pure JS functions (single source of truth for the demo).
 *
 * These mirror hyper_reason/engine/{math_utils,entropy,verifier}.py EXACTLY.
 * tests/test_js_parity.py runs this file under Node and compares to Python, so
 * the browser demo cannot silently drift from the package.
 *
 * Browser: defines globals. Node: exports the same functions.
 */

function shannonFromCounts(counts){
  var total = counts.reduce(function(a,b){return a+b;},0);
  if(total<=0) return 0;
  var h=0; for(var i=0;i<counts.length;i++){ var c=counts[i]; if(c>0){ var p=c/total; h-=p*Math.log2(p); } } return h;
}
function normalizeStep(text){
  return String(text).replace(/\\boxed\{[^}]*\}/g,'[boxed]').toLowerCase().replace(/\s+/g,' ').trim();
}
function sampleDiversityEntropy(texts){
  if(!texts || !texts.length) return 0;
  var b={}; for(var i=0;i<texts.length;i++){ var k=normalizeStep(texts[i]); b[k]=(b[k]||0)+1; }
  return shannonFromCounts(Object.keys(b).map(function(k){return b[k];}));
}
function priorsFromDiversity(texts){
  if(!texts || !texts.length) return [];
  var n=texts.map(normalizeStep); var c={}; for(var i=0;i<n.length;i++)c[n[i]]=(c[n[i]]||0)+1;
  return n.map(function(x){return c[x]/n.length;});
}
function extractFinalAnswer(text){
  text = String(text);
  var m=text.match(/\\boxed\{([^{}]*)\}/g);
  if(m){ var last=m[m.length-1].replace(/\\boxed\{|\}/g,'').trim(); if(last) return last; }
  var nums=text.match(/-?\d+(?:\.\d+)?/g); if(nums&&nums.length) return nums[nums.length-1];
  return "__unparsable__";
}
function selfConsistency(traces){
  if(!traces || !traces.length) return ["__unparsable__",0,{}];
  var ans=traces.map(extractFinalAnswer); var c={};
  for(var i=0;i<ans.length;i++)c[ans[i]]=(c[ans[i]]||0)+1;
  var best=Object.keys(c).reduce(function(a,b){return c[a]>=c[b]?a:b;});
  return [best, Math.round((c[best]/ans.length)*10000)/10000, c];
}

if (typeof window !== 'undefined') {
  window.shannonFromCounts = shannonFromCounts;
  window.normalizeStep = normalizeStep;
  window.sampleDiversityEntropy = sampleDiversityEntropy;
  window.priorsFromDiversity = priorsFromDiversity;
  window.extractFinalAnswer = extractFinalAnswer;
  window.selfConsistency = selfConsistency;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { shannonFromCounts:shannonFromCounts, normalizeStep:normalizeStep,
    sampleDiversityEntropy:sampleDiversityEntropy, priorsFromDiversity:priorsFromDiversity,
    extractFinalAnswer:extractFinalAnswer, selfConsistency:selfConsistency };
}
