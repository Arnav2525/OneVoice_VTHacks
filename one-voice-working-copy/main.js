import {createWorld} from './scene.js';
const $=selector=>document.querySelector(selector);
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
const expedition=$('.expedition'),worldElement=$('.world'),chapters=[...document.querySelectorAll('[data-chapter]')],jumps=[...document.querySelectorAll('[data-jump]')];
const names=['00 — INTRODUCTION','01 — PERSPECTIVE','02 — FORM','03 — CONNECTION','04 — THE MOMENT'];
let world=null,target=0,current=0,last=performance.now(),paused=reduced.matches,sceneFailed=false,lensArrived=false;
function measure(){const rect=expedition.getBoundingClientRect();target=Math.min(1,Math.max(0,-rect.top/(expedition.offsetHeight-worldElement.offsetHeight)));$('.site-header').classList.toggle('is-grounded',rect.bottom<worldElement.offsetHeight*.75);}
addEventListener('scroll',measure,{passive:true});addEventListener('resize',measure);measure();
function tick(now){const dt=Math.min((now-last)/1000,.06);last=now;current=reduced.matches?target:current+(target-current)*(1-Math.exp(-dt*9));if(Math.abs(target-current)<.0001)current=target;const p=current*6;
  chapters.forEach((chapter,i)=>{const distance=Math.abs(p-i),opacity=Math.max(0,1-distance*(i===4&&p>4?5:1.65));chapter.style.opacity=opacity;chapter.inert=opacity<.55;chapter.setAttribute('aria-hidden',String(opacity<.1));const shift=(i-p)*26;chapter.style.transform=i===4?`translate(-50%,${shift}px)`:`translateY(${shift}px)`;});
  const active=Math.min(4,Math.round(p));jumps.forEach((el,i)=>{if(i===active)el.setAttribute('aria-current','step');else el.removeAttribute('aria-current');});
  $('#chapter-counter').textContent=names[active];$('.world-progress span').style.width=`${current*100}%`;
  const chromeOpacity=Math.max(0,1-(p-3.7)*1.5);
  for(const el of document.querySelectorAll('.world-bottom,.scene-coordinates,.chapter-nav,.experience-tools,.travel-caption')){el.style.opacity=chromeOpacity;el.inert=chromeOpacity<.2;}
  if(sceneFailed){$('#conversation').style.opacity=0;$('#world-error').hidden=false;}
  $('.manifesto').style.opacity=Math.max(0,1-p*2);$('#annotation-label').textContent=p<1.6?'C270 / CAMERA DOCK':p<2.5?'ONE-PIECE / V5':'WIRED / LISTENING';
  if(world)world.setProgress(current);
  if(p>=4.5&&!appPrefetchStarted){appPrefetchStarted=true;checkApp().catch(()=>{});}
  if(p<4.3)appPrefetchStarted=false;
  if(p>=5.62&&!lensArrived&&!dialog.open){lensArrived=true;connectToApp();}
  if(p<5.3)lensArrived=false;
  requestAnimationFrame(tick);
}requestAnimationFrame(tick);
function goToProgress(progress){const top=expedition.getBoundingClientRect().top+scrollY+(expedition.offsetHeight-worldElement.offsetHeight)*progress;scrollTo({top,behavior:reduced.matches?'instant':'smooth'});}
for(const jump of jumps)jump.addEventListener('click',()=>goToProgress(Number(jump.dataset.jump)===4?1:Number(jump.dataset.jump)/6));
$('#scroll-prompt').addEventListener('click',e=>{e.preventDefault();goToProgress(Math.min(1,(Math.floor(current*6)+1)/6));});
for(const link of document.querySelectorAll('a[href="#conversation"]:not(#scroll-prompt)'))link.addEventListener('click',e=>{e.preventDefault();goToProgress(1);});
$('#rotate-left').addEventListener('click',()=>world?.rotate(-.3));$('#rotate-right').addEventListener('click',()=>world?.rotate(.3));$('#reset-view').addEventListener('click',()=>world?.reset());
function openAbout(){
  if(!world||sceneFailed){location.assign('about.html');return;}
  goToProgress(0);
  world.travel();
}
$('#travel-toggle').addEventListener('click',openAbout);
for(const link of document.querySelectorAll('[data-about]'))link.addEventListener('click',e=>{e.preventDefault();openAbout();});
$('#yeti-hello').addEventListener('click',()=>world?.greet());
$('#details-toggle').addEventListener('click',()=>{const b=$('#details-toggle'),value=b.getAttribute('aria-pressed')!=='true';b.setAttribute('aria-pressed',String(value));b.innerHTML=value?'HIDE DETAILS <span>−</span>':'EXPLORE DETAILS <span>+</span>';world?.setDetails(value);});
$('#light-toggle').addEventListener('click',()=>{const b=$('#light-toggle'),value=b.getAttribute('aria-pressed')!=='true';b.setAttribute('aria-pressed',String(value));b.innerHTML=value?'SUNSET <span>◐</span>':'DAYLIGHT <span>◐</span>';b.setAttribute('aria-label',value?'Switch to daylight lighting':'Switch to sunset lighting');world?.setSunset(value);});
for(const pin of document.querySelectorAll('[data-detail]'))pin.addEventListener('click',()=>{world?.reset();goToProgress((Number(pin.dataset.detail)+1)/6);});
function updateMotion(){const button=$('#motion-toggle');button.textContent=paused?'MOTION OFF':'MOTION ON';button.setAttribute('aria-pressed',String(paused));button.setAttribute('aria-label',paused?'Resume ambient motion':'Pause ambient motion');world?.setPaused(paused);}updateMotion();
$('#motion-toggle').addEventListener('click',()=>{paused=!paused;updateMotion();});reduced.addEventListener('change',e=>{paused=e.matches;updateMotion();});
createWorld($('#world-canvas'),()=>$('#loading').classList.add('done'),()=>location.assign('about.html?arrival=yeti')).then(result=>{world=result;$("#travel-toggle").disabled=false;updateMotion();world.setDetails($("#details-toggle").getAttribute("aria-pressed")==="true");world.setSunset($("#light-toggle").getAttribute("aria-pressed")==="true");}).catch(error=>{console.error('3D scene could not start',error);sceneFailed=true;$('#loading').classList.add('done');$('#world-error').hidden=false;for(const el of document.querySelectorAll('.scene-controls button'))el.disabled=true;});

const dialog=$('#connection-dialog');
const appUrl='http://127.0.0.1:8771/';
let connecting=false,appCheck=null,appCheckStarted=0,appPrefetchStarted=false;
function checkApp(){
  if(appCheck&&Date.now()-appCheckStarted<5000)return appCheck;
  appCheckStarted=Date.now();
  appCheck=fetch(`${appUrl}api/ping`,{cache:'no-store',signal:AbortSignal.timeout(2500)})
    .then(async response=>{if(!response.ok||(await response.json()).app!=='onevoice')throw new Error('OneVoice app is unavailable');})
    .catch(error=>{appCheck=null;throw error;});
  return appCheck;
}
async function connectToApp(){
  if(connecting)return;
  connecting=true;
  $('#conversation').classList.remove('handoff-failed');
  try{
    await checkApp();
    location.assign(`${appUrl}?from=story`);
  }catch{
    $('#conversation').classList.add('handoff-failed');
    if(!dialog.open)dialog.showModal();
  }finally{connecting=false;}
}
for(const button of document.querySelectorAll('[data-connect]'))button.addEventListener('click',e=>{e.preventDefault();connectToApp();});
window.oneVoiceConnect=connectToApp;
$('#close-connection').addEventListener('click',()=>dialog.close());
$('#back-to-story').addEventListener('click',()=>dialog.close());
$('#retry-connection').addEventListener('click',connectToApp);
dialog.addEventListener('click',e=>{const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();});

addEventListener('keydown',e=>{
  if(!['PageDown','PageUp'].includes(e.key)||dialog.open||e.ctrlKey||e.altKey||e.metaKey)return;
  if(e.target.matches('input,textarea,select,[contenteditable="true"]'))return;
  const start=expedition.getBoundingClientRect().top+scrollY;
  const travel=expedition.offsetHeight-worldElement.offsetHeight;
  const stops=[0,1/6,2/6,3/6,4/6,1].map(p=>start+travel*p);
  stops.push($('#connect').getBoundingClientRect().top+scrollY-70);
  const next=e.key==='PageDown'?stops.find(y=>y>scrollY+30):stops.reverse().find(y=>y<scrollY-30);
  if(next!==undefined){e.preventDefault();scrollTo({top:next,behavior:reduced.matches?'instant':'smooth'});}
});

for(const event of ['wheel','touchmove'])worldElement.addEventListener(event,e=>{if(worldElement.classList.contains('is-travelling'))e.preventDefault();},{passive:false});
