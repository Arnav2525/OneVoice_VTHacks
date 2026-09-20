export function makeWordParticles(host){
 const canvas=document.createElement('canvas');canvas.className='word-particles';canvas.setAttribute('aria-hidden','true');host.append(canvas);
 const ctx=canvas.getContext('2d'),mask=document.createElement('canvas');mask.width=1100;mask.height=220;
 const ink=mask.getContext('2d');ink.fillStyle='#fff';ink.font='600 130px "Segoe UI",Arial,sans-serif';ink.textAlign='center';ink.textBaseline='middle';ink.fillText('ONE VOICE',550,110);
 const pixels=ink.getImageData(0,0,1100,220).data,columns=[];
 for(let x=100;x<1000;x+=3){const spans=[];let start=-1;for(let y=10;y<211;y++){const on=y<210&&pixels[(y*1100+x)*4+3]>100;if(on&&start<0)start=y;if(!on&&start>=0){spans.push([start-110,y-110]);start=-1;}}columns.push({x:x-550,spans});}
 const ease=(x,a,b)=>{x=Math.max(0,Math.min(1,(x-a)/(b-a)));return x*x*x*(x*(x*6-15)+10);};let width=0,height=0;
 return {draw(time,reduced=false){
 const w=host.clientWidth,h=host.clientHeight,dpr=Math.min(devicePixelRatio,1.5);
 if(w!==width||h!==height){width=w;height=h;canvas.width=w*dpr;canvas.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);}
 ctx.clearRect(0,0,w,h);if(time<2.35||time>6.85)return;
 const morph=reduced?1:ease(time,3.3,4.65),resolve=ease(time,4.45,4.9),exit=ease(time,5.6,6.65),scale=Math.min(w/1150,h/500);
 host.dataset.particlePhase=morph<.99?'waveform':exit<.05?'hold':'release';
 ctx.save();ctx.translate(w/2,h*.49);ctx.scale(scale*(1+exit*.16),scale*(1+exit*.16));
 const gradient=ctx.createLinearGradient(-450,0,450,0);gradient.addColorStop(0,'#76d6ed');gradient.addColorStop(.32,'#d9eaff');gradient.addColorStop(.62,'#efb2bb');gradient.addColorStop(1,'#b2c6ff');
 ctx.strokeStyle=gradient;ctx.lineWidth=1.7;ctx.lineCap='round';ctx.shadowColor='#93cfff';ctx.shadowBlur=8*(1-resolve);
 for(const c of columns){
 const nx=c.x/450,envelope=Math.pow(Math.max(0,1-nx*nx),.7),amplitude=(18+75*Math.pow(Math.sin(nx*7.5-time*2.4),2)+22*Math.sin(nx*15+time))*envelope,center=Math.sin(nx*7-time*1.8)*18*envelope;
 const spans=c.spans.length?c.spans:[[0,0]],count=spans.length;ctx.globalAlpha=(1-exit)*(1-resolve)*(c.spans.length?1:1-morph);
 for(let j=0;j<count;j++){const from=center-amplitude+2*amplitude*j/count,to=center-amplitude+2*amplitude*(j+1)/count,y1=from*(1-morph)+spans[j][0]*morph,y2=to*(1-morph)+spans[j][1]*morph;ctx.beginPath();ctx.moveTo(c.x,y1);ctx.lineTo(c.x,y2);ctx.stroke();}}
 ctx.shadowBlur=16;ctx.shadowColor='#9bc8eb44';ctx.globalAlpha=resolve*(1-exit);ctx.drawImage(mask,-550,-110);
 ctx.shadowBlur=0;ctx.globalAlpha=(1-morph)*.28*(1-exit);ctx.lineWidth=.7;
 for(let line=0;line<3;line++){ctx.beginPath();for(let x=-450;x<=450;x+=5){const y=Math.sin(x*.018-time*1.9+line*.8)*45*Math.max(0,1-(x/450)**2);if(x===-450)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();}
 ctx.restore();ctx.globalAlpha=1;
 }};
}
