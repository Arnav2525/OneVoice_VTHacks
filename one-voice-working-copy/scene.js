import * as THREE from 'three';
import { RoomEnvironment } from './vendor/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { makeTerrain, elevation, hash } from './terrain.js';
import { makeWind } from './weather.js';
import { makeYeti } from './yeti.js';
import { makeAtmosphere } from './atmosphere.js';
import {makeSuitcase} from './journey.js';

export async function createWorld(canvas, onReady, onPacked) {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = .95;
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFShadowMap;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#a8adba');
  scene.fog = new THREE.FogExp2('#b5c5d6', .0105);
  const atmosphere=makeAtmosphere(scene);
  const camera = new THREE.PerspectiveCamera(37, 1, .015, 240);
  camera.position.set(0, 1.8, 12.6);
  camera.lookAt(0, .1, 0);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  scene.environment = pmrem.fromScene(room, .04).texture;
  room.dispose(); pmrem.dispose();
  const skyLight=new THREE.HemisphereLight('#dfeaff','#263247',.72);scene.add(skyLight);
  const sun = new THREE.DirectionalLight('#fff2df', 2.8);
  sun.position.set(-20, 19, 12);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);Object.assign(sun.shadow.camera,{left:-28,right:28,top:28,bottom:-28,near:.5,far:100});sun.shadow.bias=-.00025;sun.shadow.normalBias=.06;sun.shadow.radius=2;scene.add(sun);
  const rim = new THREE.DirectionalLight('#b1d1ff', 1.35);
  rim.position.set(5, 3, -8); scene.add(rim);

  const alpineTerrain=new THREE.Group();scene.add(alpineTerrain);await makeTerrain(alpineTerrain,renderer);

  const haze = new THREE.Mesh(new THREE.PlaneGeometry(240,80),new THREE.MeshBasicMaterial({color:'#b9c4d3',transparent:true,opacity:.12,depthWrite:false}));
  haze.position.set(0,20,-110); scene.add(haze);
  const product=new THREE.Group(); scene.add(product);
  const cradle=new THREE.Group(); product.add(cradle);
  const response=await fetch('assets/glasses-v5.ovm');
  if(!response.ok)throw new Error('V5 model is unavailable');
  const buffer=await response.arrayBuffer(),header=new DataView(buffer);
  if(header.getUint32(0,true)!==0x314d564f)throw new Error('Invalid V5 model');
  const vertexCount=header.getUint32(4,true),indexCount=header.getUint32(8,true);
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(buffer,16,vertexCount*3),3));
  geometry.setAttribute('normal',new THREE.BufferAttribute(new Float32Array(buffer,16+vertexCount*12,vertexCount*3),3));
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(buffer,16+vertexCount*24,indexCount),1));
  geometry.computeBoundingSphere();
  const frameMaterial = new THREE.MeshPhysicalMaterial({color:'#536c88',metalness:.72,roughness:.22,clearcoat:.8,clearcoatRoughness:.23,envMapIntensity:1.45});
  frameMaterial.onBeforeCompile=shader=>{
    shader.fragmentShader=shader.fragmentShader.replace('#include <opaque_fragment>','outgoingLight += vec3(.65,.85,1.) * pow(1.-abs(dot(normal,normalize(vViewPosition))),3.) * .8;\n#include <opaque_fragment>');
  };
  const frame=new THREE.Mesh(geometry,frameMaterial); frame.castShadow=true;frame.receiveShadow=true;cradle.add(frame);

  function roundedBlock(w,h,d,r,material) {
    const s=new THREE.Shape(),x=-w/2,y=-h/2;
    s.moveTo(x+r,y);s.lineTo(x+w-r,y);s.quadraticCurveTo(x+w,y,x+w,y+r);s.lineTo(x+w,y+h-r);s.quadraticCurveTo(x+w,y+h,x+w-r,y+h);s.lineTo(x+r,y+h);s.quadraticCurveTo(x,y+h,x,y+h-r);s.lineTo(x,y+r);s.quadraticCurveTo(x,y,x+r,y);
    const g=new THREE.ExtrudeGeometry(s,{depth:d,bevelEnabled:true,bevelSegments:3,steps:1,bevelSize:.018,bevelThickness:.018,curveSegments:10});g.translate(0,0,-d/2);
    const object=new THREE.Mesh(g,material);object.castShadow=true;object.receiveShadow=true;return object;
  }
  const dark=new THREE.MeshStandardMaterial({color:'#182434',metalness:.48,roughness:.31});
  const silver=new THREE.MeshStandardMaterial({color:'#7c8da3',metalness:.8,roughness:.28});
  const cameraHousing=new THREE.Group();cameraHousing.position.set(0,.82,2.3);cradle.add(cameraHousing);
  const cameraClip=roundedBlock(.62,.43,.46,.04,dark);cameraClip.position.set(0,.4,2.15);cradle.add(cameraClip);
  cameraHousing.add(roundedBlock(1.78,.73,.40,.17,dark));
  const front=roundedBlock(1.61,.55,.04,.14,silver);front.position.z=.225;cameraHousing.add(front);
  const lensRing=new THREE.Mesh(new THREE.CylinderGeometry(.205,.205,.07,48),dark);lensRing.rotation.x=Math.PI/2;lensRing.position.set(-.36,0,.28);cameraHousing.add(lensRing);
  const lens=new THREE.Mesh(new THREE.CircleGeometry(.139,48),new THREE.MeshPhysicalMaterial({color:'#06111b',metalness:.9,roughness:.04,clearcoat:1}));lens.position.set(-.36,0,.32);cameraHousing.add(lens);
  const lensLight=new THREE.Mesh(new THREE.CircleGeometry(.04,20),new THREE.MeshBasicMaterial({color:'#9bc9ff'}));lensLight.position.set(-.405,.04,.324);cameraHousing.add(lensLight);
  const led=new THREE.Mesh(new THREE.SphereGeometry(.025,12,8),new THREE.MeshBasicMaterial({color:'#a9e6e0'}));led.position.set(.58,.03,.27);cameraHousing.add(led);
  for(let i=0;i<5;i++){const slot=roundedBlock(.23,.013,.01,.005,dark);slot.position.set(.25,-.09+i*.045,.257);cameraHousing.add(slot);}

  const windRoot=new THREE.Group();scene.add(windRoot);const wind=makeWind(windRoot);
  const yeti=makeYeti();
  yeti.root.position.set(7.0,elevation(7.0,-10)-.08,-10);
  yeti.root.scale.setScalar(1.25);
  scene.add(yeti.root);

  const earbuds=new THREE.Group();cradle.add(earbuds);
  for(const side of [-1,1]){
    const curve=new THREE.CatmullRomCurve3([new THREE.Vector3(side*1.66,-.05,-1.45),new THREE.Vector3(side*1.9,-.8,-1.6),new THREE.Vector3(side*1.7,-1.9,-1.25),new THREE.Vector3(side*1.43,-1.45,-.65)]);
    const cable=new THREE.Mesh(new THREE.TubeGeometry(curve,48,.024,6,false),dark);earbuds.add(cable);
    const bud=new THREE.Mesh(new THREE.SphereGeometry(.17,20,16),silver);bud.scale.set(.72,1,1);bud.position.copy(curve.getPoint(1));earbuds.add(bud);
    const tip=new THREE.Mesh(new THREE.SphereGeometry(.095,16,12),dark);tip.position.copy(bud.position);tip.position.x-=side*.12;earbuds.add(tip);
  }

  const count=2400,positions=new Float32Array(count*3),seeds=new Float32Array(count);
  for(let i=0;i<count;i++){positions[i*3]=(hash(i,1)-.5)*70;positions[i*3+1]=hash(i,2)*35-10;positions[i*3+2]=(hash(i,3)-.5)*75;seeds[i]=hash(i,4);}
  const snowGeo=new THREE.BufferGeometry();snowGeo.setAttribute('position',new THREE.BufferAttribute(positions,3));snowGeo.setAttribute('seed',new THREE.BufferAttribute(seeds,1));
  const snowMat=new THREE.ShaderMaterial({uniforms:{time:{value:0},pixelRatio:{value:renderer.getPixelRatio()}},transparent:true,depthWrite:false,
    vertexShader:'uniform float time;uniform float pixelRatio;attribute float seed;varying float alpha;void main(){vec3 p=position;p.y=mod(p.y+10.-time*(.28+seed*.4),35.)-10.;p.x=mod(p.x+35.+time*(1.8+seed*1.5)+sin(time*.4+seed*90.)*1.4,70.)-35.;vec4 mv=modelViewMatrix*vec4(p,1.);gl_Position=projectionMatrix*mv;gl_PointSize=clamp((42.+seed*85.)/-mv.z,1.,6.)*pixelRatio;alpha=(.25+seed*.55)*smoothstep(0.,5.,-mv.z);}',
    fragmentShader:'varying float alpha;void main(){vec2 p=gl_PointCoord-.5;p.y*=1.7;float d=length(p);gl_FragColor=vec4(.93,.97,1.,alpha*smoothstep(.5,.05,d));}'
  });const snowPoints=new THREE.Points(snowGeo,snowMat);scene.add(snowPoints);

  const composer=new EffectComposer(renderer);composer.addPass(new RenderPass(scene,camera));
  const bloom=new UnrealBloomPass(new THREE.Vector2(800,600),.12,.45,2.5);composer.addPass(bloom);composer.addPass(new OutputPass());
  let width=0,height=0,progress=0,rotation=0,pitch=0,paused=reduced.matches,visible=true,clock=0,last=performance.now(),raf;
  let pointerX=0,pointerY=0,parallaxX=0,parallaxY=0;
  let sunset=false,warmth=0,details=false;
  let suitcase=null,travelTime=-1,travelPending=false;

  const travelButton=document.querySelector('#travel-toggle'),travelStatus=document.querySelector('#travel-status');
  async function travel(){
    if(travelTime>=0||travelPending)return;
    travelPending=true;travelButton.disabled=true;travelButton.textContent='PACKING THE GLASSES…';
    try{suitcase??=await makeSuitcase();scene.add(suitcase.root);rotation=0;pitch=0;}
    catch(error){travelPending=false;travelButton.disabled=false;travelButton.textContent='ABOUT / PACK UP ↗';travelStatus.textContent='The case could not load. Use the About link to continue.';console.error(error);onPacked();}
  }
  const dayFog=new THREE.Color('#b5c5d6'),eveningFog=new THREE.Color('#c6b7ac');
  const daySun=new THREE.Color('#fff2df'),eveningSun=new THREE.Color('#ffc388');
  const nightSky=new THREE.Color('#a9bfdf'),daySky=new THREE.Color('#dfeaff');
  const hello=document.querySelector('#yeti-hello');
  const detailButtons=[...document.querySelectorAll('[data-detail]')];
  const detailPositions=[new THREE.Vector3(-.36,.82,2.64),new THREE.Vector3(1.68,-.05,-.65),new THREE.Vector3(-1.43,-1.45,-.65)];
  const pin=new THREE.Vector3();
  const states=[{x:1.0,y:.15,s:1,ry:-.42,rx:.05,cy:1.8,cz:12.6},{x:-1.55,y:.1,s:1.14,ry:.48,rx:.03,cy:2.3,cz:12.4},{x:-1.45,y:.25,s:1.02,ry:2.45,rx:.19,cy:3.2,cz:12.8},{x:-1.4,y:.4,s:.95,ry:5.53,rx:.06,cy:2,cz:12.6},{x:0,y:1.15,s:1.12,ry:Math.PI*2,rx:0,cy:1.6,cz:10.5}];
  const projected=new THREE.Vector3();
  const lensCenter=new THREE.Vector3(),lensScreen=new THREE.Vector3(),lensEdge=new THREE.Vector3(),lookTarget=new THREE.Vector3();
  const portal=document.querySelector('#conversation'),portalRing=document.querySelector('#lens-ring'),flare=document.querySelector('#lens-flare');
  const annotation=document.querySelector('#annotation');
  function resize(){const w=canvas.clientWidth,h=canvas.clientHeight;if(w===width&&h===height)return;width=w;height=h;renderer.setSize(w,h,false);composer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}
  new ResizeObserver(resize).observe(canvas);resize();
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;},{rootMargin:'100px'}).observe(canvas);
  function render(now){
    raf=requestAnimationFrame(render);const dt=Math.min((now-last)/1000,.05);last=now;if(!visible||document.hidden)return;
    if(!paused)clock+=dt;
    if(travelPending&&suitcase&&progress<.025){travelPending=false;travelTime=0;document.querySelector('.world').classList.add('is-travelling');}
    if(travelTime>=0)travelTime+=dt*(reduced.matches?3:1);
    const p=progress*6,idx=Math.min(3,Math.floor(p)),u=THREE.MathUtils.smoothstep(p-idx,0,1),a=states[idx],b=states[idx+1],mix=k=>THREE.MathUtils.lerp(a[k],b[k],u),mobile=width<650;
    const displayScale=mobile?.64-.10*THREE.MathUtils.smoothstep(p,0,.8):Math.min(1,width/1100);
    const x=mobile?0:mix('x');
    const y=mobile?(p<.35?-.35:THREE.MathUtils.lerp(-.35,1.6,Math.min(1,(p-.35)*2))):mix('y');
    const exploration=1-THREE.MathUtils.smoothstep(p,3.5,4.1);
    product.position.set(x,y+Math.sin(clock*.65)*.055*exploration,0);
    product.scale.setScalar(mix('s')*displayScale);
    product.rotation.set(mix('rx')+pitch*exploration,mix('ry')+rotation*exploration,Math.sin(clock*.3)*.012*exploration);
    if(travelTime>=0){
      const t=travelTime,pack=THREE.MathUtils.smoothstep(t,.4,1.7);
      suitcase.root.visible=true;suitcase.root.position.set(x,-1.6,0);suitcase.root.rotation.y=-.2;suitcase.root.scale.setScalar(displayScale);
      suitcase.hinge.rotation.x=-1.8*(1-THREE.MathUtils.smoothstep(t,1.6,2.6));
      product.position.y-=pack*.9;product.scale.multiplyScalar(1-pack*.52);
      product.scale.multiplyScalar(1-.15*Math.sin(Math.min(t/6,1)*Math.PI));
      product.rotation.y+=Math.sin(Math.min(t/6,1)*Math.PI)*.35;
      product.visible=t<2.55;
      travelStatus.textContent=t<2.6?'PACKING THE GLASSES':'OPENING ABOUT';
      if(t>3.7){travelTime=-1;onPacked();}
    }
    parallaxX+=(pointerX-parallaxX)*(1-Math.exp(-dt*3));parallaxY+=(pointerY-parallaxY)*(1-Math.exp(-dt*3));
    camera.position.set(Math.sin(Math.min(p,4)/4*Math.PI)*.85+(paused?0:parallaxX*.48*exploration),mix('cy')+(paused?0:parallaxY*.18*exploration),mix('cz'));
    lensCenter.set(-.36,.82,2.622);product.localToWorld(lensCenter);
    const approach=reduced.matches?0:THREE.MathUtils.smoothstep(p,4.15,5.65);
    camera.position.lerp(new THREE.Vector3(lensCenter.x,lensCenter.y,lensCenter.z+.055),approach);
    lookTarget.set(0,.1,0).lerp(lensCenter,THREE.MathUtils.smoothstep(p,4.0,4.85));camera.lookAt(lookTarget);camera.updateMatrixWorld();
    lensScreen.copy(lensCenter).project(camera);
    const portalProgress=THREE.MathUtils.smoothstep(p,4.5,5.65),cx=(lensScreen.x*.5+.5)*width,cy=(-lensScreen.y*.5+.5)*height;
    lensEdge.set(-.221,.82,2.622);product.localToWorld(lensEdge);lensEdge.project(camera);
    const opticalRadius=Math.abs(lensEdge.x-lensScreen.x)*width*.5;
    const radius=opticalRadius*THREE.MathUtils.smoothstep(p,4.4,4.7);
    portal.style.opacity=p>4.4?(reduced.matches?portalProgress:1):0;
    portal.style.clipPath=reduced.matches||p>5.63?'none':`circle(${radius}px at ${cx}px ${cy}px)`;
    portal.style.setProperty('--cafe-scale',1.16-portalProgress*.16);
    portalRing.style.left=`${cx}px`;portalRing.style.top=`${cy}px`;portalRing.style.width=portalRing.style.height=`${radius*2}px`;portalRing.style.opacity=portalProgress>0&&portalProgress<.85?Math.sin(portalProgress*Math.PI)*.7:0;
    flare.style.opacity=reduced.matches?0:Math.sin(portalProgress*Math.PI)*.14;
    warmth+=(Number(sunset)-warmth)*(1-Math.exp(-dt*1.8));
    scene.fog.color.copy(dayFog).lerp(eveningFog,warmth);scene.fog.density=.0105;
    sun.color.copy(daySun).lerp(eveningSun,warmth);sun.position.y=19-warmth*8;sun.intensity=2.8-warmth*.4;
    skyLight.color.copy(daySky).lerp(nightSky,warmth);
    atmosphere.update(clock,warmth);
    wind.update(clock,.7+Math.sin(portalProgress*Math.PI)*1.15);
    yeti.update(clock,parallaxX);
    yeti.root.visible=!mobile&&p<3.6;
    pin.copy(yeti.root.position);pin.y-=.2;pin.project(camera);hello.hidden=!yeti.root.visible||p>.6;
    const helloWidth=hello.offsetWidth||120;
    hello.style.left=`${Math.min((pin.x*.5+.5)*width+helloWidth*.4,width-helloWidth*.5-16)}px`;hello.style.top=`${(-pin.y*.5+.5)*height}px`;
    detailButtons.forEach((button,i)=>{button.hidden=travelTime>=0||!details||p>3.6||(i===2&&p<2.35);pin.copy(detailPositions[i]);product.localToWorld(pin);pin.project(camera);button.style.left=`${(pin.x*.5+.5)*width}px`;button.style.top=`${(-pin.y*.5+.5)*height}px`;});
    earbuds.visible=p>2.35; snowMat.uniforms.time.value=clock;
    projected.set(p<1.6?-.36:1.65,p<1.6?.82:0,p<1.6?2.63:-1.3);product.localToWorld(projected);projected.project(camera);
    const reverse=p>1.6;annotation.classList.toggle('reverse',reverse);
    annotation.style.transform=`translate(${reverse?'calc(':''}${(projected.x*.5+.5)*width}px${reverse?' - 100%)':''},${(-projected.y*.5+.5)*height}px)`;
    annotation.style.opacity=String(p>3.5?0:.7);document.querySelector('#rotation-reading').textContent=`ROT. ${Math.round(THREE.MathUtils.radToDeg(product.rotation.y)).toString().padStart(3,'0')}°`;
    if(portalProgress<1)composer.render();
  }
  annotation.style.left='0';annotation.style.top='0';
  let drag=null;
  canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,rotation,pitch};canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointermove',e=>{if(e.pointerType==='mouse'){pointerX=e.clientX/width-.5;pointerY=e.clientY/height-.5;}if(!drag)return;rotation=drag.rotation+(e.clientX-drag.x)*.008;if(e.pointerType==='mouse')pitch=THREE.MathUtils.clamp(drag.pitch+(e.clientY-drag.y)*.003,-.5,.5);});
  canvas.addEventListener('pointerleave',()=>{pointerX=0;pointerY=0;});
  const release=()=>drag=null;canvas.addEventListener('pointerup',release);canvas.addEventListener('pointercancel',release);
  canvas.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();rotation+=(e.key==='ArrowLeft'?-.25:.25);}if(e.key==='Home'){rotation=0;pitch=0;}});
  let compileTimer;
  try {
    await Promise.race([renderer.compileAsync(scene,camera),new Promise(resolve=>{compileTimer=setTimeout(resolve,5000);})]);
  } finally {clearTimeout(compileTimer);}
  composer.render();
  raf=requestAnimationFrame(render);onReady();
  return {travel,setProgress(p){if(travelTime<0)progress=p;},rotate(delta){rotation+=delta;},reset(){rotation=0;pitch=0;},setDetails(value){details=value;},setSunset(value){sunset=value;},greet(){yeti.greet(clock-(paused?.5:0));},setPaused(value){paused=value;},get paused(){return paused;},dispose(){cancelAnimationFrame(raf);renderer.dispose();composer.dispose();}};
}








