import * as THREE from 'three';
import {makeYeti} from './yeti.js';

const canvas=document.querySelector('#about-yeti-canvas');

try{
  const renderer=new THREE.WebGLRenderer({canvas,alpha:true,antialias:true,powerPreference:'low-power'});
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
  renderer.setClearColor(0x000000,0);
  renderer.toneMapping=THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure=1.4;
  const scene=new THREE.Scene();
  scene.add(new THREE.HemisphereLight('#f5fbff','#7b94a2',2.8));
  const key=new THREE.DirectionalLight('#ffffff',2.3);key.position.set(-3,7,8);scene.add(key);
  const camera=new THREE.PerspectiveCamera(33,1,.1,40);
  camera.position.set(0,1.45,7.7);camera.lookAt(0,1.35,0);
  const yeti=makeYeti();yeti.root.position.set(0,-.85,0);yeti.root.rotation.y=-.24;scene.add(yeti.root);
  const resize=()=>{const width=canvas.clientWidth,height=canvas.clientHeight;if(!width||!height)return;renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();};
  new ResizeObserver(resize).observe(canvas);resize();
  const start=performance.now();let frame;
  function draw(now){
    const time=(now-start)/1000;
    if(!document.hidden){
      yeti.update(time);
      renderer.render(scene,camera);
    }
    frame=requestAnimationFrame(draw);
  }
  frame=requestAnimationFrame(draw);
  addEventListener('pagehide',()=>{cancelAnimationFrame(frame);renderer.dispose();},{once:true});
}catch(error){
  console.warn('The About yeti could not render',error);
  canvas.hidden=true;
}
