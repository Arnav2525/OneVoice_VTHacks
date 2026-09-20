import * as THREE from 'three';

export function makeYeti() {
  const root = new THREE.Group(), body = new THREE.Group(); root.add(body);
  const snow = new THREE.MeshStandardMaterial({color:'#aabacb',roughness:.94,envMapIntensity:.25});
  const muzzle = new THREE.MeshStandardMaterial({color:'#9aafbf',roughness:.9});
  const frame = new THREE.MeshStandardMaterial({color:'#233e53',roughness:.48,metalness:.3});
  const eyes = new THREE.MeshStandardMaterial({color:'#132635',roughness:.18});
  const horn = new THREE.MeshStandardMaterial({color:'#d9c9a8',roughness:.82});
  const white = new THREE.MeshBasicMaterial({color:'#f4f9ff'});
  const sphere = new THREE.SphereGeometry(1,28,20);
  function oval(parent,pos,scale,material=snow){
    const m=new THREE.Mesh(sphere,material);m.position.set(...pos);m.scale.set(...scale);m.castShadow=true;m.receiveShadow=true;parent.add(m);return m;
  }
  const outline=[[.05,.4],[.38,.42],[.64,.65],[.74,1],[.70,1.4],[.59,1.8],[.48,2],[.05,2.08]].map(([r,y])=>new THREE.Vector2(r,y));
  const torso=new THREE.Mesh(new THREE.LatheGeometry(outline,40),snow);torso.scale.z=.76;torso.castShadow=true;torso.receiveShadow=true;body.add(torso);
  const head=new THREE.Group();head.position.set(0,2.13,.07);body.add(head);
  oval(head,[0,0,0],[.62,.62,.51]);
  oval(head,[0,-.08,.41],[.43,.31,.105],muzzle);
  oval(head,[0,-.17,.50],[.27,.17,.085]);
  oval(head,[0,-.045,.535],[.09,.065,.06],frame);
  const smileCurve=new THREE.QuadraticBezierCurve3(new THREE.Vector3(-.12,-.19,.587),new THREE.Vector3(0,-.29,.61),new THREE.Vector3(.12,-.19,.587));
  head.add(new THREE.Mesh(new THREE.TubeGeometry(smileCurve,18,.011,6,false),frame));
  for(const side of [-1,1]){
    oval(body,[side*.33,.39,.03],[.29,.38,.30]);oval(body,[side*.34,.10,.18],[.3,.15,.42]);
    oval(head,[side*.55,.08,-.03],[.16,.21,.12]);
    oval(head,[side*.18,.025,.511],[.045,.065,.022],eyes);
    oval(head,[side*.18-.011,.045,.536],[.012,.018,.008],white);
    const h=new THREE.Mesh(new THREE.ConeGeometry(.09,.25,16),horn);h.position.set(side*.44,.49,.0);h.rotation.z=-side*.28;head.add(h);
    const ring=new THREE.Mesh(new THREE.TorusGeometry(.185,.026,8,40),frame);ring.position.set(side*.2,.015,.54);ring.scale.set(1,.82,1);head.add(ring);
  }
  const bridge=new THREE.Mesh(new THREE.BoxGeometry(.085,.029,.035),frame);bridge.position.set(0,.02,.54);head.add(bridge);
  const littleCamera=new THREE.Mesh(new THREE.BoxGeometry(.40,.15,.10),frame);littleCamera.position.set(0,.31,.47);head.add(littleCamera);
  const optic=new THREE.Mesh(new THREE.CylinderGeometry(.039,.039,.015,20),eyes);optic.rotation.x=Math.PI/2;optic.position.set(-.065,.31,.529);head.add(optic);
  for(let i=0;i<5;i++){const tuft=oval(head,[(i-2)*.1,.55+Math.sin(i)*.015,.04],[.09,.115,.12]);tuft.rotation.z=(i-2)*-.12;}
  const arms=[];
  for(const side of [-1,1]){
    const arm=new THREE.Group();arm.position.set(side*.57,1.72,0);body.add(arm);arm.rotation.z=side*.18;
    oval(arm,[side*.06,-.37,.03],[.235,.52,.25]);oval(arm,[side*.09,-.78,.1],[.23,.25,.22]);arms.push(arm);
  }
  let waveStart=-100;
  root.rotation.y=-.24;
  return {root,greet(time){waveStart=time;},update(time,pointer=0){
    const age=time-waveStart,envelope=age>=0&&age<2.8?Math.sin(Math.min(1,age/.35)*Math.PI/2)*Math.min(1,(2.8-age)/.45):0;
    body.rotation.z=Math.sin(time*.6)*.012;head.rotation.y=Math.sin(time*.22)*.07+pointer*.15;
    arms[1].rotation.z=.18+envelope*(2.2+Math.sin(age*9)*.19);arms[0].rotation.x=Math.sin(time*.7)*.04;
    body.position.y=Math.sin(time*.9)*.018;
  }};
}

