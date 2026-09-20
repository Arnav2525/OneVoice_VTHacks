import * as THREE from 'three';
import {GLTFLoader} from './vendor/addons/loaders/GLTFLoader.js';
import {hash} from './terrain.js';

export async function makeForest(){
  const root=new THREE.Group();root.visible=false;
  const loader=new THREE.TextureLoader();
  const [map,normalMap]=await Promise.all(['rock-diffuse.jpg','rock-normal.jpg'].map(n=>loader.loadAsync('assets/terrain/'+n)));
  for(const t of [map,normalMap]){t.wrapS=t.wrapT=THREE.RepeatWrapping;t.repeat.set(1.5,1.5);}
  map.colorSpace=THREE.SRGBColorSpace;
  const rockMaterial=new THREE.MeshStandardMaterial({color:'#454e38',map,normalMap,normalScale:new THREE.Vector2(.6,.6),roughness:.93,envMapIntensity:.25});
  const rockGeometry=new THREE.SphereGeometry(1,64,40);
  const rp=rockGeometry.attributes.position;for(let i=0;i<rp.count;i++){const x=rp.getX(i),y=rp.getY(i),z=rp.getZ(i),s=1+Math.sin(x*9+z*4)*Math.cos(y*7)*.08;rp.setXYZ(i,x*s,y*s,z*s);}rockGeometry.computeVertexNormals();
  const pedestal=new THREE.Mesh(rockGeometry,rockMaterial);pedestal.scale.set(3.3,2.2,2.4);pedestal.position.set(1,-3.2,0);pedestal.castShadow=pedestal.receiveShadow=true;root.add(pedestal);
  const motesGeometry=new THREE.BufferGeometry(),positions=new Float32Array(180*3);for(let i=0;i<positions.length;i++)positions[i]=(hash(i,300)-.5)*35;motesGeometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
  const motes=new THREE.Points(motesGeometry,new THREE.PointsMaterial({color:'#dfd396',size:.035,transparent:true,opacity:.45,depthWrite:false}));root.add(motes);
  const forestPlate=await loader.loadAsync('assets/forest-clearing.png');forestPlate.colorSpace=THREE.SRGBColorSpace;
  const backdrop=new THREE.Mesh(new THREE.PlaneGeometry(82,46.2),new THREE.MeshBasicMaterial({map:forestPlate,fog:false,toneMapped:false}));backdrop.position.set(0,1,-31);root.add(backdrop);
  return {root,pedestal,update(time,p){motes.rotation.y=time*.007;pedestal.visible=p<3.6;}};
}

export async function makeSuitcase(){
  const asset=await new GLTFLoader().loadAsync('assets/suitcase/case.gltf');
  const root=new THREE.Group(),body=new THREE.Group();root.add(body);
  for(const node of [...asset.scene.children])if(node.name.includes('_01_'))body.add(node);
  body.rotation.x=-Math.PI/2;body.updateMatrixWorld(true);
  const box=new THREE.Box3().setFromObject(body),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3()),scale=4.8/size.x;
  body.scale.setScalar(scale);body.position.set(-center.x*scale,-box.min.y*scale,-center.z*scale);root.updateMatrixWorld(true);
  const top=body.getObjectByName('vintage_suitcase_01_top'),topBox=new THREE.Box3().setFromObject(top),hinge=new THREE.Group();
  hinge.position.set(0,topBox.min.y,topBox.min.z);root.add(hinge);root.updateMatrixWorld(true);hinge.attach(top);
  root.traverse(mesh=>{if(mesh.isMesh){mesh.castShadow=mesh.receiveShadow=true;mesh.material.envMapIntensity=.6;}});
  root.visible=false;return {root,hinge};
}
