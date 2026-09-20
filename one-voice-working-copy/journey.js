import * as THREE from 'three';
import {GLTFLoader} from './vendor/addons/loaders/GLTFLoader.js';

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
