import * as THREE from 'three';

const fract=n=>n-Math.floor(n);
export const hash=(x,y)=>fract(Math.sin(x*127.1+y*311.7)*43758.5453123);
function noise(x,y){const a=Math.floor(x),b=Math.floor(y),u=x-a,v=y-b,s=u*u*(3-2*u),t=v*v*(3-2*v);return THREE.MathUtils.lerp(THREE.MathUtils.lerp(hash(a,b),hash(a+1,b),s),THREE.MathUtils.lerp(hash(a,b+1),hash(a+1,b+1),s),t);}
function fbm(x,z){let n=0,amp=.5;for(let i=0;i<6;i++){n+=noise(x,z)*amp;x=x*2.03+11.2;z=z*2.03+4.1;amp*=.5;}return n;}
const peaks=[[-28,-43,19,17],[21,-59,25,21],[-5,-86,25,20],[49,-67,25,19],[-60,-72,27,22],[34,-24,13,15],[-36,-20,11,16]];
export function elevation(x,z){
  let mountain=0;for(const[px,pz,h,s]of peaks)mountain+=h*Math.exp(-((x-px)**2+(z-pz)**2)/(s*s));
  const ridge=1-Math.abs(noise(x*.14,z*.14)*2-1),clearing=THREE.MathUtils.smoothstep(Math.hypot(x,z*.65),10,27);
  const ledge=2.8*Math.exp(-(x*x/18+(z-.5)**2/30));
  const foreground=3.3*Math.exp(-((x+7)**2/10+(z-2)**2/23))+4.4*Math.exp(-((x-9)**2/15+(z+3)**2/30));
  const crags=fbm(x*.8+fbm(x*.2,z*.2),z*.8)*.9;
  return -4.6+ledge+foreground*(.5+fbm(x*.4,z*.4)) + clearing*mountain*(.27+.32*fbm(x*.09,z*.09)+ridge*.22) +crags;
}
export async function makeTerrain(scene,renderer){
  const loader=new THREE.TextureLoader();
  const [rock,snow,rockNormal,snowNormal]=await Promise.all(['rock-diffuse.jpg','snow-diffuse.jpg','rock-normal.jpg','snow-normal.jpg'].map(name=>loader.loadAsync('assets/terrain/'+name)));
  for(const texture of [rock,snow,rockNormal,snowNormal]){texture.wrapS=texture.wrapT=THREE.RepeatWrapping;texture.anisotropy=Math.min(8,renderer.capabilities.getMaxAnisotropy());}
  rock.colorSpace=snow.colorSpace=THREE.SRGBColorSpace;
  const geometry=new THREE.PlaneGeometry(200,200,420,420);geometry.rotateX(-Math.PI/2);
  const positions=geometry.attributes.position;
  for(let i=0;i<positions.count;i++){
    let x=positions.getX(i),z=positions.getZ(i);
    x=Math.sign(x)*Math.pow(Math.abs(x)/100,1.8)*100;
    z=Math.sign(z)*Math.pow(Math.abs(z)/100,1.65)*(z<0?155:36);
    positions.setXYZ(i,x,elevation(x,z),z);
  }geometry.computeVertexNormals();
  const material=new THREE.MeshStandardMaterial({color:'#d5dce4',roughness:.86,metalness:0});
  material.onBeforeCompile=shader=>{
    Object.assign(shader.uniforms,{rockMap:{value:rock},snowMap:{value:snow},rockN:{value:rockNormal},snowN:{value:snowNormal}});
    shader.vertexShader=shader.vertexShader.replace('#include <common>','#include <common>\nvarying vec3 vTerrain;varying vec3 vTerrainNormal;').replace('#include <begin_vertex>','#include <begin_vertex>\nvTerrain=position;vTerrainNormal=normal;');
    shader.fragmentShader=shader.fragmentShader.replace('#include <common>',`#include <common>
      varying vec3 vTerrain; varying vec3 vTerrainNormal;
      uniform sampler2D rockMap;uniform sampler2D snowMap;uniform sampler2D rockN;uniform sampler2D snowN;
      float hash3(vec3 p){return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453);}
      float grain(vec3 p){vec3 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(mix(hash3(i),hash3(i+vec3(1,0,0)),f.x),mix(hash3(i+vec3(0,1,0)),hash3(i+vec3(1,1,0)),f.x),f.y),mix(mix(hash3(i+vec3(0,0,1)),hash3(i+vec3(1,0,1)),f.x),mix(hash3(i+vec3(0,1,1)),hash3(i+vec3(1,1,1)),f.x),f.y),f.z);}
      vec3 triplanar(sampler2D tex,vec3 p,vec3 weights){return texture2D(tex,p.yz).rgb*weights.x+texture2D(tex,p.xz).rgb*weights.y+texture2D(tex,p.xy).rgb*weights.z;}
    `).replace('#include <color_fragment>',`#include <color_fragment>
      vec3 terrainWeights=pow(abs(normalize(vTerrainNormal)),vec3(4.));terrainWeights/=dot(terrainWeights,vec3(1.));
      float patches=grain(vTerrain*.37)*.6+grain(vTerrain*1.4)*.25;
      float snowAmount=smoothstep(.72,.985,normalize(vTerrainNormal).y+patches*.18-.08);
      vec3 stone=triplanar(rockMap,vTerrain*.20,terrainWeights);
      float stoneValue=dot(stone,vec3(.2126,.7152,.0722));
      stone=mix(stone,vec3(stoneValue),.75)*vec3(.69,.76,.86);
      vec3 powder=triplanar(snowMap,vTerrain*.40,terrainWeights)*vec3(.94,.96,1.);
      diffuseColor.rgb*=mix(stone*.80,powder*1.06,snowAmount);
    `).replace('#include <normal_fragment_maps>',`#include <normal_fragment_maps>
      vec2 surfaceUV=vTerrain.xz*.25;
      vec3 detail=mix(texture2D(rockN,surfaceUV).xyz,texture2D(snowN,surfaceUV).xyz,snowAmount)*2.-1.;
      vec3 q1=dFdx(-vViewPosition),q2=dFdy(-vViewPosition);vec2 st1=dFdx(surfaceUV),st2=dFdy(surfaceUV);
      vec3 S=normalize(q1*st2.t-q2*st1.t),T=normalize(-q1*st2.s+q2*st1.s);
      normal=normalize(mat3(S,T,normal)*vec3(detail.xy*.8,detail.z));
    `);
  };
  const mesh=new THREE.Mesh(geometry,material);mesh.receiveShadow=true;mesh.castShadow=true;scene.add(mesh);
  return mesh;
}
