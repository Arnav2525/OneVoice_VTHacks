import * as THREE from 'three';

export function makeWind(scene){
  const uniforms={time:{value:0},strength:{value:1}};
  const material=new THREE.ShaderMaterial({uniforms,transparent:true,depthWrite:false,side:THREE.DoubleSide,
    vertexShader:'varying vec2 vUv;uniform float time;void main(){vUv=uv;vec3 p=position;p.y+=sin(p.x*.22+time*.35)*.3;gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.);}',
    fragmentShader:`varying vec2 vUv;uniform float time;uniform float strength;
      float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
      float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);}
      void main(){vec2 uv=vUv;float n=noise(vec2(uv.x*4.-time*.2,uv.y*5.+time*.08))*.65+noise(vec2(uv.x*10.-time*.5,uv.y*9.))*.35;float mask=pow(max(0.,sin(uv.x*3.14159)*sin(uv.y*3.14159)),1.2);float wisps=smoothstep(.30,.77,n)*mask;gl_FragColor=vec4(.88,.92,1.,wisps*.07*strength);}`
  });
  const layers=[];
  for(const[x,y,z,w,h]of [[-8,-1.0,-10,26,3.5],[5,-.5,-23,40,6],[-12,1.8,-48,65,10]]){const mesh=new THREE.Mesh(new THREE.PlaneGeometry(w,h,20,4),material);mesh.position.set(x,y,z);scene.add(mesh);layers.push(mesh);}
  return {update(time,strength){uniforms.time.value=time;uniforms.strength.value=strength;layers[0].position.x=Math.sin(time*.09)*6;layers[1].position.x=5+Math.sin(time*.05+1)*9;}};
}

