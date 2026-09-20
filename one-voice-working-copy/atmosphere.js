import * as THREE from 'three';

export function makeAtmosphere(scene){
  const uniforms={time:{value:0},warmth:{value:0}};
  const sky=new THREE.Mesh(new THREE.SphereGeometry(160,32,20),new THREE.ShaderMaterial({
    side:THREE.BackSide,depthWrite:false,uniforms,
    vertexShader:'varying vec3 direction;void main(){direction=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
    fragmentShader:`varying vec3 direction;uniform float time;uniform float warmth;
    float noise(vec2 p){vec2 a=floor(p),f=fract(p);f=f*f*(3.-2.*f);float x=dot(a,vec2(127.1,311.7));return mix(mix(fract(sin(x)*43758.5453),fract(sin(x+127.1)*43758.5453),f.x),mix(fract(sin(x+311.7)*43758.5453),fract(sin(x+438.8)*43758.5453),f.x),f.y);}
    void main(){vec3 d=normalize(direction);float elevation=smoothstep(-.05,.65,d.y);
      vec3 horizon=mix(vec3(.77,.82,.86),vec3(.90,.66,.44),warmth);
      vec3 zenith=mix(vec3(.27,.43,.61),vec3(.22,.30,.47),warmth);
      vec3 color=mix(horizon,zenith,elevation);
      vec3 sun=normalize(vec3(-.6,.27,-.8));float glow=pow(max(0.,dot(d,sun)),35.);
      color+=vec3(.24,.19,.10)*glow*(.5+warmth);
      vec2 uv=d.xz/max(.12,d.y+.4)*2.5+vec2(time*.004,0.);
      float n=noise(uv)*.55+noise(uv*2.1)*.28+noise(uv*4.2)*.17;
      float clouds=smoothstep(.49,.79,n)*smoothstep(-.02,.25,d.y)*.23;
      color=mix(color,vec3(.9,.92,.94),clouds);gl_FragColor=vec4(color,1.);
      #include <tonemapping_fragment>
      #include <colorspace_fragment>
    }`
  }));scene.add(sky);
  return {update(time,warmth){uniforms.time.value=time;uniforms.warmth.value=warmth;}};
}
