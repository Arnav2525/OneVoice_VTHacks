const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const root = __dirname;
const port = Number(process.env.ONEVOICE_PORT || 4319);
const types = {'.gltf':'model/gltf+json','.bin':'application/octet-stream','.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg','.stl':'model/stl','.ovm':'application/octet-stream','.woff2':'font/woff2'};
const server = http.createServer((req,res) => {
  if(!['GET','HEAD'].includes(req.method)){res.writeHead(405);return res.end('Method not allowed');}
  let pathname;try{pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);}catch{res.writeHead(400);return res.end('Invalid path');}
  const file=path.resolve(root,'.'+(pathname==='/'?'/index.html':pathname));
  if(!file.startsWith(root+path.sep)||!types[path.extname(file)]){res.writeHead(404);return res.end('Not found');}
  const compressed=path.extname(file)==='.ovm'&&/\bgzip\b/.test(req.headers['accept-encoding']||'')&&fs.existsSync(file+'.gz');
  const servedFile=compressed?file+'.gz':file;
  fs.stat(servedFile,(error,stat)=>{
    if(error||!stat.isFile()){res.writeHead(404);return res.end('Not found');}
    const cacheable=['.ovm','.png','.jpg'].includes(path.extname(file));
    res.writeHead(200,{'Content-Type':types[path.extname(file)],'Content-Length':stat.size,'Cache-Control':cacheable?'private, max-age=3600':'no-cache','X-Content-Type-Options':'nosniff',...(compressed?{'Content-Encoding':'gzip','Vary':'Accept-Encoding'}:{})});
    if(req.method==='HEAD')return res.end();
    const stream=fs.createReadStream(servedFile);stream.on('error',()=>res.destroy());stream.pipe(res);
  });
});
server.on('error',error=>{console.error(error.code==='EADDRINUSE'?`Port ${port} is already running. Open http://127.0.0.1:${port}/`:error.message);process.exitCode=1;});
server.listen(port,'127.0.0.1',()=>console.log(`One Voice Alpine is ready at http://127.0.0.1:${port}/`));

