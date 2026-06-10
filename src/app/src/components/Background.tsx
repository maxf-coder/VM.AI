import { useEffect, useRef, useState } from "react";

const DEV_MODE = false; // damn

const DEFAULTS = {
    scale: 1.,
    speed: 0.024,
    lines: 8.0,
    glow: 18.0,
};

const RANGES = {
    scale: { min: 0.5, max: 10, step: 0.01 },
    speed: { min: 0.001, max: 0.3, step: 0.001 },
    lines: { min: 1, max: 30, step: 0.1 },
    glow: { min: 1, max: 60, step: 0.1 },
};

export default function Background() {
    const canvasRef = useRef(null);
    const configRef = useRef(DEFAULTS);
    const rafRef = useRef(null);
    const [config, setConfig] = useState(DEFAULTS);
    const [open, setOpen] = useState(false);
    const [dragging, setDragging] = useState(null);

    useEffect(() => { configRef.current = config; }, [config]);

    useEffect(() => {
        if (!dragging) return;
        const onMove = (e) => {
            const dx = (e.clientX - dragging.startX) * 0.005;
            const r = RANGES[dragging.key];
            const next = Math.min(r.max, Math.max(r.min, dragging.startVal + dx * (r.max - r.min)));
            setConfig(p => ({ ...p, [dragging.key]: parseFloat(next.toFixed(3)) }));
        };
        const onUp = () => setDragging(null);
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
        return () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
        };
    }, [dragging]);

    useEffect(() => {
        const canvas = canvasRef.current;
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        const gl = canvas.getContext("webgl");
        if (!gl) return;

        const vertSrc = `
            attribute vec2 a_position;
            void main() { gl_Position = vec4(a_position, 0.0, 1.0); }
        `;

        const fragSrc = `
            precision highp float;
            uniform vec2  u_resolution;
            uniform float u_time;
            uniform float u_scale;
            uniform float u_speed;
            uniform float u_lines;
            uniform float u_glow;
            uniform vec3  u_bg;
            uniform vec3  u_fg;

            vec3 mod289(vec3 x){return x-floor(x*(1./289.))*289.;}
            vec4 mod289(vec4 x){return x-floor(x*(1./289.))*289.;}
            vec4 permute(vec4 x){return mod289(((x*34.)+1.)*x);}
            vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}

            float snoise(vec3 v){
                const vec2 C=vec2(1./6.,1./3.);
                const vec4 D=vec4(0.,.5,1.,2.);
                vec3 i=floor(v+dot(v,C.yyy));
                vec3 x0=v-i+dot(i,C.xxx);
                vec3 g=step(x0.yzx,x0.xyz);
                vec3 l=1.-g;
                vec3 i1=min(g.xyz,l.zxy);
                vec3 i2=max(g.xyz,l.zxy);
                vec3 x1=x0-i1+C.xxx;
                vec3 x2=x0-i2+C.yyy;
                vec3 x3=x0-D.yyy;
                i=mod289(i);
                vec4 p=permute(permute(permute(
                    i.z+vec4(0.,i1.z,i2.z,1.))
                    +i.y+vec4(0.,i1.y,i2.y,1.))
                    +i.x+vec4(0.,i1.x,i2.x,1.));
                float n_=0.142857142857;
                vec3 ns=n_*D.wyz-D.xzx;
                vec4 j=p-49.*floor(p*ns.z*ns.z);
                vec4 x_=floor(j*ns.z);
                vec4 y_=floor(j-7.*x_);
                vec4 x=x_*ns.x+ns.yyyy;
                vec4 y=y_*ns.x+ns.yyyy;
                vec4 h=1.-abs(x)-abs(y);
                vec4 b0=vec4(x.xy,y.xy);
                vec4 b1=vec4(x.zw,y.zw);
                vec4 s0=floor(b0)*2.+1.;
                vec4 s1=floor(b1)*2.+1.;
                vec4 sh=-step(h,vec4(0.));
                vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy;
                vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
                vec3 p0=vec3(a0.xy,h.x);
                vec3 p1=vec3(a0.zw,h.y);
                vec3 p2=vec3(a1.xy,h.z);
                vec3 p3=vec3(a1.zw,h.w);
                vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
                p0*=norm.x;p1*=norm.y;p2*=norm.z;p3*=norm.w;
                vec4 m=max(.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.);
                m=m*m;
                return 42.*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
            }

            void main(){
                vec2 uv=gl_FragCoord.xy/u_resolution;
                uv.y=1.-uv.y;
                float t=u_time*u_speed;
                float n=snoise(vec3(uv*u_scale,t));
                n+=.5*snoise(vec3(uv*u_scale*2.,t*1.3));
                n+=.25*snoise(vec3(uv*u_scale*4.,t*1.7));
                n/=1.75;
                float bands=fract(n*u_lines);
                float line=abs(bands-.5)*2.;
                line=1.-line;
                line=pow(line,u_glow);
                vec3 color=mix(u_bg,u_fg,line);
                gl_FragColor=vec4(color,1.);
            }
        `;

        const compile = (type, src) => {
            const s = gl.createShader(type);
            gl.shaderSource(s, src);
            gl.compileShader(s);
            return s;
        };

        const prog = gl.createProgram();
        gl.attachShader(prog, compile(gl.VERTEX_SHADER, vertSrc));
        gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fragSrc));
        gl.linkProgram(prog);
        gl.useProgram(prog);

        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1
        ]), gl.STATIC_DRAW);

        const posLoc = gl.getAttribLocation(prog, "a_position");
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

        const uTime = gl.getUniformLocation(prog, "u_time");
        const uScale = gl.getUniformLocation(prog, "u_scale");
        const uSpeed = gl.getUniformLocation(prog, "u_speed");
        const uLines = gl.getUniformLocation(prog, "u_lines");
        const uGlow = gl.getUniformLocation(prog, "u_glow");
        const uRes = gl.getUniformLocation(prog, "u_resolution");
        const uBg = gl.getUniformLocation(prog, "u_bg");
        const uFg = gl.getUniformLocation(prog, "u_fg");

        const X_RES = 1
        const Y_RES = 1
        canvas.width = window.innerWidth * X_RES
        canvas.height = window.innerHeight * Y_RES
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.uniform2f(uRes, canvas.width, canvas.height);
        gl.uniform2f(uRes, canvas.width, canvas.height);
        gl.uniform3f(uBg, 0.047, 0.094, 0.129);
        gl.uniform3f(uFg, 1.0, 0.925, 0.82);

        const start = performance.now();
        let lastTime = 0;
        const FPS = 60
        const interval = 1000 / FPS;

        function draw(timestamp) {
            rafRef.current = requestAnimationFrame(draw);
            if (timestamp - lastTime < interval) return;
            lastTime = timestamp;
            const c = configRef.current;
            gl.uniform1f(uTime, (performance.now() - start) / 1000);
            gl.uniform1f(uScale, c.scale);
            gl.uniform1f(uSpeed, c.speed);
            gl.uniform1f(uLines, c.lines);
            gl.uniform1f(uGlow, c.glow);
            gl.drawArrays(gl.TRIANGLES, 0, 6);
        }
        rafRef.current = requestAnimationFrame(draw);

        const onResize = () => {
            canvas.width = window.innerWidth * X_RES
            canvas.height = window.innerHeight * Y_RES
            gl.viewport(0, 0, canvas.width, canvas.height);
            gl.uniform2f(uRes, canvas.width, canvas.height);
        };
        window.addEventListener("resize", onResize);

        return () => {
            cancelAnimationFrame(rafRef.current);
            window.removeEventListener("resize", onResize);
        };
    }, []);

    return (
        <>
            <canvas ref={canvasRef} className="fixed inset-0 -z-10" style={{ width: "100vw", height: "100vh" }} />

            {DEV_MODE && (
                <div className="fixed top-20 right-4 z-50 select-none" style={{ width: 220, fontFamily: "monospace" }}>
                    <div
                        className="flex justify-between items-center px-3 py-1 cursor-pointer"
                        style={{ background: "#1a1a1a", color: "#ddd", fontSize: 11 }}
                        onClick={() => setOpen(o => !o)}
                    >
                        <span>⚙ background</span>
                        <span style={{ opacity: 0.5 }}>{open ? "▾" : "▸"}</span>
                    </div>

                    {open && (
                        <div style={{ background: "#222", borderTop: "1px solid #333" }}>
                            {Object.keys(DEFAULTS).map((key) => {
                                const r = RANGES[key];
                                const val = config[key];
                                const pct = ((val - r.min) / (r.max - r.min)) * 100;
                                return (
                                    <div key={key} style={{ borderBottom: "1px solid #2a2a2a", padding: "4px 0" }}>
                                        <div className="flex justify-between px-3" style={{ fontSize: 10, color: "#888", marginBottom: 2 }}>
                                            <span
                                                style={{ cursor: "ew-resize", color: "#aaa", userSelect: "none" }}
                                                onMouseDown={e => setDragging({ key, startX: e.clientX, startVal: val })}
                                            >
                                                {key}
                                            </span>
                                            <input
                                                type="number"
                                                value={val}
                                                step={r.step}
                                                min={r.min}
                                                max={r.max}
                                                onChange={e => {
                                                    const v = parseFloat(e.target.value);
                                                    if (!isNaN(v)) setConfig(p => ({ ...p, [key]: v }));
                                                }}
                                                style={{ background: "transparent", border: "none", color: "#e8c97a", fontSize: 10, width: 52, textAlign: "right", outline: "none" }}
                                            />
                                        </div>
                                        <div className="px-3">
                                            <div style={{ position: "relative", height: 14, background: "#111", borderRadius: 2 }}>
                                                <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${pct}%`, background: "#3a5f8a", borderRadius: 2 }} />
                                                <input
                                                    type="range"
                                                    min={r.min} max={r.max} step={r.step}
                                                    value={val}
                                                    onChange={e => setConfig(p => ({ ...p, [key]: parseFloat(e.target.value) }))}
                                                    style={{ position: "absolute", inset: 0, width: "100%", height: "100%", opacity: 0, cursor: "ew-resize", margin: 0 }}
                                                />
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                            <div
                                className="px-3 py-2 text-center cursor-pointer"
                                style={{ fontSize: 10, color: "#666", background: "#1a1a1a" }}
                                onClick={() => setConfig(DEFAULTS)}
                            >
                                reset defaults
                            </div>
                        </div>
                    )}
                </div>
            )}
        </>
    );
}