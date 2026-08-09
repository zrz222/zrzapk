let ort = null;
let piperSession = null;
let piperTokens = null;
let piperCfg = null;
let piperReady = false;
const basePath = "piper_models/";

async function loadPiperModel() {
    if (piperReady) return true;
    try {
        ort = window.ort;
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.simd = false;

        const tokRes = await fetch(basePath + "tokens.txt");
        piperTokens = (await tokRes.text()).split("\n").filter(t => t.length > 0);
        const cfgRes = await fetch(basePath + "config.json");
        piperCfg = await cfgRes.json();
        const modelUri = basePath + "zh_CN‑huayan.onnx";
        piperSession = await ort.InferenceSession.create(modelUri);
        piperReady = true;
        return true;
    } catch (e) {
        console.error("Piper加载失败", e);
        piperReady = false;
        return false;
    }
}

function text2TokenIds(textStr) {
    const idList = [];
    for (let ch of textStr) {
        const idx = piperTokens.indexOf(ch);
        if (idx !== -1) idList.push(idx);
    }
    const eosIdx = piperTokens.indexOf("~");
    if(eosIdx >= 0) idList.push(eosIdx);
    return idList;
}

async function piperGenerateAudio(text, speed = 1.0) {
    if (!piperReady) throw new Error("模型未加载");
    const ids = text2TokenIds(text);
    if (ids.length === 0) throw new Error("无有效文字");

    const inputIds = new BigInt64Array(ids.map(v=>BigInt(v)));
    const feed = {
        input_ids: new ort.Tensor('int64', inputIds, [1, ids.length])
    };
    const result = await piperSession.run(feed);
    const audioRaw = result.output.data;
    const sampleRate = piperCfg.audio.sample_rate || 22050;

    const wavBuffer = createWav(audioRaw, sampleRate);
    return new Blob([wavBuffer], { type: "audio/wav" });
}

function createWav(pcmFloat, sampleRate){
    const numChan = 1;
    const bitDepth = 16;
    const byteRate = sampleRate * numChan * (bitDepth/8);
    const blockAlign = numChan * (bitDepth/8);
    const pcmInt16 = new Int16Array(pcmFloat.length);
    for(let i=0;i<pcmFloat.length;i++){
        let v = pcmFloat[i];
        v = Math.max(-1,Math.min(1,v));
        pcmInt16[i] = Math.round(v * 32767);
    }
    const wavLen = 44 + pcmInt16.byteLength;
    const buf = new ArrayBuffer(wavLen);
    const view = new DataView(buf);
    view.setUint8(0,0x52);view.setUint8(1,0x49);view.setUint8(2,0x46);view.setUint8(3,0x46);
    view.setUint32(4,wavLen‑8,true);
    view.setUint8(8,0x57);view.setUint8(9,0x41);view.setUint8(10,0x56);view.setUint8(11,0x45);
    view.setUint8(12,0x66);view.setUint8(13,0x6d);view.setUint8(14,0x74);view.setUint8(15,0x20);
    view.setUint32(16,16,true);
    view.setUint16(20,1,true);
    view.setUint16(22,numChan,true);
    view.setUint32(24,sampleRate,true);
    view.setUint32(28,byteRate,true);
    view.setUint16(32,blockAlign,true);
    view.setUint16(34,bitDepth,true);
    view.setUint8(36,0x64);view.setUint8(37,0x61);view.setUint8(38,0x74);view.setUint8(39,0x61);
    view.setUint32(40,pcmInt16.byteLength,true);
    new Int16Array(buf,44).set(pcmInt16);
    return buf;
}
