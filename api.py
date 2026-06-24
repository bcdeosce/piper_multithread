import os
import re
import io
import time
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import asyncio

import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from piper import PiperVoice, SynthesisConfig
from pydub import AudioSegment

# ---------- Configuração de logs ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(processName)s | %(message)s",
)
logger = logging.getLogger("piper-api")
ort.set_default_logger_severity(3)

# ---------- Diretórios ----------
BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"
VOICES_DIR.mkdir(exist_ok=True)
AMBIENT_DIR.mkdir(exist_ok=True)
EFFECTS_DIR.mkdir(exist_ok=True)

# ---------- Pools globais ----------
NUM_TTS_WORKERS = int(os.getenv("TTS_WORKERS", min(8, os.cpu_count() or 4)))
NUM_MIX_WORKERS = int(os.getenv("MIX_WORKERS", 2))
logger.info(f"Workers: TTS={NUM_TTS_WORKERS} processos, Mix={NUM_MIX_WORKERS} threads")

# Executor de processos para síntese (cada processo carrega os modelos)
tts_executor = ProcessPoolExecutor(max_workers=NUM_TTS_WORKERS, initializer=init_worker)
mix_executor = ThreadPoolExecutor(max_workers=NUM_MIX_WORKERS)

# ---------- Registro de vozes (compartilhado via variáveis globais no processo principal) ----------
voices_registry: Dict[str, dict] = {}

def load_voice_from_folder(voice_name: str, voice_path: Path) -> dict:
    onnx_files = list(voice_path.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"Nenhum .onnx em {voice_path}")
    model_path = str(onnx_files[0])
    base = onnx_files[0].stem
    json_path = voice_path / f"{base}.onnx.json"
    if not json_path.exists():
        json_candidates = list(voice_path.glob("*.json"))
        if not json_candidates:
            raise FileNotFoundError(f"Nenhum .json em {voice_path}")
        json_path = json_candidates[0]
    config_path = str(json_path)

    genero = "Desconhecido"
    meta_path = voice_path / f"{voice_name}.json"
    if meta_path.exists():
        import json
        try:
            with open(meta_path) as f:
                meta = json.load(f)
                genero = meta.get("genero", "Desconhecido")
        except: pass

    return {
        "model_path": model_path,
        "config_path": config_path,
        "genero": genero,
        "path": voice_path
    }

# Carrega vozes na inicialização (apenas metadados, modelo será carregado nos workers)
for item in VOICES_DIR.iterdir():
    if item.is_dir():
        try:
            entry = load_voice_from_folder(item.name, item)
            voices_registry[item.name] = entry
            logger.info(f"✅ Voz registrada: {item.name} ({entry['genero']})")
        except Exception as e:
            logger.error(f"❌ Falha ao registrar voz {item.name}: {e}")

for onnx_file in VOICES_DIR.glob("*.onnx"):
    voice_name = onnx_file.stem
    if voice_name in voices_registry:
        continue
    json_file = onnx_file.with_suffix(".onnx.json")
    if json_file.exists():
        voices_registry[voice_name] = {
            "model_path": str(onnx_file),
            "config_path": str(json_file),
            "genero": "Personalizada",
            "path": VOICES_DIR
        }
        logger.info(f"✅ Voz raiz registrada: {voice_name}")

logger.info(f"Total de vozes: {len(voices_registry)}")

# ---------- Caches de áudio (carregados sob demanda, mas padronizados) ----------
effect_cache: Dict[Tuple[str, str], AudioSegment] = {}
ambient_cache: Dict[Tuple[str, float], AudioSegment] = {}

def load_effect(voice_name: str, effect_file: str) -> AudioSegment:
    """Carrega um efeito WAV e o padroniza (mono, 16-bit, 22050 Hz)."""
    key = (voice_name, effect_file)
    if key in effect_cache:
        return effect_cache[key]

    voice_entry = voices_registry.get(voice_name)
    if not voice_entry:
        raise ValueError(f"Voz '{voice_name}' não encontrada")
    voice_dir = voice_entry["path"]

    effect_path = voice_dir / effect_file
    if not effect_path.exists():
        effect_path = EFFECTS_DIR / effect_file
    if not effect_path.exists():
        raise FileNotFoundError(f"Efeito '{effect_file}' não encontrado em {voice_dir} ou {EFFECTS_DIR}")

    seg = AudioSegment.from_wav(str(effect_path))
    # Padronização imediata
    if seg.channels > 1:
        seg = seg.set_channels(1)
    if seg.sample_width != 2:
        seg = seg.set_sample_width(2)
    if seg.frame_rate != 22050:
        seg = seg.set_frame_rate(22050)
    effect_cache[key] = seg
    logger.info(f"✔ Efeito '{effect_file}' carregado e padronizado")
    return seg

def load_ambient(ambient_file: str, volume_db: float) -> AudioSegment:
    key = (ambient_file, volume_db)
    if key in ambient_cache:
        return ambient_cache[key]

    ambient_path = AMBIENT_DIR / f"{ambient_file}.wav"
    if not ambient_path.exists():
        raise FileNotFoundError(f"Ambiente '{ambient_file}.wav' não encontrado")

    seg = AudioSegment.from_wav(str(ambient_path))
    seg = seg + volume_db
    # Padronização
    if seg.channels > 1:
        seg = seg.set_channels(1)
    if seg.sample_width != 2:
        seg = seg.set_sample_width(2)
    if seg.frame_rate != 22050:
        seg = seg.set_frame_rate(22050)
    ambient_cache[key] = seg
    logger.info(f"✔ Ambiente '{ambient_file}' carregado")
    return seg

# ---------- Função de inicialização dos workers TTS ----------
_worker_voices = {}  # dict voz_name -> PiperVoice (carregado no processo)

def init_worker():
    """Inicializa o processo worker carregando todos os modelos."""
    global _worker_voices
    for name, entry in voices_registry.items():
        try:
            voice = PiperVoice.load(entry["model_path"], entry["config_path"], use_cuda=False)
            _worker_voices[name] = voice
            logger.info(f"Worker {os.getpid()}: voz {name} carregada")
        except Exception as e:
            logger.error(f"Worker {os.getpid()}: falha ao carregar {name}: {e}")

def synthesize_text(voice_name: str, text: str, speed: float,
                    noise_scale: float, noise_w_scale: float) -> Tuple[int, bytes]:
    """Executada no processo worker. Retorna (sample_rate, pcm_bytes)."""
    voice = _worker_voices[voice_name]
    config = SynthesisConfig(
        length_scale=speed,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
        volume=1.0
    )
    chunk_gen = voice.synthesize(text, syn_config=config)
    audio_bytes = b''.join(c.audio_int16_bytes for c in chunk_gen)
    return (voice.config.sample_rate, audio_bytes)

# ---------- Função de mixagem (executada em thread) ----------
def mix_and_export(segments: List[AudioSegment], ambient_config: 'AmbientConfig') -> bytes:
    if not segments:
        raise ValueError("Nenhum segmento")

    # Concatenação direta (todos já estão em 22050 Hz, mono, 16-bit)
    combined = segments[0] if len(segments) == 1 else AudioSegment.from_mono_audiosegments(*segments)

    # Normalização
    target_dbfs = -20.0
    if combined.dBFS != target_dbfs:
        gain = target_dbfs - combined.dBFS
        combined = combined.apply_gain(gain)

    # Ambiente
    if ambient_config.enabled and ambient_config.file:
        ambient = load_ambient(ambient_config.file, ambient_config.volume_db)
        if len(ambient) < len(combined):
            ambient = ambient * ((len(combined) // len(ambient)) + 1)
        ambient = ambient[:len(combined)]
        combined = combined.overlay(ambient)

    # Exportar
    with io.BytesIO() as out:
        combined.export(out, format="webm", codec="libopus", parameters=["-b:a", "64k"])
        return out.getvalue()

# ---------- Modelos ----------
class AmbientConfig(BaseModel):
    enabled: bool = False
    file: Optional[str] = None
    volume_db: float = Field(default=-15.0, ge=-60.0, le=12.0)

class SpeakerMapping(BaseModel):
    role: str
    voice: str
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    noise_scale: Optional[float] = Field(default=None, ge=0.0, le=1.5)
    noise_w_scale: Optional[float] = Field(default=None, ge=0.0, le=2.0)

class TTSRequest(BaseModel):
    voice: Optional[str] = None
    text: str = Field(..., min_length=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    noise_scale: float = Field(default=0.667, ge=0.0, le=1.5)
    noise_w_scale: float = Field(default=0.8, ge=0.0, le=2.0)
    effects: Dict[str, str] = Field(default_factory=dict)
    ambient: AmbientConfig = Field(default_factory=AmbientConfig)
    speakers: List[SpeakerMapping] = Field(default_factory=list)

# ---------- FastAPI ----------
app = FastAPI(title="Piper TTS API otimizada")

@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    t0 = time.perf_counter()
    logger.info(f"🔔 Nova requisição: text='{req.text}', effects={req.effects}, ambient={req.ambient.enabled}")

    # Validação e mapeamento de speakers
    is_dialog = bool(req.speakers)
    if not is_dialog:
        if not req.voice:
            raise HTTPException(400, "Campo 'voice' obrigatório")
        if req.voice not in voices_registry:
            raise HTTPException(404, f"Voz não encontrada: {req.voice}")
        speaker_map = {None: (req.voice, req.speed, req.noise_scale, req.noise_w_scale)}
        current_role = None
    else:
        speaker_map = {}
        for spk in req.speakers:
            ns = spk.noise_scale if spk.noise_scale is not None else req.noise_scale
            nw = spk.noise_w_scale if spk.noise_w_scale is not None else req.noise_w_scale
            speaker_map[spk.role] = (spk.voice, spk.speed, ns, nw)
        for role, (voice_name, _, _, _) in speaker_map.items():
            if voice_name not in voices_registry:
                raise HTTPException(404, f"Voz '{voice_name}' do speaker '{role}' não encontrada")
        current_role = None

    # Divisão do texto
    parts = [p.strip() for p in re.split(r'(\[.*?\])', req.text) if p.strip()]
    logger.info(f"🔹 Partes: {parts}")

    synthesis_tasks = []
    synthesis_indices = []
    audio_segments = [None] * len(parts)
    loop = asyncio.get_running_loop()

    for idx, part in enumerate(parts):
        # Tag de speaker
        if is_dialog and part.startswith('[') and part.endswith(']'):
            role = part[1:-1]
            if role in speaker_map:
                current_role = role
                logger.info(f"🗣️ Speaker -> {role}")
            continue

        # Efeito sonoro
        if part in req.effects:
            effect_file = req.effects[part]
            logger.info(f"🎬 Efeito '{part}' -> '{effect_file}'")
            try:
                # Voz do efeito pode ser a atual (se diálogo) ou a única
                voice_for_effect = speaker_map[current_role][0] if is_dialog and current_role else req.voice
                effect_audio = load_effect(voice_for_effect, effect_file)
                audio_segments[idx] = effect_audio
                logger.info(f"🎬 Efeito adicionado: {len(effect_audio)/1000:.2f}s")
            except Exception as e:
                logger.error(f"🎬 Falha ao carregar efeito: {e}")
                audio_segments[idx] = AudioSegment.silent(duration=500, frame_rate=22050)
            continue

        # Síntese de fala
        if is_dialog:
            if current_role is None:
                raise HTTPException(400, "Nenhum speaker definido antes do texto.")
            voice_name, speed, ns, nw = speaker_map[current_role]
        else:
            voice_name, speed, ns, nw = req.voice, req.speed, req.noise_scale, req.noise_w_scale

        task = loop.run_in_executor(tts_executor, synthesize_text, voice_name, part, speed, ns, nw)
        synthesis_tasks.append(task)
        synthesis_indices.append(idx)

    # Aguardar sínteses
    t_synth_start = time.perf_counter()
    if synthesis_tasks:
        results = await asyncio.gather(*synthesis_tasks, return_exceptions=True)
    else:
        results = []
    t_synth = time.perf_counter() - t_synth_start
    logger.info(f"🔹 Sínteses concluídas em {t_synth:.3f}s")

    # Processar resultados
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error(f"Erro na síntese da parte '{parts[synthesis_indices[i]]}': {res}")
            audio_segments[synthesis_indices[i]] = AudioSegment.silent(duration=500, frame_rate=22050)
        else:
            sr, pcm = res
            seg = AudioSegment(data=pcm, sample_width=2, frame_rate=sr, channels=1)
            if sr != 22050:
                seg = seg.set_frame_rate(22050)
            audio_segments[synthesis_indices[i]] = seg

    final_segments = [s for s in audio_segments if s is not None]
    logger.info(f"🔹 Segmentos válidos: {len(final_segments)}")

    if not final_segments:
        raise HTTPException(500, "Nenhum áudio gerado")

    # Mixagem em thread separada
    t_mix_start = time.perf_counter()
    try:
        mixed_bytes = await loop.run_in_executor(mix_executor, mix_and_export, final_segments, req.ambient)
    except Exception as e:
        logger.error(f"Erro na mixagem: {e}")
        raise HTTPException(500, str(e))
    t_mix = time.perf_counter() - t_mix_start

    total_time = time.perf_counter() - t0
    logger.info(f"✅ Concluída em {total_time:.3f}s (síntese={t_synth:.3f}s, mix={t_mix:.3f}s)")
    return Response(content=mixed_bytes, media_type="audio/webm")

# ---------- Health ----------
@app.get("/started")
async def started(): return Response(status_code=200, content="started")
@app.get("/ready")
async def ready(): return Response(status_code=200 if voices_registry else 503, content="ready" if voices_registry else "loading")
@app.get("/live")
async def live(): return Response(status_code=200, content="alive")
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "voices_loaded": list(voices_registry.keys()),
        "total_voices": len(voices_registry),
        "workers_tts": NUM_TTS_WORKERS,
        "workers_mix": NUM_MIX_WORKERS
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
