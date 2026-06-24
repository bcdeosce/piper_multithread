import os
import re
import io
import wave
import time
import queue
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
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
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("piper-api")

# ---------- Forçar CPU ----------
ort.set_default_logger_severity(3)  # reduz verbosidade

# ---------- Diretórios ----------
BASE_DIR = Path("/app")
VOICES_DIR = BASE_DIR / "voices"
AMBIENT_DIR = BASE_DIR / "ambient"
EFFECTS_DIR = BASE_DIR / "effects"  # fallback global

VOICES_DIR.mkdir(exist_ok=True)
AMBIENT_DIR.mkdir(exist_ok=True)
EFFECTS_DIR.mkdir(exist_ok=True)

# ---------- Pools de workers ----------
NUM_TTS_WORKERS = int(os.getenv("TTS_WORKERS", min(8, (os.cpu_count() or 4))))
NUM_MIX_WORKERS = int(os.getenv("MIX_WORKERS", 2))
logger.info(f"Workers: TTS={NUM_TTS_WORKERS}, Mix={NUM_MIX_WORKERS}")

tts_executor = ThreadPoolExecutor(max_workers=NUM_TTS_WORKERS)
mix_executor = ThreadPoolExecutor(max_workers=NUM_MIX_WORKERS)

# ---------- Pool de instâncias (CPU) ----------
class VoicePool:
    def __init__(self, model_path: str, config_path: str, pool_size: int = 2):
        self.pool = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            voice = PiperVoice.load(
                model_path,
                config_path=config_path,
                use_cuda=False,
            )
            self.pool.put(voice)

    def get(self, timeout=2.0):
        return self.pool.get(timeout=timeout)

    def put(self, voice):
        self.pool.put(voice)

def load_voice_from_folder(voice_name: str, voice_path: Path) -> dict:
    """Carrega uma voz a partir de uma subpasta que contenha .onnx e .onnx.json."""
    onnx_files = list(voice_path.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"Nenhum arquivo .onnx encontrado em {voice_path}")
    model_path = str(onnx_files[0])
    base_name = onnx_files[0].stem
    json_path = voice_path / f"{base_name}.onnx.json"
    if not json_path.exists():
        json_candidates = list(voice_path.glob("*.json"))
        if not json_candidates:
            raise FileNotFoundError(f"Nenhum arquivo .json encontrado para a voz {voice_name}")
        json_path = json_candidates[0]
    config_path = str(json_path)

    genero = "Desconhecido"
    meta_path = voice_path / f"{voice_name}.json"
    if meta_path.exists():
        import json
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                genero = meta.get("genero", "Desconhecido")
        except:
            pass

    pool = VoicePool(model_path, config_path, pool_size=2)
    return {
        "model_path": model_path,
        "config_path": config_path,
        "genero": genero,
        "pool": pool,
        "path": voice_path
    }

voices_registry: Dict[str, dict] = {}  # nome_da_voz -> entrada

# Carregar todas as vozes
for item in VOICES_DIR.iterdir():
    if item.is_dir():
        voice_name = item.name
        try:
            entry = load_voice_from_folder(voice_name, item)
            voices_registry[voice_name] = entry
            logger.info(f"✅ Voz carregada: {voice_name} ({entry['genero']})")
        except Exception as e:
            logger.error(f"❌ Falha ao carregar voz {voice_name}: {e}")

# Backward compatibility: arquivos .onnx diretamente na raiz
for onnx_file in VOICES_DIR.glob("*.onnx"):
    voice_name = onnx_file.stem
    if voice_name in voices_registry:
        continue
    json_file = onnx_file.with_suffix(".onnx.json")
    if json_file.exists():
        try:
            pool = VoicePool(str(onnx_file), str(json_file), pool_size=2)
            voices_registry[voice_name] = {
                "model_path": str(onnx_file),
                "config_path": str(json_file),
                "genero": "Personalizada",
                "pool": pool,
                "path": VOICES_DIR
            }
            logger.info(f"✅ Voz personalizada (raiz) carregada: {voice_name}")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar voz {voice_name}: {e}")

logger.info(f"Total de vozes disponíveis: {len(voices_registry)}")
MODEL_LOADED = len(voices_registry) > 0

# ---------- Caches para áudios ----------
effect_cache: Dict[Tuple[str, str], AudioSegment] = {}
ambient_cache: Dict[Tuple[str, float], AudioSegment] = {}

def load_effect(voice_name: str, effect_file: str) -> AudioSegment:
    cache_key = (voice_name, effect_file)
    if cache_key in effect_cache:
        return effect_cache[cache_key]

    voice_entry = voices_registry.get(voice_name)
    if voice_entry is None:
        raise ValueError(f"Voz '{voice_name}' não encontrada")
    voice_dir = voice_entry.get("path", VOICES_DIR / voice_name)

    effect_path = voice_dir / effect_file
    if not effect_path.exists():
        effect_path = EFFECTS_DIR / effect_file
    if not effect_path.exists():
        raise FileNotFoundError(f"Efeito '{effect_file}' não encontrado em {voice_dir} nem em {EFFECTS_DIR}")

    seg = AudioSegment.from_wav(str(effect_path))
    effect_cache[cache_key] = seg
    logger.info(f"✔ Efeito '{effect_file}' carregado (voz: {voice_name})")
    return seg

def load_ambient(ambient_file: str, volume_db: float) -> AudioSegment:
    cache_key = (ambient_file, volume_db)
    if cache_key in ambient_cache:
        return ambient_cache[cache_key]

    ambient_path = AMBIENT_DIR / f"{ambient_file}.wav"
    if not ambient_path.exists():
        raise FileNotFoundError(f"Ambiente '{ambient_file}.wav' não encontrado em {AMBIENT_DIR}")

    seg = AudioSegment.from_wav(str(ambient_path))
    seg = seg + volume_db
    ambient_cache[cache_key] = seg
    logger.info(f"✔ Ambiente '{ambient_file}' carregado (volume {volume_db} dB)")
    return seg

# ---------- Função para padronizar segmentos ----------
def standardize_audio(seg: AudioSegment, target_rate: int = 22050) -> AudioSegment:
    """Converte segmento para mono, 16-bit, target_rate."""
    if seg.channels > 1:
        seg = seg.set_channels(1)
    if seg.sample_width != 2:
        seg = seg.set_sample_width(2)
    if seg.frame_rate != target_rate:
        seg = seg.set_frame_rate(target_rate)
    return seg

# ---------- Função executada nos workers TTS ----------
def synthesize_text(voice_name: str, text: str, speed: float,
                    noise_scale: float, noise_w_scale: float) -> Tuple[int, bytes]:
    """
    Executa a síntese de fala usando o pool da voz especificada.
    Retorna (sample_rate, audio_bytes_pcm16).
    """
    entry = voices_registry[voice_name]
    pool = entry["pool"]
    voice = pool.get()
    try:
        config = SynthesisConfig(
            length_scale=speed,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
            volume=1.0
        )
        chunk_generator = voice.synthesize(text, syn_config=config)
        audio_bytes = b''.join(chunk.audio_int16_bytes for chunk in chunk_generator)
        sample_rate = voice.config.sample_rate
        return sample_rate, audio_bytes
    finally:
        pool.put(voice)

# ---------- Função de mixagem e exportação ----------
def mix_and_export(audio_segments: List[AudioSegment],
                   ambient_config: 'AmbientConfig',
                   target_rate: int = 22050) -> bytes:
    """
    Concatena segmentos, normaliza, aplica ambiente e exporta para WebM Opus.
    Levanta ValueError em caso de erro para ser tratado pelo endpoint.
    """
    if not audio_segments:
        raise ValueError("Nenhum segmento de áudio")

    # Padroniza todos os segmentos (mono, 16-bit, target_rate)
    try:
        uniform_segments = [standardize_audio(seg, target_rate) for seg in audio_segments]
    except Exception as e:
        logger.error(f"Erro ao padronizar segmentos: {e}")
        raise ValueError(f"Falha na padronização de um segmento: {e}")

    # Concatenação eficiente (O(n))
    try:
        if len(uniform_segments) == 1:
            combined = uniform_segments[0]
        else:
            combined = AudioSegment.from_mono_audiosegments(*uniform_segments)
    except Exception as e:
        logger.error(f"Erro na concatenação: {e}")
        # Informação extra para debug
        for i, seg in enumerate(uniform_segments):
            logger.error(f"  Segmento {i}: canais={seg.channels}, sample_width={seg.sample_width}, frame_rate={seg.frame_rate}, duração={len(seg)/1000:.2f}s")
        raise ValueError(f"Falha ao concatenar segmentos: {e}")

    # Normalização de loudness para -20 dBFS
    try:
        target_dBFS = -20.0
        if combined.dBFS != target_dBFS:
            gain = target_dBFS - combined.dBFS
            combined = combined.apply_gain(gain)
    except Exception as e:
        logger.error(f"Erro na normalização: {e}")
        raise ValueError(f"Falha na normalização: {e}")

    # Mixagem do ambiente, se habilitado
    if ambient_config.enabled and ambient_config.file:
        try:
            ambient = load_ambient(ambient_config.file, ambient_config.volume_db)
            ambient = standardize_audio(ambient, target_rate)
            if len(ambient) < len(combined):
                ambient = ambient * ((len(combined) // len(ambient)) + 1)
            ambient = ambient[:len(combined)]
            combined = combined.overlay(ambient)
        except FileNotFoundError as e:
            logger.error(f"Arquivo de ambiente não encontrado: {e}")
            raise ValueError(f"Ambiente '{ambient_config.file}' não encontrado")
        except Exception as e:
            logger.error(f"Erro na mixagem do ambiente: {e}")
            raise ValueError(f"Falha ao aplicar ambiente: {e}")

    # Exportação para WebM (Opus)
    try:
        with io.BytesIO() as out_buf:
            combined.export(out_buf, format="webm", codec="libopus",
                            parameters=["-b:a", "64k"])
            return out_buf.getvalue()
    except Exception as e:
        logger.error(f"Erro na exportação para Opus: {e}")
        raise ValueError(f"Falha ao codificar áudio: {e}")

# ---------- Modelos de requisição ----------
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
    voice: Optional[str] = Field(None, description="Nome da voz (modo único)")
    text: str = Field(..., min_length=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    noise_scale: float = Field(default=0.667, ge=0.0, le=1.5)
    noise_w_scale: float = Field(default=0.8, ge=0.0, le=2.0)
    effects: Dict[str, str] = Field(default_factory=dict)
    ambient: AmbientConfig = Field(default_factory=AmbientConfig)
    speakers: List[SpeakerMapping] = Field(default_factory=list)

# ---------- FastAPI app ----------
app = FastAPI(title="Piper TTS API (CPU + mixagem paralela)")

@app.post("/synthesize", response_class=Response)
async def synthesize(req: TTSRequest):
    t_total_start = time.perf_counter()
    logger.info(f"🔔 Nova requisição: text='{req.text}', effects={list(req.effects.keys())}, ambient={req.ambient.enabled}")

    # --- Validações e mapeamento de speakers ---
    is_dialog = bool(req.speakers)
    if not is_dialog:
        if not req.voice:
            raise HTTPException(400, "Campo 'voice' é obrigatório no modo simples")
        if req.voice not in voices_registry:
            raise HTTPException(404, f"Voz não encontrada: {req.voice}")
        speaker_map = {None: (req.voice, req.speed, req.noise_scale, req.noise_w_scale)}
        current_role = None
    else:
        speaker_map = {}
        for spk in req.speakers:
            noise_s = spk.noise_scale if spk.noise_scale is not None else req.noise_scale
            noise_w = spk.noise_w_scale if spk.noise_w_scale is not None else req.noise_w_scale
            speaker_map[spk.role] = (spk.voice, spk.speed, noise_s, noise_w)
        for role, (voice_name, _, _, _) in speaker_map.items():
            if voice_name not in voices_registry:
                raise HTTPException(404, f"Voz '{voice_name}' do speaker '{role}' não encontrada")
        current_role = None

    # --- Divisão do texto ---
    t_div_start = time.perf_counter()
    parts = re.split(r'(\[.*?\])', req.text)
    parts = [p.strip() for p in parts if p.strip()]
    logger.info(f"🔹 Texto dividido em {len(parts)} partes ({time.perf_counter()-t_div_start:.4f}s)")

    # --- Planejamento das tarefas de síntese e coleta de efeitos ---
    t_plan_start = time.perf_counter()
    synthesis_tasks = []      # Lista de corrotinas (futuros assíncronos)
    synthesis_indices = []    # Em qual posição da lista final o resultado será inserido
    audio_segments = [None] * len(parts)  # Lista final ordenada

    loop = asyncio.get_running_loop()

    for idx, part in enumerate(parts):
        # 1. Tag de speaker
        if is_dialog and part.startswith('[') and part.endswith(']'):
            role = part[1:-1]
            if role in speaker_map:
                current_role = role
                logger.info(f"🗣️ Speaker trocado para: {current_role}")
            continue

        # 2. Efeito sonoro
        if part in req.effects:
            effect_file = req.effects[part]
            logger.info(f"🎬 Efeito '{part}' -> '{effect_file}'")
            try:
                voice_for_effect = speaker_map[current_role][0] if is_dialog and current_role else req.voice
                effect_audio = load_effect(voice_for_effect, effect_file)
                effect_audio = standardize_audio(effect_audio)
                audio_segments[idx] = effect_audio
                logger.info(f"🎬 Efeito adicionado: duração={len(effect_audio)/1000:.2f}s")
            except Exception as e:
                logger.error(f"🎬 Falha ao carregar efeito: {e}")
                silence = AudioSegment.silent(duration=500, frame_rate=22050)
                audio_segments[idx] = standardize_audio(silence)
            continue

        # 3. Síntese de fala
        if is_dialog:
            if current_role is None:
                raise HTTPException(400, "Nenhum speaker definido antes do texto. Use [papel] no início.")
            voice_name, speed, noise_s, noise_w = speaker_map[current_role]
        else:
            voice_name = req.voice
            speed = req.speed
            noise_s = req.noise_scale
            noise_w = req.noise_w_scale

        # Agenda tarefa de síntese no pool TTS
        task = loop.run_in_executor(
            tts_executor,
            synthesize_text,
            voice_name, part, speed, noise_s, noise_w
        )
        synthesis_tasks.append(task)
        synthesis_indices.append(idx)

    logger.info(f"🔹 Planejamento concluído: {len(synthesis_tasks)} sínteses agendadas, "
                f"{sum(1 for s in audio_segments if s is not None)} efeitos diretos "
                f"({time.perf_counter()-t_plan_start:.4f}s)")

    # --- Aguardar todas as sínteses (paralelo) com tratamento de erros ---
    t_synth_start = time.perf_counter()
    if synthesis_tasks:
        raw_results = await asyncio.gather(*synthesis_tasks, return_exceptions=True)
    else:
        raw_results = []
    t_synth_end = time.perf_counter()

    # Processa resultados, substituindo exceções por silêncio e logando
    synth_results = []
    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            logger.error(f"Erro na síntese da parte '{parts[synthesis_indices[i]]}': {result}")
            synth_results.append((22050, AudioSegment.silent(duration=500, frame_rate=22050).raw_data))
        else:
            synth_results.append(result)

    logger.info(f"🔹 Sínteses concluídas em {t_synth_end-t_synth_start:.4f}s "
                f"(RTF médio: {(t_synth_end-t_synth_start)/(sum(len(r[1])/2/22050 for r in synth_results) if synth_results else 1):.3f})")

    # --- Montar AudioSegments para os trechos sintetizados ---
    t_seg_start = time.perf_counter()
    for (sample_rate, pcm_bytes), idx in zip(synth_results, synthesis_indices):
        seg = AudioSegment(
            data=pcm_bytes,
            sample_width=2,
            frame_rate=sample_rate,
            channels=1
        )
        seg = standardize_audio(seg)
        audio_segments[idx] = seg
        logger.debug(f"Segmento TTS idx={idx}: duração={len(seg)/1000:.2f}s, dBFS={seg.dBFS:.1f}")

    # Remove eventuais None (tags de speaker que não geraram áudio) e mantém a ordem
    final_segments = [s for s in audio_segments if s is not None]
    logger.info(f"🔹 Segmentos montados: {len(final_segments)} segmentos "
                f"({time.perf_counter()-t_seg_start:.4f}s)")

    if not final_segments:
        raise HTTPException(500, "Nenhum áudio foi gerado")

    # --- Enviar para mixagem e exportação (em thread separada) ---
    t_mix_start = time.perf_counter()
    try:
        mixed_bytes = await loop.run_in_executor(
            mix_executor,
            mix_and_export,
            final_segments,
            req.ambient
        )
    except ValueError as e:
        logger.error(f"Erro na mixagem/exportação: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Erro inesperado na mixagem/exportação: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {e}")
    t_mix_end = time.perf_counter()
    logger.info(f"🔹 Mixagem + exportação: {t_mix_end-t_mix_start:.4f}s")

    # --- Finalização ---
    duracao_final = sum(len(s) for s in final_segments) / 1000.0
    tempo_total = time.perf_counter() - t_total_start
    logger.info(f"✅ Síntese finalizada | tempo_total={tempo_total:.3f}s | "
                f"áudio={duracao_final:.1f}s | RTF total={tempo_total/duracao_final:.3f}")

    return Response(content=mixed_bytes, media_type="audio/webm")

# ---------- Endpoints de saúde ----------
@app.get("/started")
async def started():
    return Response(status_code=200, content="started")

@app.get("/ready")
async def ready():
    if MODEL_LOADED:
        return Response(status_code=200, content="ready")
    return Response(status_code=503, content="loading model")

@app.get("/live")
async def live():
    return Response(status_code=200, content="alive")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gpu": False,
        "voices_loaded": list(voices_registry.keys()),
        "total_voices": len(voices_registry),
        "workers": {
            "tts": NUM_TTS_WORKERS,
            "mix": NUM_MIX_WORKERS
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
