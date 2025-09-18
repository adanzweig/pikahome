#!/usr/bin/env/ python3
# LiveKit Agents v1: AgentSession + Agent + OpenAI Realtime + Spotify + Camera
import os, asyncio, base64, subprocess, datetime, sys, fcntl
from pathlib import Path
from dotenv import load_dotenv
import contextlib
import asyncio
from livekit import agents
from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    function_tool,
    RunContext,
    ModelSettings
)
from livekit.plugins import openai  # Realtime model lives here

# Spotify
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyOauthError
import inspect
import tempfile,contextlib
import re
import unicodedata
import json
from openai import OpenAI
from openai.types.beta.realtime.session import TurnDetection
load_dotenv()

Path("/home/bk/pika-voice/pikahome").mkdir(parents=True, exist_ok=True)
fd = os.open("/home/bk/pika-voice/lock", os.O_CREAT | os.O_RDWR, 0o644)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print("[pikahome] another instance is running; exiting.", file=sys.stderr)
    sys.exit(0)

VISUAL_MODEL = os.getenv("VISUAL_MODEL", "gpt-4o-mini")
# --- LiveKit / OpenAI ---
LIVEKIT_URL        = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY    = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
OPENAI_VOICE = os.getenv("OPENAI_VOICE","alloy")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL     = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")  # OpenAI Realtime
PRIMARY_LANG       = os.getenv("PRIMARY_LANG", "es-AR")

# --- Spotify env ---
SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")
SPOTIFY_DEVICE_NAME   = os.getenv("SPOTIFY_DEVICE_NAME", "pikahome")
SPOTIFY_AUTH_ERROR_MSG = (
    "Spotify necesita que vuelvas a iniciar sesión. Abrí el setup de Spotify y autorizá de nuevo, porfi."
)

# --- Camera env ---
PHOTO_W   = int(os.getenv("PHOTO_WIDTH", "640"))
PHOTO_H   = int(os.getenv("PHOTO_HEIGHT","480"))
SAVE_DIRS = [p for p in [os.getenv("SAVE_PHOTOS_DIR","")] if p]
if SAVE_DIRS:
    Path(SAVE_DIRS[0]).expanduser().mkdir(parents=True, exist_ok=True)
USE_LOCAL_TTS_FALLBACK = os.getenv("LOCAL_TTS_FALLBACK", "1") == "1"
OUTPUT_ALSA_DEVICE = os.getenv("OUTPUT_ALSA_DEVICE") or "plughw:3,0"
TTS_LANG = os.getenv("TTS_LANG", "es-ES")                  # for pico2wave/espeak

async def _say_local_exact(text: str, device: str | None = None):
    """Legacy local speech fallback (espeak-ng). Kept for debugging."""
    if not text:
        return
    espeak = subprocess.Popen(
        ["espeak-ng", "-v", "es-la", "-s", "170", "--stdout"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    aplay = subprocess.Popen(
        ["aplay", "-q"],
        stdin=espeak.stdout
    )
    espeak.stdin.write(text.encode("utf-8"))
    espeak.stdin.close()
    aplay.wait()
    espeak.wait()

    #tmp = Path(tempfile.gettempdir()) / "pika_tool_say.wav"
    # ---------- Try 1: OpenAI TTS (correct modern SDK call) ----------
    #try:
    #    from openai import OpenAI
    #    api_key = os.getenv("OPENAI_API_KEY")
    #    if not api_key:
    #        raise RuntimeError("OPENAI_API_KEY missing")
    #    client = OpenAI(api_key=api_key)
#
        # Newer SDKs use streaming; no 'format=' kw
    #    with client.audio.speech.with_streaming_response.create(
    #        model="gpt-4o-mini-tts",  # or "tts-1" if you prefer
    #        voice=OPENAI_VOICE,
    #        input=text,
    #    ) as resp:
    #        resp.stream_to_file(str(tmp))
        # if we got here, we have a wav
    #    return await _play_wav(str(tmp), device=device)
    #except Exception as e:
    #    print(f"[pikahome][TTS] OpenAI TTS failed: {e!r}", file=sys.stderr)

    # ---------- Try 3: espeak-ng (offline, very robust) ----------
    # apt: sudo apt-get install -y espeak-ng
    #if shutil.which("espeak-ng"):
    #try:
            # espeak 'es' or 'es-la' (latam); voices vary, but es works everywhere
    #    tmp_wav = str(tmp)
    #    cmd = ["espeak-ng", "-v", TTS_LANG.replace("es-ES", "es"), "-w", tmp_wav, text]
    #    proc = await asyncio.create_subprocess_exec(*cmd)
    #    rc = await proc.wait()
    #    if rc == 0:
    #        return await _play_wav(tmp_wav, device=device)
    #    else:
    #        print(f"[pikahome][TTS] espeak-ng rc={rc}", file=sys.stderr)
    #except Exception as e:
    #    print(f"[pikahome][TTS] espeak-ng failed: {e!r}", file=sys.stderr)

    #print("[pikahome][TTS] All TTS backends failed; nothing spoken", file=sys.stderr)

async def _play_wav(path: str, device: str | None = None):
    cmd = ["aplay", "-q"]
    out_dev = device or OUTPUT_ALSA_DEVICE
    if out_dev:
        cmd += ["-D", out_dev]
    cmd += [path]
    try:
        proc = await asyncio.create_subprocess_exec(*cmd)
        await proc.wait()
    finally:
        with contextlib.suppress(Exception):
            Path(path).unlink(missing_ok=True)
# ============== Spotify helpers ==============
_sp = None
def get_spotify():
    global _sp
    if _sp: return _sp
    if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET):
        return None
    _sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
            redirect_uri=SPOTIFY_REDIRECT_URI,
            scope="user-read-playback-state user-modify-playback-state user-read-currently-playing",
            cache_path=str(Path.home()/".cache_spotify"),
            open_browser=False,   # headless
        ),
        requests_timeout=10,
        retries=3,
    )
    return _sp
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s.strip().lower())

def _smart_search(sp, query: str, market: str = "AR"):
    """
    Devuelve (kind, uri, title, extra) donde kind ∈ {'tracklist','playlist','album','track','episode','show'}
    - 'tracklist' = lista de URIs de tracks (p.ej. top tracks de un artista)
    """
    q = _norm(query)
    # patrones comunes en español
    is_playlist_hint = any(k in q for k in ["playlist", "lista"])
    is_album_hint    = any(k in q for k in ["album", "álbum", "disco"])
    is_radio_hint    = any(k in q for k in ["radio de", "parecido a", "similar a"])
    is_sleepy        = any(k in q for k in ["dormir", "relaj", "suave", "calma"])
    is_kids          = any(k in q for k in ["infantil", "niños", "ninos", "chicos", "peques"])
    is_stories       = any(k in q for k in ["cuento", "cuentos", "historia para dormir", "cuentos para dormir"])
    artist_of = None

    # “canciones de X”, “temas de X”, “poné a X”
    m = re.search(r"(canciones|temas|pone|poneme|poné|pone a|poneme a|poné a)\s+de?\s+(.+)$", q)
    if m:
        artist_of = m.group(2).strip()

    # 1) CUENTOS → shows/episodes primero
    if is_stories:
        res = sp.search(q=query, type="episode,show,playlist", limit=1, market=market)
        if res.get("episodes", {}).get("items"):
            e = res["episodes"]["items"][0]; return ("episode", e["uri"], e["name"], None)
        if res.get("shows", {}).get("items"):
            sh = res["shows"]["items"][0];  return ("show", sh["uri"], sh["name"], None)
        # si no, playlist temática
        if res.get("playlists", {}).get("items"):
            p = res["playlists"]["items"][0]; return ("playlist", p["uri"], p["name"], None)

    # 2) ARTISTA explícito → top tracks
    if artist_of:
        ares = sp.search(q=artist_of, type="artist", limit=1, market=market)
        if ares.get("artists", {}).get("items"):
            art = ares["artists"]["items"][0]
            tops = sp.artist_top_tracks(art["id"], country=market).get("tracks", [])
            if tops:
                uris = [t["uri"] for t in tops[:15]]
                title = f"Top de {art['name']}"
                return ("tracklist", uris, title, {"seed_artist": art["id"]})

    # 3) Radio / similares → recomendaciones por artista/track
    if is_radio_hint:
        # intentar extraer nombre luego de "radio de "
        m2 = re.search(r"radio de\s+(.+)$", q)
        seed_name = m2.group(1).strip() if m2 else q
        s = sp.search(q=seed_name, type="track,artist", limit=1, market=market)
        seed_art = s.get("artists", {}).get("items", [])
        seed_trk = s.get("tracks", {}).get("items", [])
        seeds = {}
        if seed_art: seeds["seed_artists"] = [seed_art[0]["id"]]
        if seed_trk: seeds["seed_tracks"]  = [seed_trk[0]["id"]]
        if seeds:
            recs = sp.recommendations(market=market, limit=20, **seeds).get("tracks", [])
            if recs:
                uris = [t["uri"] for t in recs]
                return ("tracklist", uris, f"Radio de {seed_name}", seeds)

    # 4) Pistas generales → si piden playlist/lista o moods, priorizamos playlist
    if is_playlist_hint or is_sleepy or is_kids:
        res = sp.search(q=query, type="playlist", limit=1, market=market)
        if res.get("playlists", {}).get("items"):
            p = res["playlists"]["items"][0]; return ("playlist", p["uri"], p["name"], None)

    # 5) Álbum explícito
    if is_album_hint:
        res = sp.search(q=query, type="album", limit=1, market=market)
        if res.get("albums", {}).get("items"):
            a = res["albums"]["items"][0]; return ("album", a["uri"], a["name"], None)

    # 6) Fallback: track → playlist → album
    res = sp.search(q=query, type="track,playlist,album", limit=1, market=market)
    if res.get("tracks", {}).get("items"):
        t = res["tracks"]["items"][0]; return ("track", t["uri"], t["name"], None)
    if res.get("playlists", {}).get("items"):
        p = res["playlists"]["items"][0]; return ("playlist", p["uri"], p["name"], None)
    if res.get("albums", {}).get("items"):
        a = res["albums"]["items"][0]; return ("album", a["uri"], a["name"], None)

    return (None, None, None, None)
def _pick_device(sp) -> str | None:
    try:
        devices = sp.devices() or {}
    except SpotifyOauthError:
        raise
    devs = devices.get("devices", [])
    if not devs: return None
    name = (SPOTIFY_DEVICE_NAME or "").lower()
    did = None
    active = None
    for d in devs:
        if d.get("is_active"):
            active = d.get("id")
        if name and d.get("name","").lower() == name:
            did = d.get("id")
    return did or active or devs[0].get("id")

def _is_playing_on(sp, device_id: str) -> bool:
    try:
        pb = sp.current_playback()
    except Exception:
        return False
    if not pb: return False
    if pb.get("device", {}).get("id") != device_id:
        return False
    return bool(pb.get("is_playing"))

def _nudge_and_verify(sp, device_id: str, delay: float = 0.9, tries: int = 2) -> bool:
    """Handle quirky states: transfer focus, try resume, then verify."""
    import time
    for _ in range(tries):
        time.sleep(delay)
        if _is_playing_on(sp, device_id):
            return True
        title = None
        try:
            sp.transfer_playback(device_id=device_id)   # no 'force' kwarg in new spotipy
        except Exception:
            pass
        try:
            sp.start_playback(device_id=device_id)      # resume context if any
        except Exception:
            pass
    return _is_playing_on(sp, device_id)
def spotify_device_id(sp):
    try:
        devs = sp.devices().get('devices', [])
        if not devs: return None
        for d in devs:
            if d.get("name","").lower() == SPOTIFY_DEVICE_NAME.lower():
                return d.get("id")
        for d in devs:
            if d.get("is_active"): return d.get("id")
        return devs[0].get("id")
    except Exception:
        return None

# ============== Agent definition (tools via decorators) ==============
def _log(*a):
    # tiny helper for quick visibility in journalctl
    try:
        print("[pikahome]", *a, flush=True)
    except Exception:
        pass

class PikaAgent(Agent):
    def __init__(self) -> None:
        # We’ll keep a minimal “awake” flag the model can toggle via a tool.
        self._awake = False
        super().__init__(
            instructions=(
                "Eres Pika, un amigo tipo Pikachu para niños pequeños. "
                "Habla en español (es-AR) con frases cortas y simples. "
                "No uses emojis. Usa 'pika' o 'pika-pi' ocasionalmente. "
                "No hables si no estás 'despertado'. Para despertarte, el humano dice 'hola pikachu'. "
                "Para dormirte, el humano dice 'chau pikachu'. Cuando escuches esas frases, usa la herramienta "
                "'set_awake' con awake=true o awake=false y responde brevemente. "
                "Usa la cámara SOLO si te lo piden explícitamen"
                "(por ejemplo: '¿qué ves?' o '¿qué Pokémon es este"
                "Para música, usa la herramienta play_music'."
                "Cuando uses la herramienta take_photo: "
                "- NO inventes. "
                "- Si el resultado tiene la clave 'assistant_response' o 'answer', RESPONDE EXACTAMENTE ese texto, sin agregar ni cambiar palabras. "
                "- Si 'confidence' es 'baja', pedí otra foto o más luz. "
                "- Si querés más detalles, pedilos en una ronda nueva, pero en la primera respuesta usá solo el texto del resultado. "
                "- Si usas take_photo, ANTES de hablar espera que termine y no inventes hasta tener el resultado. "
                "Si el niño pregunta otra cosa sobre la MISMA foto, usask_áabout_photo'. "
                "Evita temas de adultos; ofrece juegos, cuentos o ciencia sencilla."
                "Para música: "
                "- 'poné...', 'quiero escuchar ...' -> usa play_music. "
                "- 'basta', 'pará', 'pausá-> usa pause_music. "
                "- 'seguí', 'reanudar-> usa resume_music. "
                "- 'cambiá de canción','poné otra', 'siguient-> usa next_track. "
                "- 'volumen al 50%' -> usa set_volume(50). "
                "- 'mezclá', 'modo aleatorio'-> usa set_shuffle(true); "
                "  'sacá el aleatorio' -> usa set_shuffle(false). "
                "- 'repetir canción'-> usa set_repeat('track'); "
                "  'sacá repetir' ->  usa set_repeat('off'). "
                )
            
        )
        self._camera_lock = asyncio.Lock()
        self._last_photo_path: Path | None = None
        self._last_data_url: str | None = None

    @staticmethod
    def _extract_answer_text(result: dict) -> str:
        if not isinstance(result, dict):
            return ""
        for key in ("answer", "assistant_response", "response", "caption", "description"):
            value = result.get(key)
            if not value:
                continue
            if isinstance(value, str):
                text = value.strip()
            else:
                text = str(value).strip()
            if text:
                return text
        return ""

    async def _say_via_session(
        self,
        session,
        text: str,
        *,
        allow_fallback: bool = False,
        context: RunContext | None = None,
        await_playout: bool = False,
    ) -> None:
        """Speak using the realtime voice, falling back to local TTS if allowed."""
        text = (text or "").strip()
        if not text:
            return
        if session is None:
            _log("skip speech (no active session)")
            if context is not None:
                await self._say_via_generate_reply(context, text, await_playout=await_playout)
            return
        try:
            handle = session.say(text)
        except Exception as exc:
            _log("session.say failed", exc)
            if context is not None:
                await self._say_via_generate_reply(context, text, await_playout=await_playout)
            elif allow_fallback and USE_LOCAL_TTS_FALLBACK:
                await _say_local_exact(text)
            return

        wait_coro = getattr(handle, "wait_for_playout", None)
        if callable(wait_coro):
            coro = wait_coro()
            if await_playout:
                try:
                    await asyncio.wait_for(asyncio.shield(coro), timeout=10.0)
                except Exception:
                    pass
            else:
                async def _wait() -> None:
                    try:
                        await asyncio.shield(coro)
                    except Exception:
                        pass

                asyncio.create_task(_wait())

    async def _say_via_generate_reply(
        self,
        context: RunContext | None,
        text: str,
        *,
        await_playout: bool = False,
    ) -> None:
        text = (text or "").strip()
        if not text or context is None:
            return
        session = getattr(context, "session", None)
        if session is None:
            return
        await self._wait_for_active_playout(context, timeout=10.0)
        try:
            handle = session.generate_reply(
                instructions=(
                    "Decí exactamente el siguiente texto y no agregues nada más:\n"
                    f"{text}"
                ),
                tool_choice="none",
            )
        except Exception as exc:
            _log("generate_reply failed", exc)
            return

        wait_coro = getattr(handle, "wait_for_playout", None)
        if callable(wait_coro):
            coro = wait_coro()
            if await_playout:
                try:
                    await asyncio.wait_for(asyncio.shield(coro), timeout=10.0)
                except Exception:
                    pass
            else:
                async def _wait() -> None:
                    try:
                        await asyncio.shield(coro)
                    except Exception:
                        pass

                asyncio.create_task(_wait())

    async def _wait_for_active_playout(
        self,
        context: RunContext | None,
        *,
        timeout: float = 5.0,
    ) -> None:
        if context is None:
            return
        waiter = getattr(context, "wait_for_playout", None)
        if not callable(waiter):
            return
        try:
            await asyncio.wait_for(waiter(), timeout=timeout)
        except Exception:
            pass
    # Tiny state toggle the model can call when it hears the wake/stop words.
    @function_tool(description="Activa o desactiva el modo 'despierto' del asistente.")
    async def set_awake(self, context: RunContext, awake: bool) -> str:
        was = self._awake
        self._awake = bool(awake)
        session = getattr(context, "session", None)
        if self._awake and not was:
            await self._say_via_session(
                session,
                "¡Pika! ¿En qué te ayudo?",
                allow_fallback=True,
                context=context,
            )
        elif (not self._awake) and was:
            await self._say_via_session(
                session,
                "Hasta luego. Pika.",
                allow_fallback=True,
                context=context,
            )
        return f"awake={self._awake}"

    @function_tool(
        description=("Reproduce música en Spotify según la búsqueda indicada. "
                 "Ej: 'poné canciones de Tini', 'canciones para dormir', "
                 "'poné playlist X', 'poné el álbum Y', 'radio de Shakira', 'cuentos para dormir'.")
    )
    async def play_music(self, context: RunContext, query: str) -> str:
        if not self._awake:
            return "Estoy dormido por ahora."
        sp = get_spotify()
        if not sp:
            return "Spotify no está configurado."

        try:
            did = _pick_device(sp)
        except SpotifyOauthError:
            return SPOTIFY_AUTH_ERROR_MSG
        if not did:
            return ("No veo dispositivos de Spotify. Abrí Spotify en tu celu/PC, "
                "elegí 'pikahome' y volvé a pedirme música.")

        try:
            # foco + volumen
            try: sp.transfer_playback(device_id=did)
            except Exception: pass
            try: sp.volume(75, device_id=did)
            except Exception: pass

            kind, uri_or_uris, title, _ = _smart_search(sp, query, market=os.getenv("SPOTIFY_MARKET","AR"))
            if not kind:
                return "No encontré eso en Spotify."

            # ejecutar según el tipo
            try:
                if kind == "tracklist":
                    sp.start_playback(device_id=did, uris=uri_or_uris)
                elif kind in ("playlist", "album", "show"):
                    sp.start_playback(device_id=did, context_uri=uri_or_uris)
                elif kind in ("track", "episode"):
                    sp.start_playback(device_id=did, uris=[uri_or_uris])
                else:
                    sp.start_playback(device_id=did)  # reanudar contexto si quedara algo
            except Exception:
                # pequeño wake-up y reintento
                try:
                    sp.transfer_playback(device_id=did)
                    sp.start_playback(device_id=did)
                    if kind == "tracklist":
                        sp.start_playback(device_id=did, uris=uri_or_uris)
                    elif kind in ("playlist", "album", "show"):
                        sp.start_playback(device_id=did, context_uri=uri_or_uris)
                    elif kind in ("track", "episode"):
                        sp.start_playback(device_id=did, uris=[uri_or_uris])
                except Exception:
                    pass
        except SpotifyOauthError:
            return SPOTIFY_AUTH_ERROR_MSG

        if _nudge_and_verify(sp, did):
            nice = title or query
            return f"Reproduciendo {nice} en Spotify."
        else:
            return ("Intenté reproducir pero no empezó. En tu Spotify elegí 'pikahome', "
                "subí el volumen, y probemos de nuevo.")

    @function_tool(description="Pausa la música en Spotify (ej: 'basta', 'pará').")
    async def pause_music(self, context: RunContext) -> str:
        sp = get_spotify()
        if not sp: return "Spotify no está configurado."
        try:
            did = _pick_device(sp)
        except SpotifyOauthError:
            return SPOTIFY_AUTH_ERROR_MSG
        if not did: return "No veo el parlante 'pikahome'."
        try:
            sp.pause_playback(device_id=did)
            return "Listo, paro la música."
        except SpotifyOauthError:
            return SPOTIFY_AUTH_ERROR_MSG
        except Exception:
            return "No pude pausar. Probemos de nuevo."

    @function_tool(description="Reanuda la música en Spotify (ej: 'seguí', 'reanudar').")
    async def resume_music(self, context: RunContext) -> str:
        sp = get_spotify()
        if not sp: return "Spotify no está configurado."
        try:
            did = _pick_device(sp)
        except SpotifyOauthError:
            return SPOTIFY_AUTH_ERROR_MSG
        if not did: return "No veo el parlante 'pikahome'."
        try:
            sp.start_playback(device_id=did)  # resume current context
            return "Sigo con la música."
        except SpotifyOauthError:
            return SPOTIFY_AUTH_ERROR_MSG
        except Exception:
            return "No pude seguir. Probemos de nuevo."

    @function_tool(description="Pasa a la siguiente canción (ej: 'cambiá de canción', 'poné otra').")
    async def next_track(self, context: RunContext) -> str:
        sp = get_spotify()
        if not sp: return "Spotify no está configurado."
        try:
            did = _pick_device(sp)
        except SpotifyOauthError:
            return SPOTIFY_AUTH_ERROR_MSG
        if not did: return "No veo el parlante 'pikahome'."
        try:
            sp.next_track(device_id=did)
            return "Listo, cambio de canción."
        except SpotifyOauthError:
            return SPOTIFY_AUTH_ERROR_MSG
        except Exception:
            return "No pude cambiar. Probemos de nuevo."

    @function_tool(description="Vuelve a la canción anterior.")
    async def previous_track(self, context: RunContext) -> str:
        sp = get_spotify()
        if not sp: return "Spotify no está configurado."
        try:
            did = _pick_device(sp)
        except SpotifyOauthError:
            return SPOTIFY_AUTH_ERROR_MSG
        if not did: return "No veo el parlante 'pikahome'."
        try:
            sp.previous_track(device_id=did)
            return "Vuelvo a la anterior."
        except SpotifyOauthError:
            return SPOTIFY_AUTH_ERROR_MSG
        except Exception:
            return "No pude volver. Probemos de nuevo."

    @function_tool(description="Ajusta el volumen (0 a 100).")
    async def set_volume(self, context: RunContext, volume: int) -> str:
        sp = get_spotify()
        if not sp: return "Spotify no está configurado."
        try:
            did = _pick_device(sp)
        except SpotifyOauthError:
            return SPOTIFY_AUTH_ERROR_MSG
        if not did: return "No veo el parlante 'pikahome'."
        vol = max(0, min(100, int(volume)))
        try:
            sp.volume(vol, device_id=did)
            return f"Volumen a {vol}%."
        except SpotifyOauthError:
            return SPOTIFY_AUTH_ERROR_MSG
        except Exception:
            return "No pude cambiar el volumen."

    @function_tool(description="Activa o desactiva el modo aleatorio (shuffle).")
    async def set_shuffle(self, context: RunContext, enabled: bool) -> str:
        sp = get_spotify()
        if not sp: return "Spotify no está configurado."
        try:
            did = _pick_device(sp)
        except SpotifyOauthError:
            return SPOTIFY_AUTH_ERROR_MSG
        if not did: return "No veo el parlante 'pikahome'."
        try:
            sp.shuffle(enabled, device_id=did)
            return "Mezclo las canciones." if enabled else "Saco el modo aleatorio."
        except SpotifyOauthError:
            return SPOTIFY_AUTH_ERROR_MSG
        except Exception:
            return "No pude cambiar el modo aleatorio."

    @function_tool(description="Configura el modo repetir: 'track', 'context' o 'off'.")
    async def set_repeat(self, context: RunContext, mode: str) -> str:
        sp = get_spotify()
        if not sp: return "Spotify no está configurado."
        try:
            did = _pick_device(sp)
        except SpotifyOauthError:
            return SPOTIFY_AUTH_ERROR_MSG
        if not did: return "No veo el parlante 'pikahome'."
        mode = (mode or "off").lower()
        if mode not in ("track","context","off"):
            mode = "off"
        try:
            sp.repeat(mode, device_id=did)
            return f"Repito: {mode}."
        except SpotifyOauthError:
            return SPOTIFY_AUTH_ERROR_MSG
        except Exception:
            return "No pude cambiar el modo repetir."

    @function_tool(
        description=(
            "Saca una foto y la analiza. Sirve para preguntas como: "
            "'¿qué Pokémon es este?', '¿qué hay en mi remera?', '¿qué es el muñeco del fondo?', "
            "'¿qué objetos hay en la mesa?'. Acepta parámetro 'question' (opcional)."
        )
    )
    async def take_photo(self, context: RunContext, question: str | None = None) -> dict:
        if not self._awake:
            return {"error": "Estoy dormido por ahora."}
        await self._wait_for_active_playout(context)
        session = getattr(context, "session", None)
        await self._say_via_session(
            session,
            "Mmm... dejame ver un segundito.",
            context=context,
            await_playout=True,
        )
        async with self._camera_lock:
            tmp = Path("/tmp") / f"pika_{int(datetime.datetime.now().timestamp())}.jpg"

            def snap() -> None:
                cmd = [
                    "fswebcam", "-d", "/dev/video0",
                    "-r", f"{PHOTO_W}x{PHOTO_H}", "--no-banner", str(tmp)
                ]
                subprocess.run(cmd, check=True)

            # captura con reintento
            try:
                await asyncio.to_thread(snap)
            except Exception:
                import time
                await asyncio.sleep(0.3)
                await asyncio.to_thread(snap)

            img_bytes = await asyncio.to_thread(tmp.read_bytes)
            if SAVE_DIRS:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out = Path(SAVE_DIRS[0]).expanduser() / f"photo_{ts}.jpg"
                with contextlib.suppress(Exception):
                    out.write_bytes(img_bytes)

            data_url = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("ascii")
            self._last_photo_path = tmp
            self._last_data_url = data_url

            try:
                result = await self._vision_analyze(data_url, question)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as e:
                result = {
                    "answer": f"No pude analizar bien la imagen ({e}). Probemos con más luz y acercando el objeto.",
                    "objects": [], "pokemon": None, "text": "", "brands": [], "scene_tags": [], "confidence": "baja",
                }

        speak = self._extract_answer_text(result)
        if not speak:
            speak = "No pude analizar bien la imagen. Probemos con más luz y acercando el objeto."
        if isinstance(result, dict):
            result["answer"] = speak
            result["assistant_response"] = speak

        await self._say_via_session(session, speak, context=context, await_playout=True)
        # respuesta pensada para *hablar*
        return {
            "assistant_response": speak,
            "answer": speak,
            "objects": result.get("objects", []),
            "pokemon": result.get("pokemon"),
            "text": result.get("text", ""),
            "brands": result.get("brands", []),
            "scene_tags": result.get("scene_tags", []),
            "confidence": result.get("confidence", "baja"),
            "image_url": data_url,
            "width": PHOTO_W,
            "height": PHOTO_H,
        }
    @function_tool(
        description=(
            "Responde una nueva pregunta sobre la *última foto* tomada, sin volver a sacar otra."
            )

    )
    async def ask_about_photo(self, context: RunContext, question: str) -> dict:
        if not self._awake:
            return {"error": "Estoy dormido por ahora."}
        if not self._last_data_url:
            return {"error": "No tengo una foto reciente. Decime: 'sacá una foto'."}
        await self._wait_for_active_playout(context)

        try:
            result = await self._vision_analyze(self._last_data_url, question)
            if inspect.isawaitable(result):
                result = await result
        except Exception as e:
            result = {
                "answer": f"No pude analizar bien la imagen ({e}). Probemos con más luz y acercando el objeto.",
                "objects": [], "pokemon": None, "text": "", "brands": [], "scene_tags": [], "confidence": "baja",
            }
        speak = self._extract_answer_text(result)
        if not speak:
            speak = "No pude analizar bien la imagen. Probemos con más luz y acercando el objeto."
        result["answer"] = speak
        result["assistant_response"] = speak
        session = getattr(context, "session", None)
        await self._say_via_session(session, speak, context=context, await_playout=True)
        return {
            "assistant_response": speak,
            "answer": speak,
            "objects": result.get("objects", []),
            "pokemon": result.get("pokemon"),
            "text": result.get("text", ""),
            "brands": result.get("brands", []),
            "scene_tags": result.get("scene_tags", []),
            "confidence": result.get("confidence", "baja"),
            "image_url": self._last_data_url,
            "width": PHOTO_W,
            "height": PHOTO_H,
        }

    def _vision_analyze_sync(self, data_url: str, question: str | None) -> dict:
        client = OpenAI(api_key=OPENAI_API_KEY)
        q = question or "¿Qué ves? Nombra objetos, personajes y texto importante."
        system = (
            "Sos un asistente de visión para un niño (es-AR). "
            "Responde breve y amable. Devolvé JSON con llaves: "
            "answer (string), objects (lista de strings), text (string OCR), "
            "brands (lista strings), scene_tags (lista strings), confidence ('alta'|'media'|'baja')."
        )

        resp = client.chat.completions.create(
            model=VISUAL_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": q},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )

        content = resp.choices[0].message.content.strip()
        try:
            parsed = json.loads(content)
            for k in ["answer","objects","text","brands","scene_tags","confidence"]:
                parsed.setdefault(k, [] if k in ("objects","brands","scene_tags") else "")
            return parsed
        except Exception:
            return {
                "assistant_response":content,
                "answer": content,
                "objects": [],
                "text": "",
                "brands": [],
                "scene_tags": [],
                "confidence": "media",
            }

    async def _vision_analyze(self, data_url: str, question: str | None) -> dict:
        """Runs the blocking vision request off the event loop."""
        return await asyncio.to_thread(self._vision_analyze_sync, data_url, question)
    # version-agnostic "stop talking now" helper
    async def _interrupt_safely(self, session):
        """
        Interrupt any ongoing realtime response/speech, across LK Agents variants.
        Tries public methods first; falls back to the internal activity.
        Never raises if the method is missing.
        """
        try:
            m = getattr(session, "interrupt_response", None)
            if callable(m):
                await m(); return
            m = getattr(session, "interrupt", None)
            if callable(m):
                await m(); return
            act = getattr(session, "_activity", None)
            if act is not None:
                m = getattr(act, "interrupt", None)
                if callable(m):
                    # some builds accept no args; some accept a reason
                    if len(inspect.signature(m).parameters) == 0:
                        await m()
                    else:
                        await m("tool_override")
                    return
        except Exception:
            pass
# ============== Entrypoint ==============
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    # Realtime model handles STT + LLM + TTS in one go.
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            model=REALTIME_MODEL,
            voice=OPENAI_VOICE,
            turn_detection=TurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=500,
                create_response=True,
                interrupt_response=True,
            )
            # You can configure turn detection via the plugin if you want;
            # OpenAI Realtime has built-in VAD/turn-detection by default.
        ),
    )

    await session.start(
        room=ctx.room,
        agent=PikaAgent(),
        room_input_options=RoomInputOptions(),  # add noise cancellation if you enable that plugin   
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
